import os
import time
import logging
import copy

from functools import reduce
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from tqdm import trange

from scipy.interpolate import griddata
from matplotlib.path import Path as mpath

# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ioff()  # Turn interactive plotting off


# Local modules
from utils import  get_cpu_memory_usage, get_cuda_memory_usage, plot_magnitude_and_save_absolute_error, craft_bc_and_ic_dataset, craft_validation_dataset, get_coordinates_for_generator,  from_time_to_index, build_stratum_dataset

class NullContainer:
    def __getitem__(self, key):
        # You can return a default value or even self to allow chaining.
        return None
    

def exact_solution(x, t, c):
    return  np.sin(x)*np.cos(c*t) - np.sin(c*t)*np.cos(x)

def activation_func(case):

    if case == 'swish':
        activation =  torch.nn.SiLU()
    if case == 'tanh':
        activation =  torch.nn.Tanh()
    if case == 'gelu':
        activation = torch.nn.GELU()
    
    return activation

def metric_l2_relative_error(exact, pred):
    return torch.linalg.norm((exact-pred), 2)/torch.linalg.norm(exact, 2)

def metric_l2_absolute_error(exact, pred):
    return torch.linalg.norm((exact-pred), 2)

class FourierEmbedding(nn.Module):
    
    def __init__(self, fourier_emb, original_in_dim, device):

        super().__init__()
        
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        embed_dim = fourier_emb['embed_dim']

        kernel_shape = (original_in_dim, embed_dim)
        stddev = fourier_emb['stddev']

        kernel = torch.randn(kernel_shape) * stddev

        self.kernel = kernel.to(device)
    
    def __call__(self, x):
        
        # Compute the Fourier embeddings
        xp = torch.matmul(x, self.kernel)
        # torch.cat((torch.cos(2*self.pi*xp), torch.sin(2*self.pi*xp)), dim=1)
        return torch.cat((torch.cos(2*self.pi*xp), torch.sin(2*self.pi*xp)), dim=len(xp.shape) - 1)
    
class EncoderDense_DeepONet(nn.Module):
    
    def __init__(self, config, in_dim, network_component):
        
        super().__init__()

        # Activation function
        self.activation_encoder =  activation_func(config[network_component]['neuralNet']['activation'])
        self.random_weight_fact = config['random_weight_fact']
        self.has_random_weight_fact = config['random_weight_fact']['enabled']
        
        # Architecture
        num_layers = 1
        hidden_dim = config[network_component]['neuralNet']['hidden_dim']
        out_dim = config[network_component]['neuralNet']['out_dim']

        hidden_layers = num_layers*[hidden_dim]

        if self.has_random_weight_fact:

            layer_shapes= []
            shape0=(in_dim, hidden_layers[0])
            layer_shapes.append(shape0)
            
            # Random weight factorization
            def factorized_glorot_normal(shape, mean=1.0, stddev=0.1):
                w = torch.nn.init.xavier_normal_(torch.empty(shape[1],shape[0]))
                s = mean + torch.randn(shape[0]) * stddev
                s = torch.exp(s)
                # scale = torch.unsqueeze(s,0)
                v = w / s
                return s, v
            
            # Update only W
            mean = self.random_weight_fact['mean']
            stddev = self.random_weight_fact['stddev']

            # Initialize parameters for all layers using factorized_glorot_normal
            self.s_rwf = nn.ParameterList()
            self.v_rwf = nn.ParameterList()
            self.b_rwf = nn.ParameterList()

            for i,shape in enumerate(layer_shapes):
                s, v = factorized_glorot_normal(shape, mean, stddev)
                self.s_rwf.append(nn.Parameter(s))
                self.v_rwf.append(nn.Parameter(v))
                self.b_rwf.append(nn.Parameter(torch.zeros(shape[1])))
        
        else:   
            
            core_net = []
            # Input
            core_net.append(nn.Linear(in_dim, hidden_layers[0]))
            core_net.append(self.activation_encoder)
            self.net = nn.Sequential(*core_net)
            
            # Xavier normal initialization  (also known as Glorot scheme)
            if config[network_component]['neuralNet']['xavier_init']:
                if isinstance(self.net[0],nn.Linear):
                    nn.init.xavier_normal_(self.net[0].weight.data)
                    nn.init.zeros_(self.net[0].bias.data)
       
    def __call__(self, x):
        
        if self.has_random_weight_fact:
            for i in range(1):
                kernel = self.s_rwf[i] * self.v_rwf[i]
                x = torch.matmul(x, torch.transpose(kernel,0,1)) + self.b_rwf[i]
                x = self.activation_encoder(x)
        else:
            x = self.net(x)

        return x

class StandardDense_DeepONet(nn.Module):
    
    def __init__(self, config, in_dim, network_component):
        
        super().__init__()

        # Activation function
        self.architecture = config[network_component]['neuralNet']['architecture']
        self.activation =  activation_func(config[network_component]['neuralNet']['activation'])
        self.random_weight_fact = config['random_weight_fact']
        self.has_random_weight_fact = config['random_weight_fact']['enabled']

        # Architecture
        self.num_layers = config[network_component]['neuralNet']['num_layers']
        hidden_dim = config[network_component]['neuralNet']['hidden_dim']
        out_dim = config[network_component]['neuralNet']['out_dim']

        hidden_layers = self.num_layers*[hidden_dim]


        if self.has_random_weight_fact:
            # Random weight factorization
            layer_shapes= []
            shape0=(in_dim, hidden_layers[0])
            layer_shapes.append(shape0)
            for i in range(self.num_layers-1):
                shapei=(hidden_layers[i], hidden_layers[i+1])
                layer_shapes.append(shapei)
            shapeLast = (hidden_layers[-1], out_dim)
            layer_shapes.append(shapeLast)
            
            def factorized_glorot_normal(shape, mean=1.0, stddev=0.1):
                w = torch.nn.init.xavier_normal_(torch.empty(shape[1],shape[0]))
                s = mean + torch.randn(shape[0]) * stddev
                s = torch.exp(s)
                v = w / s
                return s, v
            
            # Update only W
            mean = self.random_weight_fact['mean']
            stddev = self.random_weight_fact['stddev']

            # Initialize parameters for all layers using factorized_glorot_normal
            self.s_rwf = nn.ParameterList()
            self.v_rwf = nn.ParameterList()
            self.b_rwf = nn.ParameterList()

            for i,shape in enumerate(layer_shapes):
                s, v = factorized_glorot_normal(shape, mean, stddev)
                self.s_rwf.append(nn.Parameter(s))
                self.v_rwf.append(nn.Parameter(v))
                self.b_rwf.append(nn.Parameter(torch.zeros(shape[1])))
            
        else:
            
            core_net = []
            # Input
            core_net.append(nn.Linear(in_dim, hidden_layers[0]))
            core_net.append(self.activation)
            # Hidden layers
            for i in range(self.num_layers-1):
                core_net.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                core_net.append(self.activation)
            # Output
            core_net.append(nn.Linear(hidden_layers[-1], out_dim))
            self.net = nn.Sequential(*core_net)
            
            # Xavier normal initialization  (also known as Glorot scheme)
            if config[network_component]['neuralNet']['xavier_init']:
                for i in range(self.num_layers+1):
                    if isinstance(self.net[2*i],nn.Linear):
                        nn.init.xavier_normal_(self.net[2*i].weight.data)
                        nn.init.zeros_(self.net[2*i].bias.data)
    
    def __call__(self, x):
        
        if self.has_random_weight_fact and self.architecture == 'MLP':
            
            for i in range(self.num_layers+1):
                kernel = self.s_rwf[i] * self.v_rwf[i]
                x = torch.matmul(x, torch.transpose(kernel,0,1)) + self.b_rwf[i]
                x = self.activation(x)
            
        else:
            x = self.net(x)
        
        return x
    
class  MultiLayerPerceptron_DeepONet (nn.Module):

    def __init__(self, config, network_component):

        super().__init__()

        device = config['device']
        in_dim = config[network_component]['neuralNet']['in_dim']
        self.random_weight_fact = config['random_weight_fact']
        self.has_random_weight_fact = config['random_weight_fact']['enabled']
        self.num_layers = config[network_component]['neuralNet']['num_layers']
       
        # Fourier embedding 
        self.fourier_emb = config['fourier_emb']
        self.has_fourier_emb = config['fourier_emb']['enabled']
        
        if self.has_fourier_emb:
            in_dim = 2 * self.fourier_emb['embed_dim']
            self.call_fourier_embedding = FourierEmbedding(self.fourier_emb, config[network_component]['neuralNet']['in_dim'], device)

        # Architecture initialization
        self.standard_dense = StandardDense_DeepONet(config, in_dim, network_component)

        # Manually defined parameters
        self.param_groups = []
        for i in range(self.num_layers+1):
            if self.has_random_weight_fact:
                self.param_groups.append({'params': [self.standard_dense.s_rwf[i]]})
                self.param_groups.append({'params': [self.standard_dense.v_rwf[i]]})
                self.param_groups.append({'params': [self.standard_dense.b_rwf[i]]})
            else:
                self.param_groups.append({'params': [self.standard_dense.net[2*i].weight]})
                self.param_groups.append({'params': [self.standard_dense.net[2*i].bias]})


    def forward(self, x):

        # Fourier features embedding 
        if self.has_fourier_emb:
            x = self.call_fourier_embedding(x)

        # Forward pass
        NN_pred = self.standard_dense(x)

        return NN_pred

class ModifiedMultiLayerPerceptron_DeepONet (nn.Module):

    def __init__(self, config, network_component):

        super().__init__()


        device = config['device']
        in_dim = config[network_component]['neuralNet']['in_dim']
        self.random_weight_fact = config['random_weight_fact']
        self.has_random_weight_fact = config['random_weight_fact']['enabled']
        self.num_layers = config[network_component]['neuralNet']['num_layers']

        # Fourier embedding 
        self.fourier_emb = config['fourier_emb']
        self.has_fourier_emb = config['fourier_emb']['enabled']
        
        if self.has_fourier_emb:
            in_dim = 2 * self.fourier_emb['embed_dim']
            self.call_fourier_embedding = FourierEmbedding(self.fourier_emb, config[network_component]['neuralNet']['in_dim'], device)

       
        # Architectures initialization
        self.encoder_dense1 = EncoderDense_DeepONet(config, in_dim, network_component) 
        self.encoder_dense2 = EncoderDense_DeepONet(config, in_dim, network_component)
        self.standard_dense = StandardDense_DeepONet(config, in_dim, network_component)

        # Manually define parameters
        self.param_groups = []
        for i in range(self.num_layers+1):
            if self.has_random_weight_fact:
                self.param_groups.append({'params': [self.standard_dense.s_rwf[i]]})
                self.param_groups.append({'params': [self.standard_dense.v_rwf[i]]})
                self.param_groups.append({'params': [self.standard_dense.b_rwf[i]]})
            else:
                self.param_groups.append({'params': [self.standard_dense.net[2*i].weight]})
                self.param_groups.append({'params': [self.standard_dense.net[2*i].bias]})
        
        if self.has_random_weight_fact:
            self.param_groups.append({'params': [self.encoder_dense1.s_rwf[0]]})
            self.param_groups.append({'params': [self.encoder_dense1.v_rwf[0]]})
            self.param_groups.append({'params': [self.encoder_dense1.b_rwf[0]]})
        else:
            self.param_groups.append({'params': [self.encoder_dense1.net[0].weight]})
            self.param_groups.append({'params': [self.encoder_dense1.net[0].bias]})
        
        if self.has_random_weight_fact:
            self.param_groups.append({'params': [self.encoder_dense2.s_rwf[0]]})
            self.param_groups.append({'params': [self.encoder_dense2.v_rwf[0]]})
            self.param_groups.append({'params': [self.encoder_dense2.b_rwf[0]]})
        else:
            self.param_groups.append({'params': [self.encoder_dense2.net[0].weight]})
            self.param_groups.append({'params': [self.encoder_dense2.net[0].bias]})
        
        
    def forward(self, x):

        # Fourier features embedding 
        if self.has_fourier_emb:
            x = self.call_fourier_embedding(x)


        if self.has_random_weight_fact:
            
            # Forward
            U = self.encoder_dense1(x)
            V = self.encoder_dense2(x)
            
            kernel = self.standard_dense.s_rwf[0] * self.standard_dense.v_rwf[0]
            x = torch.matmul(x, torch.transpose(kernel,0,1)) + self.standard_dense.b_rwf[0]
            H1 = self.standard_dense.activation(x)
            
            for i in range(1, self.num_layers):

                kernel = self.standard_dense.s_rwf[i] * self.standard_dense.v_rwf[i]
                H = torch.matmul(H1, torch.transpose(kernel,0,1)) + self.standard_dense.b_rwf[i]
                H = self.standard_dense.activation(H)

                H1 = H * U + (1 - H) * V


            kernel = self.standard_dense.s_rwf[-1] * self.standard_dense.v_rwf[-1]
            H1 = torch.matmul(H1, torch.transpose(kernel,0,1)) + self.standard_dense.b_rwf[-1]
        
        else:  

            # Forward
            U = self.encoder_dense1(x)
            V = self.encoder_dense2(x)
            
            H1 = self.standard_dense.net[1](self.standard_dense.net[0](x))
            
            for i in range(1, self.num_layers):
                H = self.standard_dense.net[2*i+1](self.standard_dense.net[2*i](H1))
                H1 = H * U + (1 - H) * V

            H1 = self.standard_dense.net[-1](H1)

        return H1




#--------------------------
class NeuralNetwork(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.config = config
        
        # NN Architectures
        self.branch_list = []
        for branch_ID in config['branches_control']['branch_list_ID']:
       
            # Branch
            if config[branch_ID]['neuralNet']['architecture'] == "Modified MLP":
            
                branch_i = ModifiedMultiLayerPerceptron_DeepONet(config, network_component = branch_ID)
                self.branch_list.append(branch_i.to(self.config['device']))
            
            else: #"MLP":
                
                branch_i = MultiLayerPerceptron_DeepONet(config, network_component = branch_ID)
                self.branch_list.append(branch_i.to(self.config['device']))

            self.config['logger'].info(f'branch_component: {branch_i}')

        # Trunk
        if config['trunk']['neuralNet']['architecture'] == "Modified MLP":
        
            self.trunk = ModifiedMultiLayerPerceptron_DeepONet(config, network_component = 'trunk')
        
        else: #"MLP":
            
            self.trunk = MultiLayerPerceptron_DeepONet(config, network_component = 'trunk')
        

        self.config['logger'].info(f'trunk_component: {self.trunk}')

    
class Trainer_PIDeepONetLdata(nn.Module):
    
    def __init__(self, config, model):
        
        super().__init__()
        
        self.config = config
        self.has_random_weight_fact = self.config['random_weight_fact']['enabled']
        self.has_loss_balancing = self.config['loss_balancing']['enabled']
        self.has_exponential_decay = self.config['optim1']['exponential_decay']['enabled']

        # Initialize the SummaryWriter
        if self.config['mode']=='train':
            self.writer = SummaryWriter(log_dir=config['logs_folder_path'])

        self.model = model
        self.branch_out_dim = config['branch1']['neuralNet']['out_dim']
        self.trunk_in_dim = config['trunk']['neuralNet']['in_dim']
        self.output_dim = 1

        # PDE parameters
        self.c_dict = self.config['pde_param']['c']

         # Optimizers
        if config['optim1']['optimizer'] == 'Adam':

            # Retrieve parameter groups from each model
            param_groups_branch_sum = []
            for branch in self.model.branch_list:
                param_groups_branch_i = list(branch.param_groups)
                param_groups_branch_sum += param_groups_branch_i
            
            param_groups_trunk = list(self.model.trunk.param_groups)

            # Combine the parameter groups into a single list
            combined_param_groups = param_groups_branch_sum + param_groups_trunk

            self.optimizer_Adam = torch.optim.Adam(combined_param_groups, lr=config['optim1']['learning_rate'], \
                                                    betas=(config['optim1']['beta1'], config['optim1']['beta2']), \
                                                    eps=config['optim1']['eps'] )
            
            if self.has_exponential_decay:
                def lr_lambda_func(current_step):
                    """Custom function for the exponential decay."""
                    
                    decay_rate = self.config['optim1']['exponential_decay']['decay_rate']
                    decay_steps = self.config['optim1']['exponential_decay']['decay_steps']

                    factor = decay_rate ** (current_step / decay_steps)

                    return factor

                self.scheduler_Adam = torch.optim.lr_scheduler.LambdaLR(self.optimizer_Adam, lr_lambda_func)
        
        if config['optim2']['optimizer'] == 'LBFGS':

            # Extract parameters from custom parameter groups
            all_params = [param for group in self.model.branch_list[0].param_groups + self.model.trunk.param_groups for param in group['params']]
            # all_params = [param for group in self.model.branch_list[0].param_groups + self.model.branch_list[1].param_groups + self.model.trunk.param_groups for param in group['params']]
            # all_params = [param for group in self.model.mlp.param_groups for param in group['params']]
            
            self.optimizer_LBFGS = torch.optim.LBFGS(all_params, max_iter=config['optim2']['max_iter'], max_eval=config['optim2']['max_eval'], tolerance_grad=1.0 * torch.finfo(torch.float32).eps, history_size=50)

        
        # Loss function
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        

        # Logging
        self.total_loss = None
        self.l2_error_u = None

        # Loss terms
        self.term_loss_tensor_dict = {key:[] for key in self.config['loss_terms']}

        # lambdas initialization
        self.term_lambdas_tensor = torch.tensor(
                                    [self.config['lambda_weights_init'][i] for i in range(len(self.config['loss_terms']))], 
                                    dtype=torch.float32, requires_grad=False).to(self.config['device'])

        self.term_lambdas_tensor_dict = {key: self.term_lambdas_tensor[i] for i,key in enumerate(self.config['loss_terms'])}

        self.log_log = """LOG\n"""
        self.min_error_for_checkpoint = np.Infinity
        self.runtime0 = self.config['tick_start']
        self.best_iter = 0

    def loss_data(self):

        tau = []
        for i, f_data_tensor_i in enumerate(self.f_data_tensor):
            tau.append(self.model.branch_list[i](f_data_tensor_i))
 
        beta = self.model.trunk(self.xt_data_tensor)

        ## np.tile
        tau[0] = tau[0].reshape(-1, self.branch_out_dim)

        # Predicted
        u_pred = torch.sum(tau[0] * beta, axis = 1)[:, None]
       

        # Ground Trust
        u_GT = self.u_data_tensor
        
        # Compute losses
        u_loss = self.loss_fn(u_pred, u_GT)

    
        return u_loss
    
    def loss_initial_condition(self):

        tau = []
        for i, f_ic_tensor_i in enumerate(self.f_ic_tensor):
            tau.append(self.model.branch_list[i](f_ic_tensor_i))
 
        beta = self.model.trunk(self.xt_ic_tensor)

        ## np.tile
        tau[0] = tau[0].reshape(-1, self.branch_out_dim)

        # Predicted
        u_pred = torch.sum(tau[0] * beta, axis = 1)[:, None]
       

        # Ground Trust
        u_GT = self.u_ic_tensor
        
        # Compute losses
        u_loss = self.loss_fn(u_pred, u_GT)

    
        return u_loss
    
    def loss_physics(self):

        # forward pass
        tau = []
        for i, f_phy_tensor_i in enumerate(self.f_phy_tensor):
            tau.append(self.model.branch_list[i](f_phy_tensor_i))

        beta = self.model.trunk(torch.cat([self.x_phy, self.t_phy], 1))

        ## np.tile
        tau[0] = tau[0].reshape(-1, self.branch_out_dim)

        # Predicted
        if len(self.config['branches_control']['branch_list_ID']) == 1:
            u = torch.sum(tau[0] * beta, axis = 1)[:, None]

        # Autodiff
        u_x = torch.autograd.grad(
            u, self.x_phy, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        u_t = torch.autograd.grad(
            u, self.t_phy, grad_outputs=torch.ones_like(u), create_graph=True)[0]


        EQ1 = u_t + self.c_tensor *  u_x 
       

        # Compute losses
        loss = self.loss_fn(EQ1, self.residual_target)
       
        return loss
        
    def loss(self):

        # Forward pass and compute the losses per terms
        loss_ic = self.loss_initial_condition()
        loss_data = self.loss_data()
        loss_p = self.loss_physics()

        self.term_loss_tensor_dict['ic'] = loss_ic
        self.term_loss_tensor_dict['data'] = loss_data
        self.term_loss_tensor_dict['phy'] = loss_p
        
        # Apply adaptive lambdas
        self.loss_balancing_call()
        
    
        # Compute total loss
        loss_total = self.term_lambdas_tensor_dict['ic'] * loss_ic + self.term_lambdas_tensor_dict['data'] * loss_data + self.term_lambdas_tensor_dict['phy'] * loss_p 


        return loss_total
    
    def compute_loss_total_and_backward(self):
        self.optimizer_Adam.zero_grad()
        loss_total = self.loss()
        loss_total.backward()
        return loss_total
    
    def loss_balancing_call(self):
        
        # Update Lambdas
        if self.has_loss_balancing:
            if self.regular_iter % self.config['loss_balancing']['update_step'] == 0 or self.regular_iter == self.last_iter:
                self.loss_balancing_method()
        
        # Update lambdas dict
        for idx, key in enumerate(self.config['loss_terms']):
            self.term_lambdas_tensor_dict[key] = self.term_lambdas_tensor[idx]
    
    def loss_balancing_method(self):

        for key in self.config['loss_terms']:
            self.term_loss_tensor_dict[key].backward(retain_graph=True)

            if self.has_random_weight_fact:
                all_param_grads = self.loss_balancing_save_grads_RWF(self.model.mlp, self.term_grad_dict[key])
            else:
                all_param_grads = self.loss_balancing_save_grads(self.model.mlp, self.term_grad_dict[key])
            
            
            # Paper 1 'sum(norm) / norm': Wang et al. 2023 An Expert's Guide ...
            self.global_hat_lambdas[key] = np.linalg.norm(all_param_grads) 

            self.optimizer_Adam.zero_grad() #### NEEDED FOR THE OPTIMIZER

        
        # Paper 1 'sum(norm) / norm': Wang et al. 2023 An Expert's Guide ...
        total_sum = sum(self.global_hat_lambdas.values())
        new_lambs_hat = torch.tensor([total_sum / self.global_hat_lambdas[key] for key in self.config['loss_terms']], dtype=torch.float32, requires_grad=False).to(self.config['device'])
        # new_lambs_hat = torch.tensor([total_sum/(self.global_hat_lambdas[key] + self.config['loss_balancing']['EPS']) for key in self.config['loss_terms']], dtype=torch.float32, requires_grad=False).to(self.config['device'])
        
        # Compute new lambdas
        new_lambs = self.config['loss_balancing']['momentum'] * self.term_lambdas_tensor + (1 - self.config['loss_balancing']['momentum']) * new_lambs_hat
        
        # Update lambdas
        self.term_lambdas_tensor = new_lambs
    
    def loss_balancing_save_grads_RWF(self, mlp, term_grad_dict_key):

        all_param_grads = np.array([])

        for i,key in enumerate(term_grad_dict_key.keys()):
            
            if i <= self.config['neuralNet']['num_layers']:
                s = mlp.param_groups[3*i]['params'][0]
                v = mlp.param_groups[3*i+1]['params'][0]
                bias = mlp.param_groups[3*i+2]['params'][0]

                # Standard Dense 
                if s.grad is None:
                    print(f'\n s -> SKIP LAYER - standard dense {2*i}')
                else:
                    s_grad = s.grad.cpu().numpy()
                    term_grad_dict_key[key]['s'] = s_grad.reshape(-1)
                    all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['s']))
                
                if v.grad is None:
                    print(f'\n v -> SKIP LAYER - standard dense {2*i}')
                else:
                    v_grad = v.grad.cpu().numpy()
                    term_grad_dict_key[key]['v'] = v_grad.reshape(-1)
                    all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['v']))


                if bias.grad is None:
                    print(f'\n Bias -> SKIP LAYER - standard dense {2*i}')
                else:
                    bias_grad = bias.grad.cpu().numpy()
                    term_grad_dict_key[key]['bias'] = bias_grad.reshape(-1)
                    all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['bias']))
            else:

                if key == 'encoder_dense1_layer0':
                    s = mlp.param_groups[3*i]['params'][0]
                    v = mlp.param_groups[3*i+1]['params'][0]
                    bias = mlp.param_groups[3*i+2]['params'][0]

                    # Encoder Dense (Modified MLP)
                    if s.grad is None:
                        print(f'\n s -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        s_grad = s.grad.cpu().numpy()
                        term_grad_dict_key[key]['s'] = s_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['s']))
                    
                    if v.grad is None:
                        print(f'\n v -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        v_grad = v.grad.cpu().numpy()
                        term_grad_dict_key[key]['v'] = v_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['v']))


                    if bias.grad is None:
                        print(f'\n Bias -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        bias_grad = bias.grad.cpu().numpy()
                        term_grad_dict_key[key]['bias'] = bias_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['bias']))
               
                else: # 'encoder_dense2_layer0'
                    s = mlp.param_groups[3*i]['params'][0]
                    v = mlp.param_groups[3*i+1]['params'][0]
                    bias = mlp.param_groups[3*i+2]['params'][0]

                   # Encoder Dense (Modified MLP)
                    if s.grad is None:
                        print(f'\n s -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        s_grad = s.grad.cpu().numpy()
                        term_grad_dict_key[key]['s'] = s_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['s']))
                    
                    if v.grad is None:
                        print(f'\n v -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        v_grad = v.grad.cpu().numpy()
                        term_grad_dict_key[key]['v'] = v_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['v']))


                    if bias.grad is None:
                        print(f'\n Bias -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        bias_grad = bias.grad.cpu().numpy()
                        term_grad_dict_key[key]['bias'] = bias_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['bias']))

        return all_param_grads
    
    def loss_balancing_save_grads(self, mlp, term_grad_dict_key):

        all_param_grads = np.array([])

        for i,key in enumerate(term_grad_dict_key.keys()):
            
            if i <= self.config['neuralNet']['num_layers']:
                # Standard Dense 
                if mlp.standard_dense.net[2*i].weight.grad is None:
                    print(f'\n Weights -> SKIP LAYER - standard dense {2*i}')
                else:
                    weight_grad = mlp.standard_dense.net[2*i].weight.grad.cpu().numpy()
                    term_grad_dict_key[key]['weight'] = weight_grad.reshape(-1)
                    all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['weight']))


                if mlp.standard_dense.net[2*i].bias.grad is None:
                    print(f'\n Bias -> SKIP LAYER - standard dense {2*i}')
                else:
                    bias_grad = mlp.standard_dense.net[2*i].bias.grad.cpu().numpy()
                    term_grad_dict_key[key]['bias'] = bias_grad.reshape(-1)
                    all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['bias']))
            else:

                if key == 'encoder_dense1_layer0':
                    # Encoder Dense (Modified MLP)
                    if mlp.encoder_dense1.net[0].weight.grad is None:
                        print(f'\n Weights -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        weight_grad = mlp.encoder_dense1.net[0].weight.grad.cpu().numpy()
                        term_grad_dict_key[key]['weight'] = weight_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['weight']))


                    if mlp.encoder_dense1.net[0].bias.grad is None:
                        print(f'\n Bias -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        bias_grad = mlp.encoder_dense1.net[0].bias.grad.cpu().numpy()
                        term_grad_dict_key[key]['bias'] = bias_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['bias']))
               
                else: # 'encoder_dense2_layer0'
                   # Encoder Dense (Modified MLP)
                    if mlp.encoder_dense2.net[0].weight.grad is None:
                        print(f'\n Weights -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        weight_grad = mlp.encoder_dense2.net[0].weight.grad.cpu().numpy()
                        term_grad_dict_key[key]['weight'] = weight_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['weight']))


                    if mlp.encoder_dense2.net[0].bias.grad is None:
                        print(f'\n Bias -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        bias_grad = mlp.encoder_dense2.net[0].bias.grad.cpu().numpy()
                        term_grad_dict_key[key]['bias'] = bias_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['bias']))

        return all_param_grads
    
    def logger_call(self):
        
        # Total and term losses
        loss_train = self.total_loss
        l2_error_u = self.l2_error_u
        # l2_error_pressure = self.l2_error_pressure
        
        # Access to the learning rate
        lr = self.optimizer_Adam.param_groups[0]['lr']
        
        # LOG
        runtime1 = time.time()
        runtime_iter = (runtime1 - self.runtime0)/60
        self.runtime0 = runtime1 
        

        self.log_log = self.log_log + \
"\n---------------------------\n" + \
f"""Iter: {(self.regular_iter)}/{(self.last_iter)} - Time: {round(runtime_iter,4)}min""" + \
"\n---------------------------\n" + \
f"""- total_loss_train: {loss_train} \n- l2_relative_error - u: {l2_error_u} \n\n- lr: {lr}\n\n""" 

        for key in self.config['loss_terms']:
            self.log_log = self.log_log + \
f"""- {key}_loss: {self.term_loss_tensor_dict[key].item()}\n"""


        # Loss balancing log
        self.log_log = self.log_log +"\n"
        
        for key in self.config['loss_terms']:
            self.log_log = self.log_log + \
f"""- {key}_lambdas: {self.term_lambdas_tensor_dict[key].item()}\n"""
     
        # Monitoring resources 
        if self.config['logging']['monitoring_resources']:
            gpu_load, gpu_memory = get_gpu_usage()
            if gpu_load is not None:
                self.log_log = self.log_log + \
"\n---------\n" + \
f"""- CPU usage: {get_cpu_usage()}% \n""" + \
f"""- Memory usage: {get_memory_usage()}MB \n""" + \
f"""- GPU usage: {gpu_load}% \n""" + \
f"""- GPU memory usage: {gpu_memory}MB \n"""
            else:
                self.log_log = self.log_log + \
"\n---------\n" + \
f"""- CPU usage: {get_cpu_usage()}% \n""" + \
f"""- Memory usage: {get_memory_usage()}MB \n""" + \
f"""- No GPU detected \n"""
                
        # Save into a .txt file
        with open(self.config['train_progress_file_path'], 'w') as file:
            file.write(self.log_log)
        
        # Tensorboard
        self.writer.add_scalar('total_loss', self.total_loss, self.regular_iter)
        self.writer.add_scalar('l2_error_u', self.l2_error_u, self.regular_iter)
        # self.writer.add_scalar('l2_error_pressure', self.l2_error_pressure, self.regular_iter)
        self.writer.add_scalar('lr', lr, self.regular_iter)
        for key in self.config['loss_terms']: 
            self.writer.add_scalar(f'{key}_loss', self.term_loss_tensor_dict[key].item(), self.regular_iter)
            self.writer.add_scalar(f'{key}_lambdas', self.term_lambdas_tensor_dict[key].item(), self.regular_iter)
    
    def checkpoint_call(self, tracking_param):
        
        # Checkpoints
        if tracking_param < self.min_error_for_checkpoint:
        
            directory = self.config['checkpoints_folder_path']

            # Construct the full path for files
            filename_model_weights = f"model_weights.pt"
            path_model_weights = directory.joinpath(filename_model_weights)
           
            ## Save model weights based on the tracking_param
            model_weights = {}
            for i in range(len(self.model.branch_list)):
                end_string = '_state_dict'
                key = self.config['branches_control']['branch_list_ID'][i] + end_string
                aux_dict = {key: self.model.branch_list[i].state_dict()}
                model_weights.update(aux_dict)
            
            model_weights.update({'trunk_state_dict': self.model.trunk.state_dict()})

            torch.save(model_weights, path_model_weights)

            # Update for next iteration
            self.min_error_for_checkpoint = tracking_param 
            self.best_iter = self.regular_iter 
            self.config['logger'].info(f'New best iteration {self.best_iter} registered.')

    def collocation_points_generator(self, batch_size_coll, coordinates, time_min, time_max):
       
        N_coord = coordinates.shape[0]

        if N_coord < batch_size_coll:
            x_idx = np.random.choice(N_coord, N_coord, replace=False)  # Select all elements if N_coord < batch_size_coll
        else:
            x_idx = np.random.choice(N_coord, batch_size_coll, replace=False)
        x = coordinates[x_idx]

        #We will consider time values in the interval min(time_vector) and max(time_vector)
        t = np.sort(np.random.uniform(time_min, time_max, batch_size_coll))

        return x[:,0][:,None], t[:,None]

    def random_sampling(self, xt, u, label = 'val'):
        """Return a random uniform sample of coordinate points from label with their respective velocities and pressure.
        """
        N_coord = xt.shape[0]
        idx = np.random.choice(N_coord,  round(self.config['train'][f'batch_dfraction'][label]*N_coord), replace=False)
        
        idx = np.sort(idx)

        return idx
    
    def train(self, timePrm, full_ds, dataPrm, val_ds):
        
        print("### TRAINING ... ###")

        self.last_iter = self.config['train']['adam_steps'] + self.config['train']['lbfgs_steps'] - 1
        self.regular_iter = 0

        time_vector = timePrm.time_vector[:,None]

        xt_ic = {} 
        u_ic = {} 
        
        xt_data = {} 
        u_data = {} 
        
        xt_val = {} 
        u_val = {}
      
        c_list = []

        for chosen_flow_label in self.config['train']['training_param_label']:


            #---------------------
            # Fixed dataset
            #---------------------
            ## BOUNDARY CONDITIONS AND INITIAL CONDITION
            xt_ic[chosen_flow_label], u_ic[chosen_flow_label]= craft_bc_and_ic_dataset(full_ds[chosen_flow_label], time_vector)
            
            ## DATA 
            xt_data[chosen_flow_label], u_data[chosen_flow_label] = build_stratum_dataset(dataPrm[chosen_flow_label], time_vector)

            ## VALIDATION DATASET
            xt_val[chosen_flow_label], u_val[chosen_flow_label] = craft_validation_dataset(val_ds[chosen_flow_label], time_vector)

            c_list.append(self.c_dict[chosen_flow_label])

        # """The coordinate points are the same between the different datasets, so we can choose a geometry among the options"""
        coordinates, time_min, time_max = get_coordinates_for_generator(full_ds[self.config['train']['training_param_label'][0]], time_vector)

        # c parameter - For Loss PHY
        c_array = np.tile(np.array(c_list), self.config['train']['batch_size_coll'])
        self.c_tensor = torch.tensor(c_array[:,None], dtype = torch.float32, requires_grad= False).to(self.config['device'])

        # Workflow
        self.custom_bar = trange(self.last_iter + 1)
        
        for regular_iter in self.custom_bar:

            self.regular_iter = regular_iter

            # Branch
            branch_input = {}
            for key in self.config['branches_control']['branch_input_ID']:
                branch_input.update({key:[]})

            # Trunk
            xt_ic_sample_all = []
            u_ic_target = []
           
            xt_data_sample_all = []
            u_data_target = []

           
            # Val
            xt_val_sample_all = []
            u_val_target = []
           

            ## Random Sampling - Get Indexes
            # IC 
            idx_ic = self.random_sampling(xt_ic[chosen_flow_label], u_ic[chosen_flow_label], label = 'ic')
            
            # DATA 
            idx_data = self.random_sampling(xt_data[chosen_flow_label], u_data[chosen_flow_label], label = 'data')

            # VAL 
            idx_val = self.random_sampling(xt_val[chosen_flow_label], u_val[chosen_flow_label], label = 'val')

            for chosen_flow_label in self.config['train']['training_param_label']:
                
                # IC 
                xt_ic_sample =  xt_ic[chosen_flow_label][idx_ic]
                u_ic_sample = u_ic[chosen_flow_label][idx_ic]
                
                # DATA 
                xt_data_sample =  xt_data[chosen_flow_label][idx_data]
                u_data_sample = u_data[chosen_flow_label][idx_data]

                # VAL 
                xt_val_sample = xt_val[chosen_flow_label][idx_val]
                u_val_sample = u_val[chosen_flow_label][idx_val]

                # Branch
                branch_input[self.config['branches_control']['branch_input_ID'][0]].append(np.repeat(np.array([self.c_dict[chosen_flow_label]]),90)[None,:])


                ## Trunk
                xt_ic_sample_all.append(xt_ic_sample)
                u_ic_target.append(u_ic_sample)
                
                xt_data_sample_all.append(xt_data_sample)
                u_data_target.append(u_data_sample)

                xt_val_sample_all.append(xt_val_sample)
                u_val_target.append(u_val_sample)

            # Trunk
            xt_ic_sample_all = np.array(xt_ic_sample_all)
            u_ic_target = np.array(u_ic_target)
            
            xt_data_sample_all = np.array(xt_data_sample_all)
            u_data_target = np.array(u_data_target)

            xt_val_sample_all = np.array(xt_val_sample_all)
            u_val_target = np.array(u_val_target)
            
            # IC
            # N = 4, P = 1800 
            self.xt_ic_tensor = torch.tensor(np.swapaxes(xt_ic_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            # self.f_ic_tensor = []
            # for key in branch_input.keys():
            #     f_ic_array = np.tile(np.array(branch_input[key]), (xyzt_ic_sample.shape[0],1))
            #     self.f_ic_tensor.append(torch.tensor(f_ic_array, dtype = torch.float32, requires_grad= False).to(self.config['device']))
            self.f_ic_tensor = []
            num_samples = xt_ic_sample.shape[0]  # 51850
            target_device = self.config['device']

            for key in branch_input.keys():
                
                base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]),dtype=torch.float32,device=target_device)
                
                final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
                
                self.f_ic_tensor.append(final_view)
            
            self.u_ic_tensor = torch.tensor(np.swapaxes(u_ic_target,0,1).reshape(-1,self.output_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            # DATA
            self.xt_data_tensor = torch.tensor(np.swapaxes(xt_data_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            self.f_data_tensor = []
            num_samples = xt_data_sample.shape[0]  
            target_device = self.config['device']

            for key in branch_input.keys():
                
                base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]),dtype=torch.float32,device=target_device)
                
                final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
                
                self.f_data_tensor.append(final_view)
            
            self.u_data_tensor = torch.tensor(np.swapaxes(u_data_target,0,1).reshape(-1,self.output_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            
            
           
            # Validation dataset
            # N = 5, P = xyzt_val_fixed.shape[0]
            self.xt_val_tensor = torch.tensor(np.swapaxes(xt_val_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            
            self.f_val_tensor = []
            num_samples = xt_val_sample.shape[0] 
            target_device = self.config['device']

            for key in branch_input.keys():


                base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]), dtype=torch.float32, device=target_device)
               
                final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
               
                self.f_val_tensor.append(final_view)
            
            self.u_val_tensor = torch.tensor(np.swapaxes(u_val_target,0,1).reshape(-1,self.output_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
           

            ## COLLOCATION POINTS
            # Random sample
            x_phy, t_phy = self.collocation_points_generator(self.config['train']['batch_size_coll'], coordinates, time_min, time_max)

            # N = 4, P = batch_size_coll
            x_phy_array = np.repeat(x_phy, self.config['train']['n_training_param'], axis=0)
            t_phy_array = np.repeat(t_phy, self.config['train']['n_training_param'], axis=0)

            self.x_phy = torch.tensor(x_phy_array, dtype = torch.float32, requires_grad= True).to(self.config['device'])
            self.t_phy = torch.tensor(t_phy_array, dtype = torch.float32, requires_grad= True).to(self.config['device'])

            
            self.f_phy_tensor = []
            num_samples = self.config['train']['batch_size_coll']  # 1024
            target_device = self.config['device']

            for key in branch_input.keys():
                
                base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]),dtype=torch.float32,device=target_device)
               
                final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
              
                self.f_phy_tensor.append(final_view)
            
            self.residual_target = torch.zeros((self.x_phy.shape[0],1), dtype = torch.float32, requires_grad= False).to(self.config['device'])


        
            # Compute total loss, apply backward pass and optimize
            loss_total = self.compute_loss_total_and_backward()
            self.optimizer_Adam.step()

            # Store batch losses
            self.total_loss = loss_total.item()


            # Compute l2 relative error with Validation dataset
            ## REFERENCE
            u_ref = self.u_val_tensor
            

            ## PREDICTED
            with torch.no_grad():
                tau = []
                for i, f_star_tensor_i in enumerate(self.f_val_tensor):
                    tau.append(self.model.branch_list[i](f_star_tensor_i))
                
                beta = self.model.trunk(self.xt_val_tensor)

                tau[0] = tau[0].reshape(-1, self.branch_out_dim)

            u_pred = torch.sum(tau[0] * beta, axis = 1)[:, None]
               
            
            l2_relative_error_u = metric_l2_relative_error(exact = u_ref, pred = u_pred)
            
           
            # Store errors
            self.l2_error_u = l2_relative_error_u.item()
           

            # -------------
            if self.regular_iter % self.config['logging']['log_every_steps'] == 0 or self.regular_iter == self.last_iter:
                self.logger_call()

            self.checkpoint_call(self.total_loss)

            # Conditions to stop the loop
            if self.regular_iter == self.config['train']['stop_iter']:
                self.logger_call()
                break
            
            if self.has_exponential_decay:
                self.scheduler_Adam.step()
            # -------------

        self.config['logger'].info("###-----------------------------------------###")
        self.config['logger'].info("### The best model is obtained in iteration = " + f"{self.best_iter}" + " with a total_loss = " + f"{self.min_error_for_checkpoint}.") 


class Trainer_PIDeepONet(nn.Module):
    
    def __init__(self, config, model):
        
        super().__init__()
        
        self.config = config
        self.has_random_weight_fact = self.config['random_weight_fact']['enabled']
        self.has_loss_balancing = self.config['loss_balancing']['enabled']
        self.has_exponential_decay = self.config['optim1']['exponential_decay']['enabled']

        # Initialize the SummaryWriter
        if self.config['mode']=='train':
            self.writer = SummaryWriter(log_dir=config['logs_folder_path'])

        self.model = model
        self.branch_out_dim = config['branch1']['neuralNet']['out_dim']
        self.trunk_in_dim = config['trunk']['neuralNet']['in_dim']
        self.output_dim = 1

        # PDE parameters
        self.c_dict = self.config['pde_param']['c']

         # Optimizers
        if config['optim1']['optimizer'] == 'Adam':

            # Retrieve parameter groups from each model
            param_groups_branch_sum = []
            for branch in self.model.branch_list:
                param_groups_branch_i = list(branch.param_groups)
                param_groups_branch_sum += param_groups_branch_i
            
            param_groups_trunk = list(self.model.trunk.param_groups)

            # Combine the parameter groups into a single list
            combined_param_groups = param_groups_branch_sum + param_groups_trunk

            self.optimizer_Adam = torch.optim.Adam(combined_param_groups, lr=config['optim1']['learning_rate'], \
                                                    betas=(config['optim1']['beta1'], config['optim1']['beta2']), \
                                                    eps=config['optim1']['eps'] )
            
            if self.has_exponential_decay:
                def lr_lambda_func(current_step):
                    """Custom function for the exponential decay."""
                    
                    decay_rate = self.config['optim1']['exponential_decay']['decay_rate']
                    decay_steps = self.config['optim1']['exponential_decay']['decay_steps']

                    factor = decay_rate ** (current_step / decay_steps)

                    return factor

                self.scheduler_Adam = torch.optim.lr_scheduler.LambdaLR(self.optimizer_Adam, lr_lambda_func)
        
        if config['optim2']['optimizer'] == 'LBFGS':

            # Extract parameters from custom parameter groups
            all_params = [param for group in self.model.branch_list[0].param_groups + self.model.trunk.param_groups for param in group['params']]
            # all_params = [param for group in self.model.branch_list[0].param_groups + self.model.branch_list[1].param_groups + self.model.trunk.param_groups for param in group['params']]
            # all_params = [param for group in self.model.mlp.param_groups for param in group['params']]
            
            self.optimizer_LBFGS = torch.optim.LBFGS(all_params, max_iter=config['optim2']['max_iter'], max_eval=config['optim2']['max_eval'], tolerance_grad=1.0 * torch.finfo(torch.float32).eps, history_size=50)

        
        # Loss function
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        

        # Logging
        self.total_loss = None
        self.l2_error_u = None

        # Loss terms
        self.term_loss_tensor_dict = {key:[] for key in self.config['loss_terms']}

        # lambdas initialization
        self.term_lambdas_tensor = torch.tensor(
                                    [self.config['lambda_weights_init'][i] for i in range(len(self.config['loss_terms']))], 
                                    dtype=torch.float32, requires_grad=False).to(self.config['device'])

        self.term_lambdas_tensor_dict = {key: self.term_lambdas_tensor[i] for i,key in enumerate(self.config['loss_terms'])}

        self.log_log = """LOG\n"""
        self.min_error_for_checkpoint = np.Infinity
        self.runtime0 = self.config['tick_start']
        self.best_iter = 0

    def loss_initial_condition(self):

        tau = []
        for i, f_ic_tensor_i in enumerate(self.f_ic_tensor):
            tau.append(self.model.branch_list[i](f_ic_tensor_i))
 
        beta = self.model.trunk(self.xt_ic_tensor)

        ## np.tile
        tau[0] = tau[0].reshape(-1, self.branch_out_dim)

        # Predicted
        u_pred = torch.sum(tau[0] * beta, axis = 1)[:, None]
       

        # Ground Trust
        u_GT = self.u_ic_tensor
        
        # Compute losses
        u_loss = self.loss_fn(u_pred, u_GT)

    
        return u_loss
    
    def loss_physics(self):

        # forward pass
        tau = []
        for i, f_phy_tensor_i in enumerate(self.f_phy_tensor):
            tau.append(self.model.branch_list[i](f_phy_tensor_i))

        beta = self.model.trunk(torch.cat([self.x_phy, self.t_phy], 1))

        ## np.tile
        tau[0] = tau[0].reshape(-1, self.branch_out_dim)

        # Predicted
        if len(self.config['branches_control']['branch_list_ID']) == 1:
            u = torch.sum(tau[0] * beta, axis = 1)[:, None]

        # Autodiff
        u_x = torch.autograd.grad(
            u, self.x_phy, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        u_t = torch.autograd.grad(
            u, self.t_phy, grad_outputs=torch.ones_like(u), create_graph=True)[0]


        EQ1 = u_t + self.c_tensor *  u_x 
       

        # Compute losses
        loss = self.loss_fn(EQ1, self.residual_target)
       
        return loss
        
    def loss(self):

        # Forward pass and compute the losses per terms
        loss_ic = self.loss_initial_condition()
        loss_p = self.loss_physics()

        self.term_loss_tensor_dict['ic'] = loss_ic
        self.term_loss_tensor_dict['phy'] = loss_p
        
        # Apply adaptive lambdas
        self.loss_balancing_call()
        
    
        # Compute total loss
        loss_total = self.term_lambdas_tensor_dict['ic'] * loss_ic + self.term_lambdas_tensor_dict['phy'] * loss_p 


        return loss_total
    
    def compute_loss_total_and_backward(self):
        self.optimizer_Adam.zero_grad()
        loss_total = self.loss()
        loss_total.backward()
        return loss_total
    
    def loss_balancing_call(self):
        
        # Update Lambdas
        if self.has_loss_balancing:
            if self.regular_iter % self.config['loss_balancing']['update_step'] == 0 or self.regular_iter == self.last_iter:
                self.loss_balancing_method()
        
        # Update lambdas dict
        for idx, key in enumerate(self.config['loss_terms']):
            self.term_lambdas_tensor_dict[key] = self.term_lambdas_tensor[idx]
    
    def loss_balancing_method(self):

        for key in self.config['loss_terms']:
            self.term_loss_tensor_dict[key].backward(retain_graph=True)

            if self.has_random_weight_fact:
                all_param_grads = self.loss_balancing_save_grads_RWF(self.model.mlp, self.term_grad_dict[key])
            else:
                all_param_grads = self.loss_balancing_save_grads(self.model.mlp, self.term_grad_dict[key])
            
            
            # Paper 1 'sum(norm) / norm': Wang et al. 2023 An Expert's Guide ...
            self.global_hat_lambdas[key] = np.linalg.norm(all_param_grads) 

            self.optimizer_Adam.zero_grad() #### NEEDED FOR THE OPTIMIZER

        
        # Paper 1 'sum(norm) / norm': Wang et al. 2023 An Expert's Guide ...
        total_sum = sum(self.global_hat_lambdas.values())
        new_lambs_hat = torch.tensor([total_sum / self.global_hat_lambdas[key] for key in self.config['loss_terms']], dtype=torch.float32, requires_grad=False).to(self.config['device'])
        # new_lambs_hat = torch.tensor([total_sum/(self.global_hat_lambdas[key] + self.config['loss_balancing']['EPS']) for key in self.config['loss_terms']], dtype=torch.float32, requires_grad=False).to(self.config['device'])
        
        # Compute new lambdas
        new_lambs = self.config['loss_balancing']['momentum'] * self.term_lambdas_tensor + (1 - self.config['loss_balancing']['momentum']) * new_lambs_hat
        
        # Update lambdas
        self.term_lambdas_tensor = new_lambs
    
    def loss_balancing_save_grads_RWF(self, mlp, term_grad_dict_key):

        all_param_grads = np.array([])

        for i,key in enumerate(term_grad_dict_key.keys()):
            
            if i <= self.config['neuralNet']['num_layers']:
                s = mlp.param_groups[3*i]['params'][0]
                v = mlp.param_groups[3*i+1]['params'][0]
                bias = mlp.param_groups[3*i+2]['params'][0]

                # Standard Dense 
                if s.grad is None:
                    print(f'\n s -> SKIP LAYER - standard dense {2*i}')
                else:
                    s_grad = s.grad.cpu().numpy()
                    term_grad_dict_key[key]['s'] = s_grad.reshape(-1)
                    all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['s']))
                
                if v.grad is None:
                    print(f'\n v -> SKIP LAYER - standard dense {2*i}')
                else:
                    v_grad = v.grad.cpu().numpy()
                    term_grad_dict_key[key]['v'] = v_grad.reshape(-1)
                    all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['v']))


                if bias.grad is None:
                    print(f'\n Bias -> SKIP LAYER - standard dense {2*i}')
                else:
                    bias_grad = bias.grad.cpu().numpy()
                    term_grad_dict_key[key]['bias'] = bias_grad.reshape(-1)
                    all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['bias']))
            else:

                if key == 'encoder_dense1_layer0':
                    s = mlp.param_groups[3*i]['params'][0]
                    v = mlp.param_groups[3*i+1]['params'][0]
                    bias = mlp.param_groups[3*i+2]['params'][0]

                    # Encoder Dense (Modified MLP)
                    if s.grad is None:
                        print(f'\n s -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        s_grad = s.grad.cpu().numpy()
                        term_grad_dict_key[key]['s'] = s_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['s']))
                    
                    if v.grad is None:
                        print(f'\n v -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        v_grad = v.grad.cpu().numpy()
                        term_grad_dict_key[key]['v'] = v_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['v']))


                    if bias.grad is None:
                        print(f'\n Bias -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        bias_grad = bias.grad.cpu().numpy()
                        term_grad_dict_key[key]['bias'] = bias_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['bias']))
               
                else: # 'encoder_dense2_layer0'
                    s = mlp.param_groups[3*i]['params'][0]
                    v = mlp.param_groups[3*i+1]['params'][0]
                    bias = mlp.param_groups[3*i+2]['params'][0]

                   # Encoder Dense (Modified MLP)
                    if s.grad is None:
                        print(f'\n s -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        s_grad = s.grad.cpu().numpy()
                        term_grad_dict_key[key]['s'] = s_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['s']))
                    
                    if v.grad is None:
                        print(f'\n v -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        v_grad = v.grad.cpu().numpy()
                        term_grad_dict_key[key]['v'] = v_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['v']))


                    if bias.grad is None:
                        print(f'\n Bias -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        bias_grad = bias.grad.cpu().numpy()
                        term_grad_dict_key[key]['bias'] = bias_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['bias']))

        return all_param_grads
    
    def loss_balancing_save_grads(self, mlp, term_grad_dict_key):

        all_param_grads = np.array([])

        for i,key in enumerate(term_grad_dict_key.keys()):
            
            if i <= self.config['neuralNet']['num_layers']:
                # Standard Dense 
                if mlp.standard_dense.net[2*i].weight.grad is None:
                    print(f'\n Weights -> SKIP LAYER - standard dense {2*i}')
                else:
                    weight_grad = mlp.standard_dense.net[2*i].weight.grad.cpu().numpy()
                    term_grad_dict_key[key]['weight'] = weight_grad.reshape(-1)
                    all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['weight']))


                if mlp.standard_dense.net[2*i].bias.grad is None:
                    print(f'\n Bias -> SKIP LAYER - standard dense {2*i}')
                else:
                    bias_grad = mlp.standard_dense.net[2*i].bias.grad.cpu().numpy()
                    term_grad_dict_key[key]['bias'] = bias_grad.reshape(-1)
                    all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['bias']))
            else:

                if key == 'encoder_dense1_layer0':
                    # Encoder Dense (Modified MLP)
                    if mlp.encoder_dense1.net[0].weight.grad is None:
                        print(f'\n Weights -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        weight_grad = mlp.encoder_dense1.net[0].weight.grad.cpu().numpy()
                        term_grad_dict_key[key]['weight'] = weight_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['weight']))


                    if mlp.encoder_dense1.net[0].bias.grad is None:
                        print(f'\n Bias -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        bias_grad = mlp.encoder_dense1.net[0].bias.grad.cpu().numpy()
                        term_grad_dict_key[key]['bias'] = bias_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['bias']))
               
                else: # 'encoder_dense2_layer0'
                   # Encoder Dense (Modified MLP)
                    if mlp.encoder_dense2.net[0].weight.grad is None:
                        print(f'\n Weights -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        weight_grad = mlp.encoder_dense2.net[0].weight.grad.cpu().numpy()
                        term_grad_dict_key[key]['weight'] = weight_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['weight']))


                    if mlp.encoder_dense2.net[0].bias.grad is None:
                        print(f'\n Bias -> SKIP LAYER - encoder dense 1 {0}')
                    else:
                        bias_grad = mlp.encoder_dense2.net[0].bias.grad.cpu().numpy()
                        term_grad_dict_key[key]['bias'] = bias_grad.reshape(-1)
                        all_param_grads = np.concatenate((all_param_grads,term_grad_dict_key[key]['bias']))

        return all_param_grads
    
    def logger_call(self):
        
        # Total and term losses
        loss_train = self.total_loss
        l2_error_u = self.l2_error_u
        # l2_error_pressure = self.l2_error_pressure
        
        # Access to the learning rate
        lr = self.optimizer_Adam.param_groups[0]['lr']
        
        # LOG
        runtime1 = time.time()
        runtime_iter = (runtime1 - self.runtime0)/60
        self.runtime0 = runtime1 
        

        self.log_log = self.log_log + \
"\n---------------------------\n" + \
f"""Iter: {(self.regular_iter)}/{(self.last_iter)} - Time: {round(runtime_iter,4)}min""" + \
"\n---------------------------\n" + \
f"""- total_loss_train: {loss_train} \n- l2_relative_error - u: {l2_error_u} \n\n- lr: {lr}\n\n""" 

        for key in self.config['loss_terms']:
            self.log_log = self.log_log + \
f"""- {key}_loss: {self.term_loss_tensor_dict[key].item()}\n"""


        # Loss balancing log
        self.log_log = self.log_log +"\n"
        
        for key in self.config['loss_terms']:
            self.log_log = self.log_log + \
f"""- {key}_lambdas: {self.term_lambdas_tensor_dict[key].item()}\n"""
     
        # Monitoring resources 
        if self.config['logging']['monitoring_resources']:
            gpu_load, gpu_memory = get_gpu_usage()
            if gpu_load is not None:
                self.log_log = self.log_log + \
"\n---------\n" + \
f"""- CPU usage: {get_cpu_usage()}% \n""" + \
f"""- Memory usage: {get_memory_usage()}MB \n""" + \
f"""- GPU usage: {gpu_load}% \n""" + \
f"""- GPU memory usage: {gpu_memory}MB \n"""
            else:
                self.log_log = self.log_log + \
"\n---------\n" + \
f"""- CPU usage: {get_cpu_usage()}% \n""" + \
f"""- Memory usage: {get_memory_usage()}MB \n""" + \
f"""- No GPU detected \n"""
                
        # Save into a .txt file
        with open(self.config['train_progress_file_path'], 'w') as file:
            file.write(self.log_log)
        
        # Tensorboard
        self.writer.add_scalar('total_loss', self.total_loss, self.regular_iter)
        self.writer.add_scalar('l2_error_u', self.l2_error_u, self.regular_iter)
        # self.writer.add_scalar('l2_error_pressure', self.l2_error_pressure, self.regular_iter)
        self.writer.add_scalar('lr', lr, self.regular_iter)
        for key in self.config['loss_terms']: 
            self.writer.add_scalar(f'{key}_loss', self.term_loss_tensor_dict[key].item(), self.regular_iter)
            self.writer.add_scalar(f'{key}_lambdas', self.term_lambdas_tensor_dict[key].item(), self.regular_iter)
    
    def checkpoint_call(self, tracking_param):
        
        # Checkpoints
        if tracking_param < self.min_error_for_checkpoint:
        
            directory = self.config['checkpoints_folder_path']

            # Construct the full path for files
            filename_model_weights = f"model_weights.pt"
            path_model_weights = directory.joinpath(filename_model_weights)
           
            ## Save model weights based on the tracking_param
            model_weights = {}
            for i in range(len(self.model.branch_list)):
                end_string = '_state_dict'
                key = self.config['branches_control']['branch_list_ID'][i] + end_string
                aux_dict = {key: self.model.branch_list[i].state_dict()}
                model_weights.update(aux_dict)
            
            model_weights.update({'trunk_state_dict': self.model.trunk.state_dict()})

            torch.save(model_weights, path_model_weights)

            # Update for next iteration
            self.min_error_for_checkpoint = tracking_param 
            self.best_iter = self.regular_iter 
            self.config['logger'].info(f'New best iteration {self.best_iter} registered.')

    def collocation_points_generator(self, batch_size_coll, coordinates, time_min, time_max):
       
        N_coord = coordinates.shape[0]

        if N_coord < batch_size_coll:
            x_idx = np.random.choice(N_coord, N_coord, replace=False)  # Select all elements if N_coord < batch_size_coll
        else:
            x_idx = np.random.choice(N_coord, batch_size_coll, replace=False)
        x = coordinates[x_idx]

        #We will consider time values in the interval min(time_vector) and max(time_vector)
        t = np.sort(np.random.uniform(time_min, time_max, batch_size_coll))

        return x[:,0][:,None], t[:,None]

    def random_sampling(self, xt, u, label = 'val'):
        """Return a random uniform sample of coordinate points from label with their respective velocities and pressure.
        """
        N_coord = xt.shape[0]
        idx = np.random.choice(N_coord,  round(self.config['train'][f'batch_dfraction'][label]*N_coord), replace=False)
        
        idx = np.sort(idx)

        return idx
    
    def train(self, timePrm, full_ds, val_ds):
        
        print("### TRAINING ... ###")

        self.last_iter = self.config['train']['adam_steps'] + self.config['train']['lbfgs_steps'] - 1
        self.regular_iter = 0

        time_vector = timePrm.time_vector[:,None]

        xt_ic = {} 
        u_ic = {} 
        
        xt_val = {} 
        u_val = {}
      
        c_list = []

        for chosen_flow_label in self.config['train']['training_param_label']:


            #---------------------
            # Fixed dataset
            #---------------------
            ## BOUNDARY CONDITIONS AND INITIAL CONDITION
            xt_ic[chosen_flow_label], u_ic[chosen_flow_label]= craft_bc_and_ic_dataset(full_ds[chosen_flow_label], time_vector)
            

            ## VALIDATION DATASET
            xt_val[chosen_flow_label], u_val[chosen_flow_label] = craft_validation_dataset(val_ds[chosen_flow_label], time_vector)

            c_list.append(self.c_dict[chosen_flow_label])

        # """The coordinate points are the same between the different datasets, so we can choose a geometry among the options"""
        coordinates, time_min, time_max = get_coordinates_for_generator(full_ds[self.config['train']['training_param_label'][0]], time_vector)

        # c parameter - For Loss PHY
        c_array = np.tile(np.array(c_list), self.config['train']['batch_size_coll'])
        self.c_tensor = torch.tensor(c_array[:,None], dtype = torch.float32, requires_grad= False).to(self.config['device'])

        # Workflow
        self.custom_bar = trange(self.last_iter + 1)
        
        for regular_iter in self.custom_bar:

            self.regular_iter = regular_iter

            # Branch
            branch_input = {}
            for key in self.config['branches_control']['branch_input_ID']:
                branch_input.update({key:[]})

            # Trunk
            xt_ic_sample_all = []
            u_ic_target = []

           
            # Val
            xt_val_sample_all = []
            u_val_target = []
           

            ## Random Sampling - Get Indexes
            # IC 
            idx_ic = self.random_sampling(xt_ic[chosen_flow_label], u_ic[chosen_flow_label], label = 'ic')

            # VAL 
            idx_val = self.random_sampling(xt_val[chosen_flow_label], u_val[chosen_flow_label], label = 'val')

            for chosen_flow_label in self.config['train']['training_param_label']:
                
                # IC 
                xt_ic_sample =  xt_ic[chosen_flow_label][idx_ic]
                u_ic_sample = u_ic[chosen_flow_label][idx_ic]

                # VAL 
                xt_val_sample = xt_val[chosen_flow_label][idx_val]
                u_val_sample = u_val[chosen_flow_label][idx_val]

                # Branch
                branch_input[self.config['branches_control']['branch_input_ID'][0]].append(np.repeat(np.array([self.c_dict[chosen_flow_label]]),90)[None,:])


                ## Trunk
                xt_ic_sample_all.append(xt_ic_sample)
                u_ic_target.append(u_ic_sample)

                xt_val_sample_all.append(xt_val_sample)
                u_val_target.append(u_val_sample)

            # Trunk
            xt_ic_sample_all = np.array(xt_ic_sample_all)
            u_ic_target = np.array(u_ic_target)

            xt_val_sample_all = np.array(xt_val_sample_all)
            u_val_target = np.array(u_val_target)
            
            # IC
            # N = 4, P = 1800 
            self.xt_ic_tensor = torch.tensor(np.swapaxes(xt_ic_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            # self.f_ic_tensor = []
            # for key in branch_input.keys():
            #     f_ic_array = np.tile(np.array(branch_input[key]), (xyzt_ic_sample.shape[0],1))
            #     self.f_ic_tensor.append(torch.tensor(f_ic_array, dtype = torch.float32, requires_grad= False).to(self.config['device']))
            self.f_ic_tensor = []
            num_samples = xt_ic_sample.shape[0]  # 51850
            target_device = self.config['device']

            for key in branch_input.keys():
                
                base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]),dtype=torch.float32,device=target_device)
                
                final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
                
                self.f_ic_tensor.append(final_view)
            
            self.u_ic_tensor = torch.tensor(np.swapaxes(u_ic_target,0,1).reshape(-1,self.output_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            
            
           
            # Validation dataset
            # N = 5, P = xyzt_val_fixed.shape[0]
            self.xt_val_tensor = torch.tensor(np.swapaxes(xt_val_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            
            self.f_val_tensor = []
            num_samples = xt_val_sample.shape[0] 
            target_device = self.config['device']

            for key in branch_input.keys():


                base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]), dtype=torch.float32, device=target_device)
               
                final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
               
                self.f_val_tensor.append(final_view)
            
            self.u_val_tensor = torch.tensor(np.swapaxes(u_val_target,0,1).reshape(-1,self.output_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
           

            ## COLLOCATION POINTS
            # Random sample
            x_phy, t_phy = self.collocation_points_generator(self.config['train']['batch_size_coll'], coordinates, time_min, time_max)

            # N = 4, P = batch_size_coll
            x_phy_array = np.repeat(x_phy, self.config['train']['n_training_param'], axis=0)
            t_phy_array = np.repeat(t_phy, self.config['train']['n_training_param'], axis=0)

            self.x_phy = torch.tensor(x_phy_array, dtype = torch.float32, requires_grad= True).to(self.config['device'])
            self.t_phy = torch.tensor(t_phy_array, dtype = torch.float32, requires_grad= True).to(self.config['device'])

            
            self.f_phy_tensor = []
            num_samples = self.config['train']['batch_size_coll']  # 1024
            target_device = self.config['device']

            for key in branch_input.keys():
                
                base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]),dtype=torch.float32,device=target_device)
               
                final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
              
                self.f_phy_tensor.append(final_view)
            
            self.residual_target = torch.zeros((self.x_phy.shape[0],1), dtype = torch.float32, requires_grad= False).to(self.config['device'])


        
            # Compute total loss, apply backward pass and optimize
            loss_total = self.compute_loss_total_and_backward()
            self.optimizer_Adam.step()

            # Store batch losses
            self.total_loss = loss_total.item()


            # Compute l2 relative error with Validation dataset
            ## REFERENCE
            u_ref = self.u_val_tensor
            

            ## PREDICTED
            with torch.no_grad():
                tau = []
                for i, f_star_tensor_i in enumerate(self.f_val_tensor):
                    tau.append(self.model.branch_list[i](f_star_tensor_i))
                
                beta = self.model.trunk(self.xt_val_tensor)

                tau[0] = tau[0].reshape(-1, self.branch_out_dim)

            u_pred = torch.sum(tau[0] * beta, axis = 1)[:, None]
               
            
            l2_relative_error_u = metric_l2_relative_error(exact = u_ref, pred = u_pred)
            
           
            # Store errors
            self.l2_error_u = l2_relative_error_u.item()
           

            # -------------
            if self.regular_iter % self.config['logging']['log_every_steps'] == 0 or self.regular_iter == self.last_iter:
                self.logger_call()

            self.checkpoint_call(self.total_loss)

            # Conditions to stop the loop
            if self.regular_iter == self.config['train']['stop_iter']:
                self.logger_call()
                break
            
            if self.has_exponential_decay:
                self.scheduler_Adam.step()
            # -------------

        self.config['logger'].info("###-----------------------------------------###")
        self.config['logger'].info("### The best model is obtained in iteration = " + f"{self.best_iter}" + " with a total_loss = " + f"{self.min_error_for_checkpoint}.")



            
class Tester(nn.Module):
    
    def __init__(self, config, model):
        
        super().__init__()
        
        self.config = config
        self.model = model

        # TODO
        # PDE parameters
        # self.R = self.config['pde_param']['R'] 
        
    def test_full(self, dataloader, N_batches, branch_input, subset = 'test'):

        print("### TESTING ... ###")
        history_log = f"""LOG TEST - {subset}_dataset\n"""
       
        l2_relative_error_u = []
        
        dataloader_iterator = iter(dataloader)
        custom_bar = trange(N_batches)

        f_tensor = []
        num_samples = self.config['test']['batch_size']
        target_device = self.config['device']

        for key in branch_input.keys():
                
            base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]),dtype=torch.float32,device=target_device)
                
            final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
                
            f_tensor.append(final_view)

       
        for batch_iter in custom_bar:

            batch, batch_labels = next(dataloader_iterator)

            inputs = batch[:, 0:2].float().to(self.config['device'])
            outputs = batch[:, 2:3].float().to(self.config['device'])
            
            # REFERENCE
            u_ref = outputs
           

            # PREDICTED
            with torch.no_grad():
                
                tau = []
                for i, f_tensor_i in enumerate(f_tensor):
                    tau.append(self.model.branch_list[i](f_tensor_i))
        
                beta = self.model.trunk(inputs)

                ## np.tile
                tau[0] = tau[0].reshape(-1, 50)

                # Predicted
                u_pred = torch.sum(tau[0] * beta, axis = 1)[:, None]


            l2_relative_error_u.append(metric_l2_relative_error(exact = u_ref, pred = u_pred).cpu().numpy())

            
            # Save into a .txt file
            history_log = history_log + \
            "\n-------------------------\n" + \
            f"""Batch: {(batch_iter + 1)}/{(N_batches)} - 'l2_relative_error_vel': {l2_relative_error_u[batch_iter]} \n"""
         
            
        # Save into a .txt file
        with open(self.config['test_progress_file_path'], 'a') as file:
            file.write(history_log + '\n\n\n')

        self.config['logger'].info(f"Accuracy in Test - {subset}_dataset")
        self.config['logger'].info(f"L2 relative error in vel: {np.mean(np.array(l2_relative_error_u))}")
        
        print(f"Accuracy in {subset}dataset")
        print(f"L2 relative error in vel: {np.mean(np.array(l2_relative_error_u))}")

    def test_value(self, t, c):
        

        # Exact
        x_array = np.linspace(0, 1, 1000)[:,None]
        
        u_exact = exact_solution(x_array, t, c)
        
        u_exact_tensor = torch.tensor(u_exact, dtype=torch.float32, requires_grad = False).to(self.config['device'])

    
        # N = 1, P = batch_size
        t_array = np.repeat(np.array([t]), x_array.shape[0], axis=0)[:,None]

        xt_array = np.hstack([x_array, t_array])

        xt_tensor = torch.tensor(xt_array, dtype = torch.float32, requires_grad= False).to(self.config['device'])
        f_a_i = c
        f_a_array = np.repeat(np.array([f_a_i]),90)[None,:]
        f_a_array = np.repeat(f_a_array, x_array.shape[0], axis = 0)
        f_a_tensor = torch.tensor(f_a_array, dtype=torch.float32, requires_grad = False).to(self.config['device'])
        
        

        with torch.no_grad():
            tau = self.model.branch_list[0](f_a_tensor)
            beta = self.model.trunk(xt_tensor)
        u_pred_tensor = torch.sum(beta * tau, axis = 1)[:,None]

        
        l2_relative_error = torch.linalg.norm((u_exact_tensor-u_pred_tensor), 2)/torch.linalg.norm(u_exact_tensor, 2)
        print(f'l2_relative_error={l2_relative_error.item()} for t = {t} and c = {f_a_i}')

        u_pred = u_pred_tensor.cpu().numpy()

        return x_array, u_exact, u_pred
    
    def visualize_comparison(self, t_all, f_a_i, x_final, u_exact_final, u_pred_final):
       

        fig, ax = plt.subplots(figsize=(5, 5))

        # Create a color gradient for lines and points
        line_colors_exact = plt.cm.Blues(np.linspace(0.5, 1, len(t_all)))
        line_colors_pred = plt.cm.Reds(np.linspace(0.5, 1, len(t_all)))
        point_colors = plt.cm.Greens(np.linspace(0.5, 1, len(t_all)))

        for i,value in enumerate(t_all):
            
            ax.plot(x_final[i], u_exact_final[i], color=line_colors_exact[i], linewidth=3, label=f't = {value} and c = {f_a_i}')
            ax.plot(x_final[i], u_pred_final[i], '--',color=line_colors_pred[i], linewidth=3)
           

        # Setting up the plot
        ax.set_xlabel('x',fontsize=18)
        ax.set_ylabel('u',fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax.set_xlim(0, 1)
        # ax.set_ylim(0.8, 4)
        ax.set_ylim(-1, 1)
        # ax.set_ylim(0.4, 2.5)
        ax.legend()
        # ax.set_title('Exact Solution for Different a Values')

        # plt.show()

        # Save the figure
        fig_path = self.config['charts_folder_path'].joinpath(f'comparison_exact_vs_predicted_c_{f_a_i}.png')
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

