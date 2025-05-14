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


class Trainer_DeepONet(nn.Module):
    
    def __init__(self, config, model):
        
        pass



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
        
        # PDE parameters
        self.Wo = self.config['pde_param']['Wo']
        self.Re_dict = self.config['pde_param']['Re']

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
        self.l2_error_vel = None
        self.l2_error_pressure = None

        # Loss terms
        self.term_loss_tensor_dict = {key:[] for key in self.config['loss_terms']}

        # lambdas initialization
        self.term_lambdas_tensor = torch.tensor(
                                    [self.config['lambda_weights_init'][i] for i in range(len(self.config['loss_terms']))], 
                                    dtype=torch.float32, requires_grad=False).to(self.config['device'])

        self.term_lambdas_tensor_dict = {key: self.term_lambdas_tensor[i] for i,key in enumerate(self.config['loss_terms'])}

    
        # if self.has_loss_balancing:           
        #     self.loss_balancing_log = f"""Loss Balancing - {self.config['loss_balancing']['scheme']}\n"""
        
        # if self.has_random_weight_fact:
        #     self.term_grad_dict = {}
        #     for key in self.config['loss_terms']:
        #         self.term_grad_dict[key] = {}
        #         for n in range(config['neuralNet']['num_layers'] + 1):
        #             self.term_grad_dict[key][f'standard_dense_layer{2*n}'] = {'s': [],'v': [], 'bias': []}
            
        #     self.global_hat_lambdas = {key:[] for key in self.config['loss_terms']}

        #     if self.config["neuralNet"]["architecture"] == "Modified MLP":
        #         for key in self.config['loss_terms']:
        #             self.term_grad_dict[key].update({f'encoder_dense1_layer{0}' : {'s': [],'v': [], 'bias': []}})
        #             self.term_grad_dict[key].update({f'encoder_dense2_layer{0}' : {'s': [],'v': [], 'bias': []}})
        
        # else:
        #     self.term_grad_dict = {}
        #     for key in self.config['loss_terms']:
        #         self.term_grad_dict[key] = {}
        #         for n in range(config['neuralNet']['num_layers'] + 1):
        #             self.term_grad_dict[key][f'standard_dense_layer{2*n}'] = {'weight': [], 'bias': []}
            
        #     self.global_hat_lambdas = {key:[] for key in self.config['loss_terms']}

        #     if self.config["neuralNet"]["architecture"] == "Modified MLP":
        #         for key in self.config['loss_terms']:
        #             self.term_grad_dict[key].update({f'encoder_dense1_layer{0}' : {'weight': [], 'bias': []}})
        #             self.term_grad_dict[key].update({f'encoder_dense2_layer{0}' : {'weight': [], 'bias': []}})

    
        self.log_log = """LOG\n"""
        self.min_error_for_checkpoint = np.Infinity
        self.runtime0 = self.config['tick_start']
        self.best_iter = 0

    def loss_data(self):

        tau = []
        for i, f_data_tensor_i in enumerate(self.f_data_tensor):
            tau.append(self.model.branch_list[i](f_data_tensor_i))
        beta = self.model.trunk(self.xyzt_data_tensor)

        # Predicted
        if len(self.config['branches_control']['branch_list_ID']) == 1:
            v1_pred = torch.sum(tau[0][:,0:100] * beta[:,0:100], axis = 1)[:, None]
            v2_pred = torch.sum(tau[0][:,100:200] * beta[:,100:200], axis = 1)[:, None]
            v3_pred = torch.sum(tau[0][:,200:300] * beta[:,200:300], axis = 1)[:, None]
            p_pred = torch.sum(tau[0][:,300:400] * beta[:,300:400], axis = 1)[:, None]
        else:
            v1_pred = torch.sum(reduce(lambda x, y: x[:, 0:100] * y[:, 0:100], tau) * beta[:,0:100], axis = 1)[:, None]
            v2_pred = torch.sum(reduce(lambda x, y: x[:,100:200] * y[:,100:200], tau) * beta[:,100:200], axis = 1)[:, None]
            v3_pred = torch.sum(reduce(lambda x, y: x[:,200:300] * y[:,200:300], tau) * beta[:,200:300], axis = 1)[:, None]
            p_pred = torch.sum(reduce(lambda x, y: x[:,300:400] * y[:,300:400], tau) * beta[:,300:400], axis = 1)[:, None]

        # Ground Trust
        v1_GT = self.vel_data_tensor[:, 0:1]
        v2_GT = self.vel_data_tensor[:, 1:2]
        v3_GT = self.vel_data_tensor[:, 2:3]
        p_GT = self.pressure_data_tensor[:, 0:1]

        # Compute losses
        v1_loss = self.loss_fn(v1_pred, v1_GT)
        v2_loss = self.loss_fn(v2_pred, v2_GT)
        v3_loss = self.loss_fn(v3_pred, v3_GT)
        p_loss = self.loss_fn(p_pred, p_GT)

        loss = v1_loss + v2_loss + v3_loss + p_loss

        return loss
    
    def loss_initial_condition(self):

        tau = []
        for i, f_ic_tensor_i in enumerate(self.f_ic_tensor):
            tau.append(self.model.branch_list[i](f_ic_tensor_i))
 
        beta = self.model.trunk(self.xyzt_ic_tensor)

        # Predicted
        if len(self.config['branches_control']['branch_list_ID']) == 1:
            v1_pred = torch.sum(tau[0][:,0:100] * beta[:,0:100], axis = 1)[:, None]
            v2_pred = torch.sum(tau[0][:,100:200] * beta[:,100:200], axis = 1)[:, None]
            v3_pred = torch.sum(tau[0][:,200:300] * beta[:,200:300], axis = 1)[:, None]
        else:
            v1_pred = torch.sum(reduce(lambda x, y: x[:, 0:100] * y[:, 0:100], tau) * beta[:,0:100], axis = 1)[:, None]
            v2_pred = torch.sum(reduce(lambda x, y: x[:,100:200] * y[:,100:200], tau) * beta[:,100:200], axis = 1)[:, None]
            v3_pred = torch.sum(reduce(lambda x, y: x[:,200:300] * y[:,200:300], tau) * beta[:,200:300], axis = 1)[:, None]
        
        # Ground Trust
        v1_GT = self.vel_ic_tensor[:, 0:1]
        v2_GT = self.vel_ic_tensor[:, 1:2]
        v3_GT = self.vel_ic_tensor[:, 2:3]

        # Compute losses
        v1_loss = self.loss_fn(v1_pred, v1_GT)
        v2_loss = self.loss_fn(v2_pred, v2_GT)
        v3_loss = self.loss_fn(v3_pred, v3_GT)

        loss = v1_loss + v2_loss + v3_loss

        return loss
    
    def loss_boundary_inlet(self):

        tau = []
        for i, f_bc_inlet_tensor_i in enumerate(self.f_bc_inlet_tensor):
            tau.append(self.model.branch_list[i](f_bc_inlet_tensor_i))
 
        beta = self.model.trunk(self.xyzt_bc_inlet_tensor)

        # Predicted
        if len(self.config['branches_control']['branch_list_ID']) == 1:
            v1_pred = torch.sum(tau[0][:,0:100] * beta[:,0:100], axis = 1)[:, None]
            v2_pred = torch.sum(tau[0][:,100:200] * beta[:,100:200], axis = 1)[:, None]
            v3_pred = torch.sum(tau[0][:,200:300] * beta[:,200:300], axis = 1)[:, None]
        else:
            v1_pred = torch.sum(reduce(lambda x, y: x[:, 0:100] * y[:, 0:100], tau) * beta[:,0:100], axis = 1)[:, None]
            v2_pred = torch.sum(reduce(lambda x, y: x[:,100:200] * y[:,100:200], tau) * beta[:,100:200], axis = 1)[:, None]
            v3_pred = torch.sum(reduce(lambda x, y: x[:,200:300] * y[:,200:300], tau) * beta[:,200:300], axis = 1)[:, None]
        
        # Ground Trust
        v1_GT = self.vel_bc_inlet_tensor[:, 0:1]
        v2_GT = self.vel_bc_inlet_tensor[:, 1:2]
        v3_GT = self.vel_bc_inlet_tensor[:, 2:3]

        # Compute losses
        v1_loss = self.loss_fn(v1_pred, v1_GT)
        v2_loss = self.loss_fn(v2_pred, v2_GT)
        v3_loss = self.loss_fn(v3_pred, v3_GT)

        loss = v1_loss + v2_loss + v3_loss

        return loss
        
    def loss_boundary_wall(self):

        tau = []
        for i, f_bc_wall_tensor_i in enumerate(self.f_bc_wall_tensor):
            tau.append(self.model.branch_list[i](f_bc_wall_tensor_i))

        beta = self.model.trunk(self.xyzt_bc_wall_tensor)

        # Predicted
        if len(self.config['branches_control']['branch_list_ID']) == 1:
            v1_pred = torch.sum(tau[0][:,0:100] * beta[:,0:100], axis = 1)[:, None]
            v2_pred = torch.sum(tau[0][:,100:200] * beta[:,100:200], axis = 1)[:, None]
            v3_pred = torch.sum(tau[0][:,200:300] * beta[:,200:300], axis = 1)[:, None]
        else:
            v1_pred = torch.sum(reduce(lambda x, y: x[:, 0:100] * y[:, 0:100], tau) * beta[:,0:100], axis = 1)[:, None]
            v2_pred = torch.sum(reduce(lambda x, y: x[:,100:200] * y[:,100:200], tau) * beta[:,100:200], axis = 1)[:, None]
            v3_pred = torch.sum(reduce(lambda x, y: x[:,200:300] * y[:,200:300], tau) * beta[:,200:300], axis = 1)[:, None]
            
            
        # Ground Trust
        v1_GT = self.vel_bc_wall_tensor[:, 0:1]
        v2_GT = self.vel_bc_wall_tensor[:, 1:2]
        v3_GT = self.vel_bc_wall_tensor[:, 2:3]

        # Compute losses
        v1_loss = self.loss_fn(v1_pred, v1_GT)
        v2_loss = self.loss_fn(v2_pred, v2_GT)
        v3_loss = self.loss_fn(v3_pred, v3_GT)

        loss = v1_loss + v2_loss + v3_loss

        return loss

    def loss_boundary_outlet(self):
        
        tau = []
        for i, f_bc_outlet_tensor_i in enumerate(self.f_bc_outlet_tensor):
            tau.append(self.model.branch_list[i](f_bc_outlet_tensor_i))
        beta = self.model.trunk(self.xyzt_bc_outlet_tensor)
        

        # Predicted
        if len(self.config['branches_control']['branch_list_ID']) == 1:
            v1_pred = torch.sum(tau[0][:,0:100] * beta[:,0:100], axis = 1)[:, None]
            v2_pred = torch.sum(tau[0][:,100:200] * beta[:,100:200], axis = 1)[:, None]
            v3_pred = torch.sum(tau[0][:,200:300] * beta[:,200:300], axis = 1)[:, None]
        else:
            v1_pred = torch.sum(reduce(lambda x, y: x[:, 0:100] * y[:, 0:100], tau) * beta[:,0:100], axis = 1)[:, None]
            v2_pred = torch.sum(reduce(lambda x, y: x[:,100:200] * y[:,100:200], tau) * beta[:,100:200], axis = 1)[:, None]
            v3_pred = torch.sum(reduce(lambda x, y: x[:,200:300] * y[:,200:300], tau) * beta[:,200:300], axis = 1)[:, None]
            
            
        # Ground Trust
        v1_GT = self.vel_bc_outlet_tensor[:, 0:1]
        v2_GT = self.vel_bc_outlet_tensor[:, 1:2]
        v3_GT = self.vel_bc_outlet_tensor[:, 2:3]

        # Compute losses
        v1_loss = self.loss_fn(v1_pred, v1_GT)
        v2_loss = self.loss_fn(v2_pred, v2_GT)
        v3_loss = self.loss_fn(v3_pred, v3_GT)

        loss = v1_loss + v2_loss + v3_loss

        return loss

    def loss_physics(self):

        # forward pass
        tau = []
        for i, f_phy_tensor_i in enumerate(self.f_phy_tensor):
            tau.append(self.model.branch_list[i](f_phy_tensor_i))

        beta = self.model.trunk(torch.cat([self.x_phy, self.y_phy, self.z_phy, self.t_phy], 1))

        # Predicted
        if len(self.config['branches_control']['branch_list_ID']) == 1:
            v1 = torch.sum(tau[0][:,0:100] * beta[:,0:100], axis = 1)[:, None]
            v2 = torch.sum(tau[0][:,100:200] * beta[:,100:200], axis = 1)[:, None]
            v3 = torch.sum(tau[0][:,200:300] * beta[:,200:300], axis = 1)[:, None]
            p = torch.sum(tau[0][:,300:400] * beta[:,300:400], axis = 1)[:, None]
        else:
            v1 = torch.sum(reduce(lambda x, y: x[:, 0:100] * y[:, 0:100], tau) * beta[:,0:100], axis = 1)[:, None]
            v2 = torch.sum(reduce(lambda x, y: x[:,100:200] * y[:,100:200], tau) * beta[:,100:200], axis = 1)[:, None]
            v3 = torch.sum(reduce(lambda x, y: x[:,200:300] * y[:,200:300], tau) * beta[:,200:300], axis = 1)[:, None]
            p = torch.sum(reduce(lambda x, y: x[:,300:400] * y[:,300:400], tau) * beta[:,300:400], axis = 1)[:, None]
            
        # Autodiff
        v1_x = torch.autograd.grad(
            v1, self.x_phy, grad_outputs=torch.ones_like(v1), create_graph=True)[0]
        v1_y = torch.autograd.grad(
            v1, self.y_phy, grad_outputs=torch.ones_like(v1), create_graph=True)[0]
        v1_z = torch.autograd.grad(
            v1, self.z_phy, grad_outputs=torch.ones_like(v1), create_graph=True)[0]
        v1_t = torch.autograd.grad(
            v1, self.t_phy, grad_outputs=torch.ones_like(v1), create_graph=True)[0]
        

        v1_xx = torch.autograd.grad(
            v1_x, self.x_phy, grad_outputs=torch.ones_like(v1_x), create_graph=True)[0]
        v1_yy = torch.autograd.grad(
            v1_y, self.y_phy, grad_outputs=torch.ones_like(v1_y), create_graph=True)[0]
        v1_zz = torch.autograd.grad(
            v1_z, self.z_phy, grad_outputs=torch.ones_like(v1_z), create_graph=True)[0]

        v2_x = torch.autograd.grad(
            v2, self.x_phy, grad_outputs=torch.ones_like(v2), create_graph=True)[0]
        v2_y = torch.autograd.grad(
            v2, self.y_phy, grad_outputs=torch.ones_like(v2), create_graph=True)[0]
        v2_z = torch.autograd.grad(
            v2, self.z_phy, grad_outputs=torch.ones_like(v2), create_graph=True)[0]
        v2_t = torch.autograd.grad(
            v2, self.t_phy, grad_outputs=torch.ones_like(v2), create_graph=True)[0]
        

        v2_xx = torch.autograd.grad(
            v2_x, self.x_phy, grad_outputs=torch.ones_like(v2_x), create_graph=True)[0]
        v2_yy = torch.autograd.grad(
            v2_y, self.y_phy, grad_outputs=torch.ones_like(v2_y), create_graph=True)[0]
        v2_zz = torch.autograd.grad(
            v2_z, self.z_phy, grad_outputs=torch.ones_like(v2_z), create_graph=True)[0]

        v3_x = torch.autograd.grad(
            v3, self.x_phy, grad_outputs=torch.ones_like(v3), create_graph=True)[0]
        v3_y = torch.autograd.grad(
            v3, self.y_phy, grad_outputs=torch.ones_like(v3), create_graph=True)[0]
        v3_z = torch.autograd.grad(
            v3, self.z_phy, grad_outputs=torch.ones_like(v3), create_graph=True)[0]
        v3_t = torch.autograd.grad(
            v3, self.t_phy, grad_outputs=torch.ones_like(v3), create_graph=True)[0]
       

        v3_xx = torch.autograd.grad(
            v3_x, self.x_phy, grad_outputs=torch.ones_like(v3_x), create_graph=True)[0]
        v3_yy = torch.autograd.grad(
            v3_y, self.y_phy, grad_outputs=torch.ones_like(v3_y), create_graph=True)[0]
        v3_zz = torch.autograd.grad(
            v3_z, self.z_phy, grad_outputs=torch.ones_like(v3_z), create_graph=True)[0]

        p_x = torch.autograd.grad(
            p, self.x_phy, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(
            p, self.y_phy, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_z = torch.autograd.grad(
            p, self.z_phy, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        

        EQ1 = self.Wo**2/self.Re_tensor * (v1_t) + v1 * v1_x + v2 * v1_y + v3 * v1_z + p_x - 1/self.Re_tensor*(v1_xx + v1_yy + v1_zz)
        EQ2 = self.Wo**2/self.Re_tensor * (v2_t) + v1 * v2_x + v2 * v2_y + v3 * v2_z + p_y - 1/self.Re_tensor*(v2_xx + v2_yy + v2_zz)
        EQ3 = self.Wo**2/self.Re_tensor * (v3_t) + v1 * v3_x + v2 * v3_y + v3 * v3_z + p_z - 1/self.Re_tensor*(v3_xx + v3_yy + v3_zz)
        EQ4 = v1_x + v2_y + v3_z

        # Compute losses
        EQ1_loss = self.loss_fn(EQ1, self.residual_target)
        EQ2_loss = self.loss_fn(EQ2, self.residual_target)
        EQ3_loss = self.loss_fn(EQ3, self.residual_target)
        EQ4_loss = self.loss_fn(EQ4, self.residual_target)

    
        loss = EQ1_loss + EQ2_loss + EQ3_loss + EQ4_loss

        
        return loss
        
    def loss(self):

        # Forward pass and compute the losses per terms
        loss_ic = self.loss_initial_condition()
        loss_bc_inlet = self.loss_boundary_inlet()
        loss_bc_wall = self.loss_boundary_wall()
        loss_bc_outlet = self.loss_boundary_outlet()
        loss_data = self.loss_data()
        loss_p = self.loss_physics()

        self.term_loss_tensor_dict['ic'] = loss_ic
        self.term_loss_tensor_dict['bc_inlet'] = loss_bc_inlet
        self.term_loss_tensor_dict['bc_wall'] = loss_bc_wall
        self.term_loss_tensor_dict['bc_outlet'] = loss_bc_outlet
        self.term_loss_tensor_dict['data'] = loss_data
        self.term_loss_tensor_dict['phy'] = loss_p
        
        # self.term_loss_tensor_dict['ic'] = self.term_lambdas_tensor_dict['ic'] * loss_ic
        # self.term_loss_tensor_dict['bc_inlet'] = self.term_lambdas_tensor_dict['bc_inlet'] * loss_bc_inlet
        # self.term_loss_tensor_dict['bc_wall'] = self.term_lambdas_tensor_dict['bc_wall'] * loss_bc_wall
        # self.term_loss_tensor_dict['bc_outlet'] = self.term_lambdas_tensor_dict['bc_outlet'] * loss_bc_outlet
        # self.term_loss_tensor_dict['data'] = self.term_lambdas_tensor_dict['data'] * loss_data
        # self.term_loss_tensor_dict['phy'] = self.term_lambdas_tensor_dict['phy'] * loss_p


        # Apply adaptive lambdas
        self.loss_balancing_call()
        
        
        # Compute total loss
        loss_total = self.term_lambdas_tensor_dict['data'] * loss_data + self.term_lambdas_tensor_dict['ic'] * loss_ic + self.term_lambdas_tensor_dict['bc_inlet'] * loss_bc_inlet + self.term_lambdas_tensor_dict['bc_wall'] * loss_bc_wall  + self.term_lambdas_tensor_dict['bc_outlet'] * loss_bc_outlet + self.term_lambdas_tensor_dict['phy'] * loss_p 


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
        l2_error_vel = self.l2_error_vel
        l2_error_pressure = self.l2_error_pressure
        
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
f"""- total_loss_train: {loss_train} \n- l2_relative_error - velocity: {l2_error_vel} \n- l2_relative_error - pressure: {l2_error_pressure} \n\n- lr: {lr}\n\n""" 

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
        self.writer.add_scalar('l2_error_vel', self.l2_error_vel, self.regular_iter)
        self.writer.add_scalar('l2_error_pressure', self.l2_error_pressure, self.regular_iter)
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
            xyz_idx = np.random.choice(N_coord, N_coord, replace=False)  # Select all elements if N_coord < batch_size_coll
        else:
            xyz_idx = np.random.choice(N_coord, batch_size_coll, replace=False)
        xyz = coordinates[xyz_idx]

        #We will consider time values in the interval min(time_vector) and max(time_vector)
        t = np.sort(np.random.uniform(time_min, time_max, batch_size_coll))

        return xyz[:,0][:,None], xyz[:,1][:,None], xyz[:,2][:,None], t[:,None]

    def random_sampling(self, xyzt, vel, pressure, label = 'val'):
        """Return a random uniform sample of coordinate points from label with their respective velocities and pressure.
        """
        N_coord = xyzt.shape[0]
        idx = np.random.choice(N_coord,  round(self.config['train'][f'batch_dfraction'][label]*N_coord), replace=False)
        
        idx = np.sort(idx)
        xyzt_sample = xyzt[idx]
        vel_sample = vel[idx]
        pressure_sample= pressure[idx] 

        return xyzt_sample, vel_sample, pressure_sample
    
    def train(self, timePrm, full_ds, dataPrm, val_ds):
        
        print("### TRAINING ... ###")

        self.last_iter = self.config['train']['adam_steps'] + self.config['train']['lbfgs_steps'] - 1
        self.regular_iter = 0

        time_vector = timePrm.time_vector[:,None]

        N_vel_branch_inlet = len(self.config['branches_control']['vel_axis_ID'])
        has_outlet_pressure = len(self.config['branches_control']['branch_input_ID']) > len(self.config['branches_control']['vel_axis_ID'])

        xyzt_bc_inlet = {} 
        vel_bc_inlet = {} 
        # pressure_bc_inlet = {} 

        xyzt_bc_outlet = {} 
        vel_bc_outlet = {} 
        pressure_bc_outlet = {} 

        xyzt_bc_wall = {} 
        vel_bc_wall = {} 
        # pressure_bc_wall = {} 

        xyzt_ic = {} 
        vel_ic = {} 
        # pressure_ic = {}

        xyzt_data = {} 
        vel_data = {} 
        pressure_data = {}

        xyzt_val = {} 
        vel_val = {}
        pressure_val = {}

        Re_list = []

        for chosen_flow_label in self.config['train']['chosen_flow_labels']:

            # stratified_val_ds[chosen_flow_label] = craft_validation_dataset_deeponet(val_ds[chosen_flow_label], time_vector)

            #---------------------
            # Fixed dataset
            #---------------------
            ## BOUNDARY CONDITIONS AND INITIAL CONDITION
            xyzt_bc_inlet[chosen_flow_label], vel_bc_inlet[chosen_flow_label], _, xyzt_bc_outlet[chosen_flow_label], vel_bc_outlet[chosen_flow_label], pressure_bc_outlet[chosen_flow_label], xyzt_bc_wall[chosen_flow_label], vel_bc_wall[chosen_flow_label], _, xyzt_ic[chosen_flow_label], vel_ic[chosen_flow_label], _ = craft_bc_and_ic_dataset(full_ds[chosen_flow_label], time_vector)
            # xyzt_bc_inlet[chosen_flow_label], vel_bc_inlet[chosen_flow_label], pressure_bc_inlet[chosen_flow_label], xyzt_bc_outlet[chosen_flow_label], vel_bc_outlet[chosen_flow_label], pressure_bc_outlet[chosen_flow_label], xyzt_bc_wall[chosen_flow_label], vel_bc_wall[chosen_flow_label], pressure_bc_wall[chosen_flow_label], xyzt_ic[chosen_flow_label], vel_ic[chosen_flow_label], pressure_ic[chosen_flow_label] = craft_bc_and_ic_dataset(full_ds[chosen_flow_label], time_vector)

            ## DATA 
            xyzt_data[chosen_flow_label], vel_data[chosen_flow_label], pressure_data[chosen_flow_label] = build_stratum_dataset(dataPrm[chosen_flow_label], time_vector)

            ## VALIDATION DATASET
            xyzt_val[chosen_flow_label], vel_val[chosen_flow_label], pressure_val[chosen_flow_label] = craft_validation_dataset(val_ds[chosen_flow_label], time_vector)

            Re_list.append(self.Re_dict[chosen_flow_label])

        # """The coordinate points are the same between the different datasets, so we can choose a geometry among the options"""
        coordinates, time_min, time_max = get_coordinates_for_generator(full_ds[self.config['train']['chosen_flow_labels'][0]], time_vector)

        # Reynolds number
        Re_array = np.tile(np.array(Re_list), self.config['train']['batch_size_coll'])
        self.Re_tensor = torch.tensor(Re_array[:,None], dtype = torch.float32, requires_grad= False).to(self.config['device'])

        # Workflow
        self.custom_bar = trange(self.last_iter + 1)
        
        for regular_iter in self.custom_bar:

            self.regular_iter = regular_iter

            # Branch
            branch_input = {}
            for key in self.config['branches_control']['branch_input_ID']:
                branch_input.update({key:[]})

            # Trunk
            vel_bc_inlet_target = []
            vel_bc_outlet_target = []
            vel_bc_wall_target = []
            vel_ic_target = []

            vel_data_target = []
            pressure_data_target = []
            
            # Val
            vel_val_target = []
            pressure_val_target = []

            for chosen_flow_label in self.config['train']['chosen_flow_labels']:
                
                ## Random Sampling
                # INLET 
                xyzt_bc_inlet_sample, vel_bc_inlet_sample, _ = self.random_sampling(xyzt_bc_inlet[chosen_flow_label], vel_bc_inlet[chosen_flow_label], NullContainer(), label = 'bc-inlet')
                # xyzt_bc_inlet_sample, vel_bc_inlet_sample, _ = self.random_sampling(xyzt_bc_inlet[chosen_flow_label], vel_bc_inlet[chosen_flow_label], pressure_bc_inlet[chosen_flow_label], label = 'bc-inlet')
                # OUTLET
                xyzt_bc_outlet_sample, vel_bc_outlet_sample, pressure_bc_outlet_sample = self.random_sampling(xyzt_bc_outlet[chosen_flow_label], vel_bc_outlet[chosen_flow_label], pressure_bc_outlet[chosen_flow_label], label = 'bc-outlet')
                # WALL 
                xyzt_bc_wall_sample, vel_bc_wall_sample, _ = self.random_sampling(xyzt_bc_wall[chosen_flow_label], vel_bc_wall[chosen_flow_label], NullContainer(), label = 'bc-wall')
                # xyzt_bc_wall_sample, vel_bc_wall_sample, _ = self.random_sampling(xyzt_bc_wall[chosen_flow_label], vel_bc_wall[chosen_flow_label], pressure_bc_wall[chosen_flow_label], label = 'bc-wall')
                
                # IC 
                xyzt_ic_sample, vel_ic_sample, _ = self.random_sampling(xyzt_ic[chosen_flow_label], vel_ic[chosen_flow_label], NullContainer(), label = 'ic')
                # xyzt_ic_sample, vel_ic_sample, _ = self.random_sampling(xyzt_ic[chosen_flow_label], vel_ic[chosen_flow_label], pressure_ic[chosen_flow_label], label = 'ic')
                
                # Data 
                xyzt_data_sample, vel_data_sample, pressure_data_sample = self.random_sampling(xyzt_data[chosen_flow_label], vel_data[chosen_flow_label], pressure_data[chosen_flow_label], label = 'data')

                # VAL 
                xyzt_val_sample, vel_val_sample, pressure_val_sample = self.random_sampling(xyzt_val[chosen_flow_label], vel_val[chosen_flow_label], pressure_val[chosen_flow_label], label = 'val')


                # Inlet
                # inlet_vel = vel_bc_inlet_sample
                # Outlet
                # outlet_vel = vel_bc_outlet_sample
                # outlet_pressure = pressure_bc_outlet_sample
                # Wall
                # wall_vel = vel_bc_wall_sample
                # IC
                # ic_vel = vel_ic_sample
                # Data
                # data_vel = vel_data_sample
                # data_pressure = pressure_data_sample
                # Val
                # val_vel = vel_val_sample 
                # val_pressure = pressure_val_sample
                
                ## Branch
                for i in range(N_vel_branch_inlet):
                    index = self.config['branches_control']['axis_indexes'][self.config['branches_control']['vel_axis_ID'][i]]
                    branch_input[self.config['branches_control']['branch_input_ID'][i]].append(vel_bc_inlet_sample[:,index].T)
                
                if has_outlet_pressure:
                    branch_input[self.config['branches_control']['branch_input_ID'][-1]].append(pressure_bc_outlet_sample[:,0].T)
                
                ## Trunk
                vel_bc_inlet_target.append(vel_bc_inlet_sample)
                vel_bc_outlet_target.append(vel_bc_outlet_sample)
                vel_bc_wall_target.append(vel_bc_wall_sample)
                vel_ic_target.append(vel_ic_sample)

                vel_data_target.append(vel_data_sample)
                pressure_data_target.append(pressure_data_sample)

                vel_val_target.append(vel_val_sample )
                pressure_val_target.append(pressure_val_sample)

                # # Coordinates
                # """The coordinate points are the same between the different datasets, so we can choose a geometry among the options"""
                # xyzt_bc_inlet_fixed = xyzt_bc_inlet_sample
                # xyzt_bc_outlet_fixed = xyzt_bc_outlet_sample
                # xyzt_bc_wall_fixed = xyzt_bc_wall_sample
                # xyzt_ic_fixed = xyzt_ic_sample
                # xyzt_data_fixed = xyzt_data_sample
                # xyzt_val_fixed = xyzt_val_sample
            

            # Trunk
            vel_bc_inlet_target = np.array(vel_bc_inlet_target)
            vel_bc_outlet_target = np.array(vel_bc_outlet_target)
            vel_bc_wall_target = np.array(vel_bc_wall_target)
            vel_ic_target = np.array(vel_ic_target)

            vel_data_target = np.array(vel_data_target)
            pressure_data_target = np.array(pressure_data_target)

            vel_val_target = np.array(vel_val_target)
            pressure_val_target = np.array(pressure_val_target)
            
            # INLET 
            # xyzt_bc_inlet = inletPrm.coordinates
            # N = 4, P = 518 (self.config['train']['fixed_bc_size']['inlet'])
            xyzt_bc_inlet_array = np.repeat(xyzt_bc_inlet_sample, self.config['train']['n_input_functions'], axis=0)
            self.xyzt_bc_inlet_tensor = torch.tensor(xyzt_bc_inlet_array, dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            self.f_bc_inlet_tensor = []
            for key in branch_input.keys():
                f_bc_inlet_array = np.tile(np.array(branch_input[key]), (xyzt_bc_inlet_sample.shape[0],1))
                self.f_bc_inlet_tensor.append(torch.tensor(f_bc_inlet_array, dtype = torch.float32, requires_grad= False).to(self.config['device']))
            
            self.vel_bc_inlet_tensor = torch.tensor(np.swapaxes(vel_bc_inlet_target,0,1).reshape(-1,3), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            # WALL 
            # xyz_bc_wall = wallPrm.coordinates
            # N = 4, P = 19810 (self.config['train']['fixed_bc_size']['wall'])
            xyzt_bc_wall_array = np.repeat(xyzt_bc_wall_sample, self.config['train']['n_input_functions'], axis=0)
            self.xyzt_bc_wall_tensor = torch.tensor(xyzt_bc_wall_array, dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
        
            self.f_bc_wall_tensor = []
            for key in branch_input.keys():
                f_bc_wall_array = np.tile(np.array(branch_input[key]), (xyzt_bc_wall_sample.shape[0],1))
                self.f_bc_wall_tensor.append(torch.tensor(f_bc_wall_array, dtype = torch.float32, requires_grad= False).to(self.config['device']))
            
            self.vel_bc_wall_tensor = torch.tensor(np.swapaxes(vel_bc_wall_target,0,1).reshape(-1,3), dtype = torch.float32, requires_grad= False).to(self.config['device'])
        
        
            # OUTLET
            # xyz_bc_outlet = outletPrm.coordinates
            # N = 4, P = 720 (self.config['train']['fixed_bc_size']['outlet'])
            xyzt_bc_outlet_array = np.repeat(xyzt_bc_outlet_sample, self.config['train']['n_input_functions'], axis=0)
            self.xyzt_bc_outlet_tensor = torch.tensor(xyzt_bc_outlet_array, dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            self.f_bc_outlet_tensor = []
            for key in branch_input.keys():
                f_bc_outlet_array = np.tile(np.array(branch_input[key]), (xyzt_bc_outlet_sample.shape[0],1))
                self.f_bc_outlet_tensor.append(torch.tensor(f_bc_outlet_array, dtype = torch.float32, requires_grad= False).to(self.config['device']))
            
            self.vel_bc_outlet_tensor = torch.tensor(np.swapaxes(vel_bc_outlet_target,0,1).reshape(-1,3), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            # IC
            # N = 4, P = 1800 
            xyzt_ic_array = np.repeat(xyzt_ic_sample, self.config['train']['n_input_functions'], axis=0)
            self.xyzt_ic_tensor = torch.tensor(xyzt_ic_array, dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            self.f_ic_tensor = []
            for key in branch_input.keys():
                f_ic_array = np.tile(np.array(branch_input[key]), (xyzt_ic_sample.shape[0],1))
                self.f_ic_tensor.append(torch.tensor(f_ic_array, dtype = torch.float32, requires_grad= False).to(self.config['device']))
            
            self.vel_ic_tensor = torch.tensor(np.swapaxes(vel_ic_target,0,1).reshape(-1,3), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            # DATA
            # N = 4, P = 1800 
            xyzt_data_array = np.repeat(xyzt_data_sample, self.config['train']['n_input_functions'], axis=0)
            self.xyzt_data_tensor = torch.tensor(xyzt_data_array, dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            self.f_data_tensor = []
            for key in branch_input.keys():
                f_data_array = np.tile(np.array(branch_input[key]), (xyzt_data_sample.shape[0],1))
                self.f_data_tensor.append(torch.tensor(f_data_array, dtype = torch.float32, requires_grad= False).to(self.config['device']))
            
            self.vel_data_tensor = torch.tensor(np.swapaxes(vel_data_target,0,1).reshape(-1,3), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            self.pressure_data_tensor = torch.tensor(np.swapaxes(pressure_data_target,0,1).reshape(-1,1), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            
            # Validation dataset
            # N = 5, P = xyzt_val_fixed.shape[0]
            xyzt_val_array = np.repeat(xyzt_val_sample, self.config['train']['n_input_functions'], axis=0)
            self.xyzt_val_tensor = torch.tensor(xyzt_val_array, dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            self.f_val_tensor = []
            for key in branch_input.keys():
                f_val_array = np.tile(np.array(branch_input[key]), (xyzt_val_sample.shape[0],1))
                self.f_val_tensor.append(torch.tensor(f_val_array, dtype = torch.float32, requires_grad= False).to(self.config['device']))
            
            self.vel_val_tensor = torch.tensor(np.swapaxes(vel_val_target,0,1).reshape(-1,3), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            self.pressure_val_tensor = torch.tensor(np.swapaxes(pressure_val_target,0,1).reshape(-1,1), dtype = torch.float32, requires_grad= False).to(self.config['device'])

            ## COLLOCATION POINTS
            # Random sample
            x_phy, y_phy, z_phy, t_phy = self.collocation_points_generator(self.config['train']['batch_size_coll'], coordinates, time_min, time_max)

            # N = 4, P = batch_size_coll
            x_phy_array = np.repeat(x_phy, self.config['train']['n_input_functions'], axis=0)
            y_phy_array = np.repeat(y_phy, self.config['train']['n_input_functions'], axis=0)
            z_phy_array = np.repeat(z_phy, self.config['train']['n_input_functions'], axis=0)
            t_phy_array = np.repeat(t_phy, self.config['train']['n_input_functions'], axis=0)

            self.x_phy = torch.tensor(x_phy_array, dtype = torch.float32, requires_grad= True).to(self.config['device'])
            self.y_phy = torch.tensor(y_phy_array, dtype = torch.float32, requires_grad= True).to(self.config['device'])
            self.z_phy = torch.tensor(z_phy_array, dtype = torch.float32, requires_grad= True).to(self.config['device'])
            self.t_phy = torch.tensor(z_phy_array, dtype = torch.float32, requires_grad= True).to(self.config['device'])

            
            self.f_phy_tensor = []
            for key in branch_input.keys():
                f_phy_array = np.tile(np.array(branch_input[key]), (self.config['train']['batch_size_coll'],1))
                self.f_phy_tensor.append(torch.tensor(f_phy_array, dtype = torch.float32, requires_grad= False).to(self.config['device']))
            
            self.residual_target = torch.zeros((self.x_phy.shape[0],1), dtype = torch.float32, requires_grad= False).to(self.config['device'])

        
            # Compute total loss, apply backward pass and optimize
            loss_total = self.compute_loss_total_and_backward()
            self.optimizer_Adam.step()

            # Store batch losses
            self.total_loss = loss_total.item()


            # Compute l2 relative error with Validation dataset
            ## REFERENCE
            vel_ref = torch.sqrt(torch.sum(self.vel_val_tensor**2, axis=1))
            p_ref =  self.pressure_val_tensor

            ## PREDICTED
            with torch.no_grad():
                tau = []
                for i, f_star_tensor_i in enumerate(self.f_val_tensor):
                    tau.append(self.model.branch_list[i](f_star_tensor_i))
                
                beta = self.model.trunk(self.xyzt_val_tensor)

            if len(self.config['branches_control']['branch_list_ID']) == 1:
                v1 = torch.sum(tau[0][:,0:100] * beta[:,0:100], axis = 1)[:, None]
                v2 = torch.sum(tau[0][:,100:200] * beta[:,100:200], axis = 1)[:, None]
                v3 = torch.sum(tau[0][:,200:300] * beta[:,200:300], axis = 1)[:, None]
                p_pred = torch.sum(tau[0][:,300:400] * beta[:,300:400], axis = 1)[:, None]
            else:    
                v1 = torch.sum(reduce(lambda x, y: x[:,0:100] * y[:,0:100], tau) * beta[:,0:100], axis = 1)[:, None]
                v2 = torch.sum(reduce(lambda x, y: x[:,100:200] * y[:,100:200], tau) * beta[:,100:200], axis = 1)[:, None]
                v3 = torch.sum(reduce(lambda x, y: x[:,200:300] * y[:,200:300], tau) * beta[:,200:300], axis = 1)[:, None]
                p_pred = torch.sum(reduce(lambda x, y: x[:,300:400] * y[:,300:400], tau) * beta[:,300:400], axis = 1)[:, None]
               
            vel = torch.cat((v1,v2,v3), axis = 1)
            vel_pred = torch.sqrt(torch.sum(vel**2, axis=1))

            
            l2_relative_error_vel = metric_l2_relative_error(exact = vel_ref, pred = vel_pred)
            l2_relative_error_pressure = metric_l2_relative_error(exact = p_ref, pred = p_pred)
           
            # Store errors
            self.l2_error_vel = l2_relative_error_vel.item()
            self.l2_error_pressure = l2_relative_error_pressure.item()

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
        
    def test(self, dataloader, N_batches, branch_input, subset = 'test'):

        print("### TESTING ... ###")
        history_log = f"""LOG TEST - {subset}_dataset\n"""
       
        l2_relative_error_u = []
        
        dataloader_iterator = iter(dataloader)
        custom_bar = trange(N_batches)

        f_tensor = []
        for key in branch_input.keys():
            f_array = np.tile(np.array(branch_input[key]), (self.config['test']['batch_size'],1))
            f_tensor.append(torch.tensor(f_array, dtype = torch.float32, requires_grad= False).to(self.config['device']))

        # # PREDICTED
        # idx_final = []
        # for i, f_tensor_i in enumerate(f_tensor):
        #     N_coord = f_tensor_i.shape[1]
        #     idx = np.random.choice(N_coord,  round(0.01*N_coord), replace=False)
        #     idx_final.append(np.sort(idx))

        for batch_iter in custom_bar:

            batch, batch_labels = next(dataloader_iterator)

            inputs = batch[:, 0:2].float().to(self.config['device'])
            outputs = batch[:, 2:3].float().to(self.config['device'])
            
            # REFERENCE
            u_ref = outputs
           

            # ----
            # # Data transformation: From dimension to dimensionless
            # inputs[:,0:3] = inputs[:,0:3]/self.config['pde_param']['2R_pipe']
            # inputs[:,3:4] = self.config['pde_param']['omega']*inputs[:,3:4]

            

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
            # f"""Batch: {(batch_iter + 1)}/{(N_batches)} - 'l2_relative_error_vel': {l2_relative_error_vel[batch_iter]} - 'l2_relative_error_pressure': {l2_relative_error_pressure[batch_iter] } - 'pressure shift value': {shift_value.item()}\n"""
            
        # Save into a .txt file
        with open(self.config['test_progress_file_path'], 'a') as file:
            file.write(history_log + '\n\n\n')

        self.config['logger'].info(f"Accuracy in Test - {subset}_dataset")
        self.config['logger'].info(f"L2 relative error in vel: {np.mean(np.array(l2_relative_error_u))}")
        
        print(f"Accuracy in {subset}dataset")
        print(f"L2 relative error in vel: {np.mean(np.array(l2_relative_error_u))}")
    


    def visualize_absolute_error_in_cross_sections(self, ds, y0, subset = 'full'):
        
        # Ground Trust
        inletPrm = ds.inletPrm
        outletPrm = ds.outletPrm
        wallPrm = ds.wallPrm
        volumePrm = ds.volumePrm

        coordinates_all = np.vstack([inletPrm.coordinates, outletPrm.coordinates, wallPrm.coordinates, volumePrm.coordinates])
        vel_all = np.vstack([inletPrm.vel, outletPrm.vel, wallPrm.vel, volumePrm.vel])
        pressure_all = np.vstack([inletPrm.pressure, outletPrm.pressure, wallPrm.pressure, volumePrm.pressure])

        # Intersection plane - 3d geometry
        tol = 0.00025
        intersection_xyz, corresponding_vel, corresponding_pressure = cross_section_plane_intersection(coordinates_all, vel_all, pressure_all, y0, tolerance=tol)
        
        # Grid in  y = y0 plane
        x_min = min(intersection_xyz[:, 0])
        x_max = max(intersection_xyz[:, 0])
        z_min = min(intersection_xyz[:, 2])
        z_max = max(intersection_xyz[:, 2])
        grid_x, grid_z = np.mgrid[x_min:x_max:500j, z_min:z_max:500j]
        
        # REFERENCE
        # Calculate the magnitude of the velocities
        vel_ref = np.sqrt(np.sum(corresponding_vel**2, axis=1))
        pressure_ref = np.squeeze(corresponding_pressure)
        
        # Interpolate vel onto the grid
        vel_ref_grid = griddata(intersection_xyz[:, [0, 2]], vel_ref, (grid_x, grid_z), method='cubic')
        pressure_ref_grid = griddata(intersection_xyz[:, [0, 2]], pressure_ref, (grid_x, grid_z), method='cubic')

    
        # PREDICTED
        # Compute predicted in the grid
        x_star = grid_x.flatten()[:,None]
        y_star = y0 * np.ones_like(x_star)
        z_star = grid_z.flatten()[:,None]

        xyz_star = np.hstack([x_star, y_star, z_star])
        xyz_star_tensor = torch.tensor(xyz_star, dtype = torch.float32, requires_grad= False).to(self.config['device'])
        
        # ----
        # Data transformation: From dimension to dimensionless
        xyz_star_tensor_dimless = xyz_star_tensor/self.config['pde_param']['2R_pipe']

        with torch.no_grad():
            NN_pred_tensor_dimless = self.forward(xyz_star_tensor_dimless)

        # Update parameters based on inlet velocity
        self.config['pde_param']['V_inlet'] = self.config['test']['vel_max_inlet']
        self.config['pde_param']['Re'] = self.config['pde_param']['rho_f']*self.config['pde_param']['2R_pipe']*self.config['pde_param']['V_inlet']/self.config['pde_param']['mu_f'] 

        # Data transformation: From dimensionless to dimension
        NN_pred_tensor = torch.zeros_like(NN_pred_tensor_dimless)
        NN_pred_tensor[:,0:3] = NN_pred_tensor_dimless[:,0:3]*self.config['pde_param']['V_inlet']
        NN_pred_tensor[:,3:4] = NN_pred_tensor_dimless[:,3:4]*(self.config['pde_param']['rho_f'] * self.config['pde_param']['V_inlet']**2)
        # ----

        vel_pred_grid_aux = torch.sqrt(torch.sum(NN_pred_tensor[:,0:3]**2, axis=1))
        pressure_pred_grid_aux = NN_pred_tensor[:,3:4]

        vel_pred_grid= vel_pred_grid_aux.cpu().numpy().reshape(grid_x.shape[0],grid_z.shape[0]).T
        pressure_pred_grid = pressure_pred_grid_aux.cpu().numpy().reshape(grid_x.shape[0],grid_z.shape[0]).T

        # Pressure shift adjustment
        shift_all =  pressure_pred_grid - pressure_ref_grid
        shift_value = np.mean(shift_all[~np.isnan(shift_all)])
        pressure_pred_grid = pressure_pred_grid - shift_value

        # Plotting purposes
        # Fill points outside the ellipse with NaN 
        ellipse = grid_x**2/((x_max-x_min)/2)**2 + grid_z**2/((z_max-z_min)/2)**2   
        outside_ellipse = ellipse > 1

        vel_pred_grid_for_plot = copy.deepcopy(vel_pred_grid)
        vel_pred_grid_for_plot[outside_ellipse] = np.nan
        pressure_pred_grid_for_plot = copy.deepcopy(pressure_pred_grid)
        pressure_pred_grid_for_plot[outside_ellipse] = np.nan

        ## Velocity
        # Calculate the minimum and maximum values for the exact and predicted vel
        vmin_vel = min(np.nanmin(vel_ref_grid), np.nanmin(vel_pred_grid_for_plot))
        vmax_vel = max(np.nanmax(vel_ref_grid), np.nanmax(vel_pred_grid_for_plot))

        # Plot
        fig = plt.figure(figsize=(20, 5))

        plt.subplot(1, 3, 1)
        plt.pcolor(grid_x, grid_z, vel_ref_grid, cmap="jet", vmin=0, vmax=0.1)
        cbar1 = plt.colorbar()
        cbar1.set_ticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.1])
        cbar1.set_label(r'$|\boldsymbol{v}(\boldsymbol{x})|$ [m/s]', fontsize=24) 
        cbar1.ax.tick_params(labelsize=18, colors='black')
        plt.xlabel(r"$x_1$ [m]",fontsize=24)
        plt.ylabel(r"$x_3$ [m]",fontsize=24)
        plt.xticks(fontsize= 18)
        plt.yticks(fontsize= 18)
        
        plt.xticks(np.linspace(np.round(np.min(grid_x),3), np.round(np.max(grid_x),3), 5))  
        plt.xlim([np.round(np.min(grid_x),3), np.round(np.max(grid_x),3)]) 

        plt.yticks(np.linspace(np.round(np.min(grid_z),3), np.round(np.max(grid_z),3), 5))  
        plt.ylim([np.round(np.min(grid_z),3), np.round(np.max(grid_z),3)]) 

        plt.title("Ground trust",fontsize=20)
        plt.tight_layout()


        plt.subplot(1, 3, 2)
        plt.pcolor(grid_x, grid_z, vel_pred_grid_for_plot, cmap="jet", vmin=0, vmax=0.1)
        cbar2 = plt.colorbar()
        cbar2.set_ticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.1])
        cbar2.set_label(r'$|\boldsymbol{\hat{v}}_{\theta}(\boldsymbol{x})|$ [m/s]', fontsize=24) 
        cbar2.ax.tick_params(labelsize=18, colors='black')
        plt.xlabel(r"$x_1$ [m]",fontsize=24)
        plt.ylabel(r"$x_3$ [m]",fontsize=24)
        plt.xticks(fontsize= 18)
        plt.yticks(fontsize= 18)
        
        plt.xticks(np.linspace(np.round(np.min(grid_x),3), np.round(np.max(grid_x),3), 5))  
        plt.xlim([np.round(np.min(grid_x),3), np.round(np.max(grid_x),3)]) 

        plt.yticks(np.linspace(np.round(np.min(grid_z),3), np.round(np.max(grid_z),3), 5))  
        plt.ylim([np.round(np.min(grid_z),3), np.round(np.max(grid_z),3)]) 

        plt.title("Predicted",fontsize=20)
        plt.tight_layout()


        abs_error_vel = np.empty_like(vel_ref_grid)
        abs_error_vel[:,:] = np.nan 
        abs_error_vel[~np.isnan(vel_ref_grid)] = abs(vel_ref_grid[~np.isnan(vel_ref_grid)] - vel_pred_grid[~np.isnan(vel_ref_grid)])
        plt.subplot(1, 3, 3)
        plt.pcolor(grid_x, grid_z, abs_error_vel, cmap="jet")
        cbar3 = plt.colorbar()
        cbar3.set_label(r'$abs(|\boldsymbol{v}(\boldsymbol{x})|-|\boldsymbol{\hat{v}}_{\theta}(\boldsymbol{x})|)$ [m/s]', fontsize=24) 
        cbar3.ax.tick_params(labelsize=18, colors='black')
        plt.xlabel(r"$x_1$ [m]",fontsize=24)
        plt.ylabel(r"$x_3$ [m]",fontsize=24)
        plt.xticks(fontsize= 18)
        plt.yticks(fontsize= 18)

        plt.xticks(np.linspace(np.round(np.min(grid_x),3), np.round(np.max(grid_x),3), 5))  
        plt.xlim([np.round(np.min(grid_x),3), np.round(np.max(grid_x),3)]) 

        plt.yticks(np.linspace(np.round(np.min(grid_z),3), np.round(np.max(grid_z),3), 5))  
        plt.ylim([np.round(np.min(grid_z),3), np.round(np.max(grid_z),3)]) 

        plt.title(f"Absolute error",fontsize=20)
        plt.tight_layout()

        plt.subplots_adjust(wspace=0.5) 

        # Save the figure
        fig_path = self.config['charts_folder_path'].joinpath(f'vel_absolute_error_in_y_{y0}.png')
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        
        ## Pressure
        # Calculate the minimum and maximum values for the exact and predicted pressure
        vmin_p = min(np.nanmin(pressure_ref_grid), np.nanmin(pressure_pred_grid_for_plot))
        vmax_p = max(np.nanmax(pressure_ref_grid), np.nanmax(pressure_pred_grid_for_plot))


        # Plot
        fig1 = plt.figure(figsize=(20, 6))
        plt.subplot(1, 3, 1)
        plt.pcolor(grid_x, grid_z, pressure_ref_grid, cmap="jet", vmin=vmin_p, vmax=vmax_p)
        plt.colorbar()
        plt.xlabel("x (m)",fontsize=20)
        plt.ylabel("z (m)",fontsize=20)
        plt.title(f"Reference p (Pa) in y={y0}",fontsize=20)
        plt.tight_layout()

        plt.subplot(1, 3, 2)
        plt.pcolor(grid_x, grid_z, pressure_pred_grid_for_plot, cmap="jet", vmin=vmin_p, vmax=vmax_p)
        plt.colorbar()
        plt.xlabel("x (m)",fontsize=20)
        plt.ylabel("z (m)",fontsize=20)
        plt.title(f"Predicted p (Pa) in y={y0}",fontsize=20)
        plt.tight_layout()

        abs_error_pressure = np.empty_like(pressure_ref_grid)
        abs_error_pressure[:,:] = np.nan 
        abs_error_pressure[~np.isnan(pressure_ref_grid)] = abs(pressure_ref_grid[~np.isnan(pressure_ref_grid)] - pressure_pred_grid[~np.isnan(pressure_ref_grid)])
        plt.subplot(1, 3, 3)
        plt.pcolor(grid_x, grid_z, abs_error_pressure, cmap="jet")
        plt.colorbar()
        plt.xlabel("x (m)",fontsize=20)
        plt.ylabel("z (m)",fontsize=20)
        plt.title(f"Absolute error of p (Pa) in y={y0}",fontsize=20)
        plt.tight_layout()

        # Save the figure
        fig1_path2 = self.config['charts_folder_path'].joinpath(f'pressure_absolute_error_in_y_{y0}.png')
        fig1.savefig(fig1_path2, bbox_inches="tight", dpi=300)
        plt.close(fig1)


        self.config['logger'].info("#-----------------------------------------#")
        self.config['logger'].info(f"Accuracy in plane y={y0} - {subset}_dataset")
        
        self.config['logger'].info(f"max absolute error in magnitude of vel: {np.max(abs_error_vel[~np.isnan(vel_ref_grid)])}")
        self.config['logger'].info(f"min absolute error in magnitude of vel: {np.min(abs_error_vel[~np.isnan(vel_ref_grid)])}")
        self.config['logger'].info(f"mean absolute error in magnitude of vel: {np.mean(abs_error_vel[~np.isnan(vel_ref_grid)])}")
        
        self.config['logger'].info(f"max absolute error in pressure: {np.max(abs_error_pressure[~np.isnan(pressure_ref_grid)])}")
        self.config['logger'].info(f"min absolute error in pressure: {np.min(abs_error_pressure[~np.isnan(pressure_ref_grid)])}")
        self.config['logger'].info(f"mean absolute error in pressure: {np.mean(abs_error_pressure[~np.isnan(pressure_ref_grid)])}")
        
        print("#-----------------------------------------#")
        print(f"Accuracy in plane y={y0}  - {subset}_dataset")
        
        print(f"max absolute error in magnitude of vel: {np.max(abs_error_vel[~np.isnan(vel_ref_grid)])}")
        print(f"min absolute error in magnitude of vel: {np.min(abs_error_vel[~np.isnan(vel_ref_grid)])}")
        print(f"mean absolute error in magnitude of vel: {np.mean(abs_error_vel[~np.isnan(vel_ref_grid)])}")
        
        print(f"max absolute error in pressure: {np.max(abs_error_pressure[~np.isnan(pressure_ref_grid)])}")
        print(f"min absolute error in pressure: {np.min(abs_error_pressure[~np.isnan(pressure_ref_grid)])}")
        print(f"mean absolute error in pressure: {np.mean(abs_error_pressure[~np.isnan(pressure_ref_grid)])}")

    def visualize_absolute_error_in_longitudinal_section(self, ds, z0, subset):
        
       # Ground Trust
        inletPrm = ds.inletPrm
        outletPrm = ds.outletPrm
        wallPrm = ds.wallPrm
        volumePrm = ds.volumePrm

        coordinates_all = np.vstack([inletPrm.coordinates, outletPrm.coordinates, wallPrm.coordinates, volumePrm.coordinates])
        vel_all = np.vstack([inletPrm.vel, outletPrm.vel, wallPrm.vel, volumePrm.vel])
        pressure_all = np.vstack([inletPrm.pressure, outletPrm.pressure, wallPrm.pressure, volumePrm.pressure])

        # Intersection plane - 3d geometry
        tol = 0.00030195  
        intersection_xyz, corresponding_vel, corresponding_pressure = longitudinal_plane_intersection(coordinates_all, vel_all, pressure_all, z0, tolerance=tol)
        
    
        # Grid in  z = z0 plane
        x_min = min(intersection_xyz[:, 0])
        x_max = max(intersection_xyz[:, 0])
        y_min = min(intersection_xyz[:, 1])
        y_max = max(intersection_xyz[:, 1])
        grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:200j]
        

        # REFERENCE
        # Calculate the magnitude of the velocities
        vel_ref = np.sqrt(np.sum(corresponding_vel**2, axis=1))
        pressure_ref = np.squeeze(corresponding_pressure)


        # Interpolate vel onto the grid
        vel_ref_grid = griddata(intersection_xyz[:, [0, 1]], vel_ref, (grid_x, grid_y), method='cubic')
        pressure_ref_grid = griddata(intersection_xyz[:, [0, 1]], pressure_ref, (grid_x, grid_y), method='cubic')

    
        # PREDICTED
        # Compute predicted in the grid
        x_star = grid_x.flatten()[:,None]
        y_star = grid_y.flatten()[:,None]
        z_star = z0 * np.ones_like(x_star)

        xyz_star = np.hstack([x_star, y_star, z_star])
        xyz_star_tensor = torch.tensor(xyz_star, dtype = torch.float32, requires_grad= False).to(self.config['device'])
        
        # ----
        # Data transformation: From dimension to dimensionless
        xyz_star_tensor_dimless = xyz_star_tensor/self.config['pde_param']['2R_pipe']

        with torch.no_grad():
            NN_pred_tensor_dimless = self.forward(xyz_star_tensor_dimless)

        # Update parameters based on inlet velocity
        self.config['pde_param']['V_inlet'] = self.config['test']['vel_max_inlet']
        self.config['pde_param']['Re'] = self.config['pde_param']['rho_f']*self.config['pde_param']['2R_pipe']*self.config['pde_param']['V_inlet']/self.config['pde_param']['mu_f'] 

        # Data transformation - From dimensionless to dimension
        NN_pred_tensor = torch.zeros_like(NN_pred_tensor_dimless)
        NN_pred_tensor[:,0:3] = NN_pred_tensor_dimless[:,0:3]*self.config['pde_param']['V_inlet']
        NN_pred_tensor[:,3:4] = NN_pred_tensor_dimless[:,3:4]*(self.config['pde_param']['rho_f'] * self.config['pde_param']['V_inlet']**2)
        # ----

        vel_pred_grid_aux = torch.sqrt(torch.sum(NN_pred_tensor[:,0:3]**2, axis=1))
        pressure_pred_grid_aux = NN_pred_tensor[:,3:4]

        vel_pred_grid = vel_pred_grid_aux.cpu().numpy().reshape(grid_x.shape[0],grid_y.shape[1])
        pressure_pred_grid = pressure_pred_grid_aux.cpu().numpy().reshape(grid_x.shape[0],grid_y.shape[1])

        # Pressure shift adjustment
        shift_all =  pressure_pred_grid - pressure_ref_grid
        shift_value = np.mean(shift_all[~np.isnan(shift_all)])
        pressure_pred_grid = pressure_pred_grid - shift_value

        
        # Plotting purposes
        coordinates_boundary = np.vstack([inletPrm.coordinates, outletPrm.coordinates, wallPrm.coordinates])
        vel_boundary = np.vstack([inletPrm.vel, outletPrm.vel, wallPrm.vel])
        pressure_boundary = np.vstack([inletPrm.pressure, outletPrm.pressure, wallPrm.pressure])
        
        # Intersection plane - 3d geometry
        tol = 0.0005  
        boundary_xyz, _, _ = longitudinal_plane_intersection(coordinates_boundary, vel_boundary, pressure_boundary, z0, tolerance=tol)
        

        vertices = np.vstack((boundary_xyz[:, [0, 1]], boundary_xyz[0, [0, 1]]))
        
        # Calculate the centroid of the shape
        centroid = np.mean(vertices[:-1], axis=0)  # Exclude the closing vertex
        centroid[1] = centroid[1] - 0.05 # Recolocate the centroid based on the shape of the geometry 

        # Sort the vertices by angle from the centroid
        def angle_from_centroid(vertex):
            return np.arctan2(vertex[1] - centroid[1], vertex[0] - centroid[0])

        sorted_vertices = sorted(vertices[:-1], key=angle_from_centroid) 
        sorted_vertices.append(sorted_vertices[0])  # Close the shape

        # Convert to a numpy array
        sorted_vertices = np.array(sorted_vertices)
        # Matplotlib Path
        boundary_path = mpath(sorted_vertices)

        def inside_boundary(x, y):
            return boundary_path.contains_point(np.array([x, y]))

        vel_pred_grid_masked = copy.deepcopy(vel_pred_grid)  
        pressure_pred_grid_masked = copy.deepcopy(pressure_pred_grid)
        vel_ref_grid_masked = copy.deepcopy(vel_ref_grid)  
        pressure_ref_grid_masked = copy.deepcopy(pressure_ref_grid)
        
        
        for i in range(vel_pred_grid_masked.shape[0]):
            
            for j in range(vel_pred_grid_masked.shape[1]):
                
                if not inside_boundary(grid_x[i, j], grid_y[i, j]):
                    # Set values outside the boundary to NaN
                    vel_pred_grid_masked[i, j] = np.nan  
                    pressure_pred_grid_masked[i, j] = np.nan  
                    vel_ref_grid_masked[i, j] = np.nan  
                    pressure_ref_grid_masked[i, j] = np.nan 
              
        ## Velocity
        # Calculate the minimum and maximum values for the exact and predicted vel
        vmin_vel = min(np.nanmin(vel_ref_grid_masked), np.nanmin(vel_pred_grid_masked))
        vmax_vel = max(np.nanmax(vel_ref_grid_masked), np.nanmax(vel_pred_grid_masked))

        # Plot
        fig = plt.figure(figsize=(18, 12))
        plt.subplot(1, 3, 1)
        plt.pcolor(grid_x, grid_y, vel_ref_grid_masked, cmap="jet", vmin=vmin_vel, vmax=vmax_vel)
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Vel - Exact in z={z0}")

        plt.subplot(1, 3, 2)
        plt.pcolor(grid_x, grid_y, vel_pred_grid_masked, cmap="jet", vmin=vmin_vel, vmax=vmax_vel)
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Vel - Predicted in z={z0}")

        abs_error_vel = np.empty_like(vel_ref_grid_masked)
        abs_error_vel[:,:] = np.nan 
        abs_error_vel[~np.isnan(vel_ref_grid_masked)] = abs(vel_ref_grid_masked[~np.isnan(vel_ref_grid_masked)] - vel_pred_grid_masked[~np.isnan(vel_ref_grid_masked)])
        plt.subplot(1, 3, 3)
        plt.pcolor(grid_x, grid_y, abs_error_vel, cmap="jet")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Vel - Absolute error in z={z0}")

        # Save the figure
        fig_path = self.config['charts_folder_path'].joinpath(f'vel_absolute_error_in_z_{z0}.png')
        fig.savefig(fig_path, bbox_inches="tight", dpi=100)
        
        ## Pressure
        # Calculate the minimum and maximum values for the exact and predicted pressure
        vmin_p = min(np.nanmin(pressure_ref_grid_masked), np.nanmin(pressure_pred_grid_masked))
        vmax_p = max(np.nanmax(pressure_ref_grid_masked), np.nanmax(pressure_pred_grid_masked))


        # Plot
        fig1 = plt.figure(figsize=(18, 12))
        plt.subplot(1, 3, 1)
        plt.pcolor(grid_x, grid_y, pressure_ref_grid_masked, cmap="jet", vmin=vmin_p, vmax=vmax_p)
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Pressure - Exact in z={z0}")

        plt.subplot(1, 3, 2)
        plt.pcolor(grid_x, grid_y, pressure_pred_grid_masked, cmap="jet", vmin=vmin_p, vmax=vmax_p)
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Pressure - Predicted in z={z0}")

        abs_error_pressure = np.empty_like(pressure_ref_grid_masked)
        abs_error_pressure[:,:] = np.nan 
        abs_error_pressure[~np.isnan(pressure_ref_grid_masked)] = abs(pressure_ref_grid_masked[~np.isnan(pressure_ref_grid_masked)] - pressure_pred_grid_masked[~np.isnan(pressure_ref_grid_masked)])
        plt.subplot(1, 3, 3)
        plt.pcolor(grid_x, grid_y, abs_error_pressure, cmap="jet")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Pressure - Absolute error in y={z0}")

        # Save the figure
        fig1_path = self.config['charts_folder_path'].joinpath(f'pressure_absolute_error_in_z_{z0}.png')
        fig1.savefig(fig1_path, bbox_inches="tight", dpi=100)


        self.config['logger'].info("#-----------------------------------------#")
        self.config['logger'].info(f"Accuracy in plane z={z0} - {subset}_dataset")
        
        self.config['logger'].info(f"max absolute error in magnitude of vel: {np.max(abs_error_vel[~np.isnan(vel_ref_grid_masked)])}")
        self.config['logger'].info(f"min absolute error in magnitude of vel: {np.min(abs_error_vel[~np.isnan(vel_ref_grid_masked)])}")
        self.config['logger'].info(f"mean absolute error in magnitude of vel: {np.mean(abs_error_vel[~np.isnan(vel_ref_grid_masked)])}")
        
        self.config['logger'].info(f"max absolute error in pressure: {np.max(abs_error_pressure[~np.isnan(pressure_ref_grid_masked)])}")
        self.config['logger'].info(f"min absolute error in pressure: {np.min(abs_error_pressure[~np.isnan(pressure_ref_grid_masked)])}")
        self.config['logger'].info(f"mean absolute error in pressure: {np.mean(abs_error_pressure[~np.isnan(pressure_ref_grid_masked)])}")
        
        print("#-----------------------------------------#")
        print(f"Accuracy in plane z={z0}  - {subset}_dataset")
        
        print(f"max absolute error in magnitude of vel: {np.max(abs_error_vel[~np.isnan(vel_ref_grid_masked)])}")
        print(f"min absolute error in magnitude of vel: {np.min(abs_error_vel[~np.isnan(vel_ref_grid_masked)])}")
        print(f"mean absolute error in magnitude of vel: {np.mean(abs_error_vel[~np.isnan(vel_ref_grid_masked)])}")
        
        print(f"max absolute error in pressure: {np.max(abs_error_pressure[~np.isnan(pressure_ref_grid_masked)])}")
        print(f"min absolute error in pressure: {np.min(abs_error_pressure[~np.isnan(pressure_ref_grid_masked)])}")
        print(f"mean absolute error in pressure: {np.mean(abs_error_pressure[~np.isnan(pressure_ref_grid_masked)])}")

    def visualize_absolute_error(self, flow_label, branch_input):
        
        # flow_label = self.config['dataset']['chosen_flow_label']
        planes_folder = self.config['dataset']['planes_folder_name']
        t_values = self.config['test']['t_values']
        
        planes_path = self.config['data_path'].joinpath(f'STRATA_{flow_label}/{planes_folder}/')

        # Get a list of files
        timestep_folders = os.listdir(planes_path)
        timestep_folders_floats = list(map( lambda x : float('0.' + x.split('_')[1]), timestep_folders))
        
        for time in t_values:
            
            idx = timestep_folders_floats.index(time)
            fixed_timestep_folder = timestep_folders[idx]

            planes_timestep_path = self.config['data_path'].joinpath(f'STRATA_{flow_label}/{planes_folder}/{fixed_timestep_folder}')
            # Get a list of files
            plane_files = os.listdir(planes_timestep_path)

            for filename in plane_files:
                if filename == 'planez0.txt':
                    data = np.loadtxt(planes_timestep_path.joinpath(filename), skiprows=1)
                    
                    dims3 = (data.shape[0],3)
                    dims1 = (data.shape[0],1)
                    
                    xyz = np.zeros((dims3))
                    corresponding_vel = np.zeros((dims3))
                    corresponding_pressure  = np.zeros((dims1))

                    xyz = data[:,1:4]
                    corresponding_vel = data[:,4:7]
                    corresponding_pressure  = data[:,7:8]

                    # REFERENCE
                    vel_ref = np.sqrt(np.sum(corresponding_vel**2, axis=1))
                    pressure_ref = np.squeeze(corresponding_pressure)
                
                    # PREDICTED
                    
                    # ----
                    # Data transformation: From dimension to dimensionless
                    x_star_dimless = xyz[:,0][:,None]/self.config['pde_param']['2R_pipe']
                    y_star_dimless = xyz[:,1][:,None]/self.config['pde_param']['2R_pipe']
                    z_star_dimless = xyz[:,2][:,None]/self.config['pde_param']['2R_pipe']
                    t_star_dimless = self.config['pde_param']['omega']* time * np.ones_like(x_star_dimless)

                    xyzt_star_dimless = np.hstack([x_star_dimless, y_star_dimless, z_star_dimless, t_star_dimless])
                    xyzt_star_tensor_dimless = torch.tensor(xyzt_star_dimless, dtype = torch.float32, requires_grad= False).to(self.config['device'])
                    

                    # with torch.no_grad():
                    #     NN_pred_tensor_dimless = self.model.forward(xyzt_star_tensor_dimless)

                
                    # # Data transformation: From dimensionless to dimension
                    # NN_pred_tensor = torch.zeros_like(NN_pred_tensor_dimless)
                    # NN_pred_tensor[:,0:3] = NN_pred_tensor_dimless[:,0:3]*self.config['pde_param']['V_inlet_max']
                    # NN_pred_tensor[:,3:4] = NN_pred_tensor_dimless[:,3:4]*(self.config['pde_param']['rho_f'] * self.config['pde_param']['V_inlet_max']**2)
                    # PREDICTED
                    f_tensor = []
                    for key in branch_input.keys():
                        f_array = np.tile(np.array(branch_input[key]), (xyzt_star_tensor_dimless.shape[0],1))
                        f_tensor.append(torch.tensor(f_array, dtype = torch.float32, requires_grad= False).to(self.config['device']))

                    # PREDICTED
                    
                    with torch.no_grad():
                        tau = []
                        for i, f_tensor_i in enumerate(f_tensor):
                            # N_coord = f_tensor_i.shape[1]
                            # idx = np.random.choice(N_coord,  round(0.01*N_coord), replace=False)
                            # idx = np.sort(idx)
                            # tau.append(self.model.branch_list[i](f_tensor_i[:,idx]))
                            tau.append(self.model.branch_list[i](f_tensor_i))
                        beta = self.model.trunk(xyzt_star_tensor_dimless)

                        # Predicted
                        if len(self.config['branches_control']['branch_list_ID']) == 1:
                            v1_pred = torch.sum(tau[0][:,0:100] * beta[:,0:100], axis = 1)[:, None]
                            v2_pred = torch.sum(tau[0][:,100:200] * beta[:,100:200], axis = 1)[:, None]
                            v3_pred = torch.sum(tau[0][:,200:300] * beta[:,200:300], axis = 1)[:, None]
                            p_pred = torch.sum(tau[0][:,300:400] * beta[:,300:400], axis = 1)[:, None]
                        else:
                            v1_pred = torch.sum(reduce(lambda x, y: x[:, 0:100] * y[:, 0:100], tau) * beta[:,0:100], axis = 1)[:, None]
                            v2_pred = torch.sum(reduce(lambda x, y: x[:,100:200] * y[:,100:200], tau) * beta[:,100:200], axis = 1)[:, None]
                            v3_pred = torch.sum(reduce(lambda x, y: x[:,200:300] * y[:,200:300], tau) * beta[:,200:300], axis = 1)[:, None]
                            p_pred = torch.sum(reduce(lambda x, y: x[:,300:400] * y[:,300:400], tau) * beta[:,300:400], axis = 1)[:, None]

                        v_pred = torch.cat((v1_pred,v2_pred,v3_pred), axis = 1)
            
                    # Data transformation: From dimensionless to dimension
                    v_pred = v_pred*self.config['pde_param']['V_inlet_max']
                    p_pred = p_pred*(self.config['pde_param']['rho_f'] * self.config['pde_param']['V_inlet_max']**2)
                    # ----
                    # ----
            

                    vel_pred_aux = torch.sqrt(torch.sum(v_pred[:,0:3]**2, axis=1))
                    pressure_pred_aux = p_pred[:,0]

                    vel_pred = vel_pred_aux.cpu().numpy()
                    pressure_pred = pressure_pred_aux.cpu().numpy()

                    ## Relative L2 error
                    vel_l2_error = np.linalg.norm((vel_ref-vel_pred), 2)/np.linalg.norm(vel_ref, 2)
                    p_l2_error = np.linalg.norm((pressure_ref-pressure_pred), 2)/np.linalg.norm(pressure_ref, 2)


                    self.config['logger'].info("#-----------------------------------------#")
                    self.config['logger'].info(f"Accuracy in {filename} and time={time} - full_dataset")
                            
                    self.config['logger'].info(f"Relative L2 error - Vel: {vel_l2_error}")
                    self.config['logger'].info(f"Relative L2 error - Pressure: {p_l2_error}")

                    ### Plots
                    idx1, idx2, xlabel, ylabel, plane_name, plane_value = extract_info(filename)
                    
                    plot_magnitude_and_save_absolute_error(self.config, xyz,vel_ref, vel_pred, idx1, idx2, xlabel, ylabel, plane_name,time, ID=f'Deb{flow_label}_Vel')
                    plot_magnitude_and_save_absolute_error(self.config, xyz,pressure_ref, pressure_pred, idx1, idx2, xlabel, ylabel, plane_name,time, ID=f'Deb{flow_label}_Pressure')
                    
                    del xyzt_star_tensor_dimless
                    del f_tensor
                    del tau
                    del beta
                    del v1_pred
                    del v2_pred
                    del v3_pred
                    del p_pred
                    del v_pred
                    del vel_pred_aux
                    del pressure_pred_aux
                    del vel_l2_error
                    del p_l2_error