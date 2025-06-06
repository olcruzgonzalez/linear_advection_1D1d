import os
import time
import logging
import copy

from functools import reduce
import numpy as np
from typing import Sequence

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
from generate_dataset.ground_trust import u_close_form, calculate_phi
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
        
        # Architecture
        num_layers = 1
        hidden_dim = config[network_component]['neuralNet']['hidden_dim']
        out_dim = config[network_component]['neuralNet']['out_dim']

        hidden_layers = num_layers*[hidden_dim]

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
        
        x = self.net(x)

        return x

class StandardDense_DeepONet(nn.Module):
    
    def __init__(self, config, in_dim, network_component):
        
        super().__init__()

        # Activation function
        self.activation =  activation_func(config[network_component]['neuralNet']['activation'])

        # Architecture
        self.num_layers = config[network_component]['neuralNet']['num_layers']
        hidden_dim = config[network_component]['neuralNet']['hidden_dim']
        out_dim = config[network_component]['neuralNet']['out_dim']

        hidden_layers = self.num_layers*[hidden_dim]

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
        
        x = self.net(x)
        
        return x
    
class  MultiLayerPerceptron_DeepONet (nn.Module):

    def __init__(self, config, network_component):

        super().__init__()

        device = config['device']
        in_dim = config[network_component]['neuralNet']['in_dim']
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
           
            self.param_groups.append({'params': [self.standard_dense.net[2*i].weight]})
            self.param_groups.append({'params': [self.standard_dense.net[2*i].bias]})
       
            self.param_groups.append({'params': [self.encoder_dense1.net[0].weight]})
            self.param_groups.append({'params': [self.encoder_dense1.net[0].bias]})
     
            self.param_groups.append({'params': [self.encoder_dense2.net[0].weight]})
            self.param_groups.append({'params': [self.encoder_dense2.net[0].bias]})
        
        
    def forward(self, x):

        # Fourier features embedding 
        if self.has_fourier_emb:
            x = self.call_fourier_embedding(x)

        # Forward
        U = self.encoder_dense1(x)
        V = self.encoder_dense2(x)
        
        H1 = self.standard_dense.net[1](self.standard_dense.net[0](x))
        
        for i in range(1, self.num_layers):
            H = self.standard_dense.net[2*i+1](self.standard_dense.net[2*i](H1))
            H1 = H * U + (1 - H) * V

        H1 = self.standard_dense.net[-1](H1)

        return H1

class ModifiedDeepONetArch (nn.Module):

    def __init__(self, config):

        super().__init__()

        device = config['device']
        in_dim_branch = config['branch1']['neuralNet']['in_dim']
        in_dim_trunk = config['trunk1']['neuralNet']['in_dim']
        
        self.branch_hidden_dim = config['branch1']['neuralNet']['hidden_dim']
        
        self.num_layers_branch = config['branch1']['neuralNet']['num_layers']
        self.num_layers_trunk = config['trunk1']['neuralNet']['num_layers']
        
        # ----------
        # #----------------------------------
        # # Fourier embedding 
        # #----------------------------------
        # self.fourier_emb = config['fourier_emb']
        
        # if self.fourier_emb:
        #     in_dim_branch1 = 2 * self.fourier_emb['embed_dim']
        #     self.call_fourier_embedding_branch1 = FourierEmbedding(self.fourier_emb, config['branch1']['neuralNet']['in_dim'], device)
            
        #     in_dim_branch2 = 2 * self.fourier_emb['embed_dim']
        #     self.call_fourier_embedding_branch2 = FourierEmbedding(self.fourier_emb, config['branch2']['neuralNet']['in_dim'], device)
            
        #     in_dim_trunk = 2 * self.fourier_emb['embed_dim']
        #     self.call_fourier_embedding_trunk = FourierEmbedding(self.fourier_emb, config['trunk']['neuralNet']['in_dim'], device)


        #----------------------------------
        # Architectures initialization
        #----------------------------------
        self.encoder_dense_branch = EncoderDense_DeepONet(config, in_dim_branch, 'branch1') 
        self.encoder_dense_trunk = EncoderDense_DeepONet(config, in_dim_trunk, 'trunk1') 
        
        self.standard_dense_branch = StandardDense_DeepONet(config, in_dim_branch, 'branch1')
        self.standard_dense_trunk = StandardDense_DeepONet(config, in_dim_trunk, 'trunk1')

        #----------------------------------
        # Manually define parameters
        #----------------------------------
    
        self.param_groups = []
        for i in range(self.num_layers_branch+1):
           
            self.param_groups.append({'params': [self.standard_dense_branch.net[2*i].weight]})
            self.param_groups.append({'params': [self.standard_dense_branch.net[2*i].bias]})
            
        for i in range(self.num_layers_trunk+1):
           
            self.param_groups.append({'params': [self.standard_dense_trunk.net[2*i].weight]})
            self.param_groups.append({'params': [self.standard_dense_trunk.net[2*i].bias]})
           
    
        
        self.param_groups.append({'params': [self.encoder_dense_branch.net[0].weight]})
        self.param_groups.append({'params': [self.encoder_dense_branch.net[0].bias]})
    
        self.param_groups.append({'params': [self.encoder_dense_trunk.net[0].weight]})
        self.param_groups.append({'params': [self.encoder_dense_trunk.net[0].bias]})
        
    def forward(self, u,y):

        # # Create a copy of y
        # copy_y = y.clone()

        # Forward
        U = self.encoder_dense_branch(u)
        V = self.encoder_dense_trunk(y)

        ## np.tile
        U = U.reshape(-1, self.branch_hidden_dim)
        
        H1u = self.standard_dense_branch.net[1](self.standard_dense_branch.net[0](u))
        H1y = self.standard_dense_trunk.net[1](self.standard_dense_trunk.net[0](y))

        ## np.tile
        H1u = H1u.reshape(-1, self.branch_hidden_dim)

        H1u = (1 - H1u) * U + H1u * V
        H1y = (1 - H1y) * U + H1y * V

        for i in range(1, self.num_layers_branch): # The three architectures have to be the same
            Hu = self.standard_dense_branch.net[2*i+1](self.standard_dense_branch.net[2*i](H1u))
            Hy = self.standard_dense_trunk.net[2*i+1](self.standard_dense_trunk.net[2*i](H1y))
            
            H1u = (1 - Hu) * U + Hu * V
            H1y = (1 - Hy) * U + Hy * V

        tau = self.standard_dense_branch.net[-1](H1u)
        beta = self.standard_dense_trunk.net[-1](H1y)
        
        u_pred = torch.sum(tau * beta, axis = 1)[:, None]

        # if self.has_hard_bc:
        #     return copy_y*u_pred
            
        return u_pred


#--------------------------
class NeuralNetwork(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.config = config
        
        # NN Architectures
        # Branch
        self.branch_list = []
        for branch_ID in config['branches_control']['branch_list_ID']:
       
            if config[branch_ID]['neuralNet']['architecture'] == "Modified MLP":
            
                branch_i = ModifiedMultiLayerPerceptron_DeepONet(config, network_component = branch_ID)
                self.branch_list.append(branch_i.to(self.config['device']))
            
            else: #"MLP":
                
                branch_i = MultiLayerPerceptron_DeepONet(config, network_component = branch_ID)
                self.branch_list.append(branch_i.to(self.config['device']))

            self.config['logger'].info(f'branch_component: {branch_i}')

        # Trunk
        self.trunk_list = []
        for trunk_ID in config['trunk_control']['trunk_list_ID']:
            
            if config[trunk_ID]['neuralNet']['architecture'] == "Modified MLP":
            
                trunk_i = ModifiedMultiLayerPerceptron_DeepONet(config, network_component = trunk_ID)
                self.trunk_list.append(trunk_i.to(self.config['device']))
            
            else: #"MLP":
                
                trunk_i = MultiLayerPerceptron_DeepONet(config, network_component = trunk_ID)
                self.trunk_list.append(trunk_i.to(self.config['device']))
        

        self.config['logger'].info(f'Definition of the NN Architectures')

    
class Trainer_PIDeepONetLdata(nn.Module):
    
    def __init__(self, config, model):
        
        super().__init__()
        
        self.config = config
        self.has_exponential_decay = self.config['optim1']['exponential_decay']['enabled']
        self.has_loss_balancing = self.config['loss_balancing']['enabled']

        # Initialize the SummaryWriter
        if self.config['mode']=='train':
            self.writer = SummaryWriter(log_dir=config['logs_folder_path'])

        self.model = model
        self.branch_in_dim = config['branch1']['neuralNet']['in_dim']
        self.branch_out_dim = config['branch1']['neuralNet']['out_dim']
        self.trunk_in_dim = config['trunk1']['neuralNet']['in_dim']
        self.output_dim = 1

        # PDE parameters
        self.c = config['dataset']['c'] 

         # Optimizers
        if config['optim1']['optimizer'] == 'Adam':

            # Retrieve parameter groups from each model
            param_groups_branch_sum = []
            for branch_i in self.model.branch_list:
                param_groups_branch_i = list(branch_i.param_groups)
                param_groups_branch_sum += param_groups_branch_i

            param_groups_trunk_sum = []
            for trunk_i in self.model.trunk_list:
                param_groups_trunk_i = list(trunk_i.param_groups)
                param_groups_trunk_sum += param_groups_trunk_i
            
            # Combine the parameter groups into a single list
            combined_param_groups = param_groups_branch_sum + param_groups_trunk_sum

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
            all_params = [param for group in self.model.branch_list[0].param_groups + self.model.trunk_list[0].param_groups for param in group['params']]
           
            self.optimizer_LBFGS = torch.optim.LBFGS(all_params, max_iter=config['optim2']['max_iter'], max_eval=config['optim2']['max_eval'], tolerance_grad=1.0 * torch.finfo(torch.float32).eps, history_size=50)

        
        # Loss function
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        

        # Logging
        self.total_loss = None
        self.l2_error_u = None

        # Loss terms
        self.term_loss_tensor_dict = {key:[] for key in self.config['loss_terms']}

        # Predicted output terms
        self.term_s_tensor_dict = {key:[] for key in self.config['loss_terms']}

        # lambdas initialization
        self.term_lambdas_tensor = torch.tensor(
                                    [self.config['lambda_weights_init'][i] for i in range(len(self.config['loss_terms']))], 
                                    dtype=torch.float32, requires_grad=False).to(self.config['device'])

        self.term_lambdas_tensor_dict = {key: self.term_lambdas_tensor[i] for i,key in enumerate(self.config['loss_terms'])}

          # Loss balancing
        scheme = self.config['loss_balancing']['scheme']
        if self.has_loss_balancing and scheme == 'ntk_guided_weights':
            
            self.K = {key:[] for key in self.config['loss_terms']}

            self.config['logger'].info(f"Using {scheme} for loss balancing.")
            self.loss_balancing_log = f"Loss Balancing - {scheme}\n"


        self.log_log = """LOG\n"""
        self.min_error_for_checkpoint = np.Infinity
        self.runtime0 = self.config['tick_start']
        self.best_iter = 0

    def loss_boundary_condition(self):

        tau = []
        for i, f_bc_tensor_i in enumerate(self.f_bc_tensor):
            tau.append(self.model.branch_list[i](f_bc_tensor_i))

        beta = []
        for i, xt_bc_tensor_i in enumerate(self.xt_bc_tensor):
            beta.append(self.model.trunk_list[i](xt_bc_tensor_i))

        ## np.tile
        tau[0] = tau[0].reshape(-1, self.branch_out_dim)

        # Predicted
        u_pred = torch.sum(tau[0] * beta[0], axis = 1)[:, None]
       

        # Ground Trust
        u_GT = self.u_bc_tensor
        
        # Compute losses
        u_loss = self.loss_fn(u_pred, u_GT)

    
        return u_loss, u_pred
    
    def loss_data(self):

        tau = []
        for i, f_data_tensor_i in enumerate(self.f_data_tensor):
            tau.append(self.model.branch_list[i](f_data_tensor_i))
 
        beta = []
        for i, xt_data_tensor_i in enumerate(self.xt_data_tensor):
            beta.append(self.model.trunk_list[i](xt_data_tensor_i))

        ## np.tile
        tau[0] = tau[0].reshape(-1, self.branch_out_dim)

        # Predicted
        u_pred = torch.sum(tau[0] * beta[0], axis = 1)[:, None]
       

        # Ground Trust
        u_GT = self.u_data_tensor
        
        # Compute losses
        u_loss = self.loss_fn(u_pred, u_GT)

    
        return u_loss, u_pred
    
    def loss_initial_condition(self):

        tau = []
        for i, f_ic_tensor_i in enumerate(self.f_ic_tensor):
            tau.append(self.model.branch_list[i](f_ic_tensor_i))
 
        beta = []
        for i, xt_ic_tensor_i in enumerate(self.xt_ic_tensor):
            beta.append(self.model.trunk_list[i](xt_ic_tensor_i))

        ## np.tile
        tau[0] = tau[0].reshape(-1, self.branch_out_dim)

        # Predicted
        u_pred = torch.sum(tau[0] * beta[0], axis = 1)[:, None]
       

        # Ground Trust
        u_GT = self.u_ic_tensor
        
        # Compute losses
        u_loss = self.loss_fn(u_pred, u_GT)

    
        return u_loss, u_pred
    
    def loss_physics(self):

        # forward pass
        tau = []
        for i, f_phy_tensor_i in enumerate(self.f_phy_tensor):
            tau.append(self.model.branch_list[i](f_phy_tensor_i))

        beta = []
        beta.append(self.model.trunk_list[0](torch.cat([self.x_phy, self.t_phy], 1)))

        ## np.tile
        tau[0] = tau[0].reshape(-1, self.branch_out_dim)

        # Predicted
        if len(self.config['branches_control']['branch_list_ID']) == 1:
            u = torch.sum(tau[0] * beta[0], axis = 1)[:, None]

        # Autodiff
        u_x = torch.autograd.grad(
            u, self.x_phy, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        u_t = torch.autograd.grad(
            u, self.t_phy, grad_outputs=torch.ones_like(u), create_graph=True)[0]


        EQ1 = u_t + self.c *  u_x 
       

        # Compute losses
        loss = self.loss_fn(EQ1, self.residual_target)
       
        return loss, u
        
    def loss(self):

        # Forward pass and compute the losses per terms
        loss_ic, pred_ic = self.loss_initial_condition()
        loss_bc, pred_bc = self.loss_boundary_condition()
        loss_data, pred_data = self.loss_data()
        loss_p, pred_phy = self.loss_physics()

        self.term_loss_tensor_dict['ic'] = loss_ic
        self.term_loss_tensor_dict['bc'] = loss_bc
        self.term_loss_tensor_dict['data'] = loss_data
        self.term_loss_tensor_dict['phy'] = loss_p

        self.term_s_tensor_dict['ic'] = pred_ic
        self.term_s_tensor_dict['bc'] = pred_bc
        self.term_s_tensor_dict['data'] = pred_data
        self.term_s_tensor_dict['phy'] = pred_phy
        
        
        # Apply adaptive lambdas
        self.loss_balancing_call()
        
    
        # Compute total loss
        loss_total = self.term_lambdas_tensor_dict['ic'] * loss_ic + self.term_lambdas_tensor_dict['bc'] * loss_bc + self.term_lambdas_tensor_dict['data'] * loss_data + self.term_lambdas_tensor_dict['phy'] * loss_p 


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

        if self.config['loss_balancing']['scheme'] == 'no_weights':
            pass

        elif self.config['loss_balancing']['scheme'] == 'fixed_weights':
            pass
        
        elif self.config['loss_balancing']['scheme'] == 'data_guided_weights':
            
            # Create a copy of the tensor values to avoid gradient issues
            term_s_tensor_dict_copy = {
                key: tensor.detach().clone() 
                for key, tensor in self.term_s_tensor_dict.items()
            }

            for idx, key in enumerate(self.config['loss_terms']):
                # Use detached tensor to compute max
                max_val = torch.max(torch.abs(term_s_tensor_dict_copy[key]))
                self.term_lambdas_tensor[idx] = 1/max_val.detach()

        elif self.config['loss_balancing']['scheme'] == 'ntk_guided_weights':
           
            # 0. grab *exactly* the parameters used by the optimiser -------------
            all_params = [param for group in self.model.branch_list[0].param_groups + self.model.trunk_list[0].param_groups for param in group['params']]
            
            # 1. gather losses --------------------------------------------------
            losses = [self.term_loss_tensor_dict[k] for k in self.config["loss_terms"]]

             # 2a. build gradient matrix  (n_terms × n_params) ------------------
            grad_mat = torch.stack([
                self.flat_grad(loss, all_params).detach()
                for loss in losses
            ]) #                ↑ detach: NTK weights should not be back-proped

            # 2b. NTK diagonal --------------------------------------------------
            k_diag = self.ntk_diag_from_grads(grad_mat)     # shape (n_terms,)

            # 3. λ update -------------------------------------------------------
            lambdas = self.normalise_ntk(
                k_diag,                                # stays on-device
                self.config["loss_balancing"]["type"]
            ).to(self.config["device"])

            # 4. # update lambdas -----------------------------------------
            self.term_lambdas_tensor = lambdas
        
        else:
            pass
    
    def normalise_ntk(self, k_vec: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Vectorised λ update:   λ_k = (‖H‖_∞ / H_kk)^α   with α ∈ {1,½}.
        """
        if mode == "global_NTK_weights":      # α = 1, global sum variant
            return k_vec.sum() / k_vec
        if mode == "local_NTK_weights":       # α = 1, max variant
            return k_vec.max() / k_vec
        if mode == "moderate_local_NTK_weights":  # α = ½
            return torch.sqrt(k_vec.max() / k_vec)

        raise ValueError(f"Unknown NTK scheme: {mode}")


    def flat_grad(self, loss: torch.Tensor,
                params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
        """
        Return a 1-D tensor containing ∂loss/∂θ for *all* parameters,
        inserting zeros where autograd returns None.
        """
        grads = torch.autograd.grad(
            loss, params,
            retain_graph=True, create_graph=False, allow_unused=True
        )
        vec = [
            (g if g is not None else torch.zeros_like(p)).flatten()
            for g, p in zip(grads, params)
        ]
        return torch.cat(vec)                      # shape: (n_params,)


    def ntk_diag_from_grads(self, grad_mat: torch.Tensor) -> torch.Tensor:
        """
        Given the gradient matrix of shape (n_terms, n_params),
        return the vector of NTK diagonal entries.
        """
        return (grad_mat ** 2).sum(dim=1)          # row-wise ‖·‖₂²
   
    def logger_call(self):

        # Monitoring resources 
        if self.config['logging']['monitoring_resources'] and self.regular_iter == 0:

            self.log_log += "\n---------\n"

            # CPU / system-RAM
            self.log_log += get_cpu_memory_usage("") + "\n"

            # GPU (if any)
            if torch.cuda.is_available():
                cuda_device = torch.device("cuda")  # default current device
                self.log_log += get_cuda_memory_usage(cuda_device,"") + "\n"
            else:
                self.log_log += "No GPU detected\n"
        
        # Total and term losses
        loss_train = self.total_loss
        l2_error_u = self.l2_error_u
        
        
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
     
                
        # Save into a .txt file
        with open(self.config['train_progress_file_path'], 'w') as file:
            file.write(self.log_log)
        
        # Tensorboard
        self.writer.add_scalar('total_loss', self.total_loss, self.regular_iter)
        self.writer.add_scalar('l2_error_u', self.l2_error_u, self.regular_iter)
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
            
            for i in range(len(self.model.trunk_list)):
                end_string = '_state_dict'
                key = self.config['trunk_control']['trunk_list_ID'][i] + end_string
                aux_dict = {key: self.model.trunk_list[i].state_dict()}
                model_weights.update(aux_dict)
            
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

    def random_sampling(self, xt, label = 'val'):
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
        
        xt_bc = {} 
        u_bc = {} 
        
        xt_data = {} 
        u_data = {} 
        
        xt_val = {} 
        u_val = {}

        for chosen_flow_label in self.config['train']['training_param_label']:


            #---------------------
            # Fixed dataset
            #---------------------
            ## BOUNDARY CONDITIONS AND INITIAL CONDITION
            xt_bc[chosen_flow_label], u_bc[chosen_flow_label], xt_ic[chosen_flow_label], u_ic[chosen_flow_label]= craft_bc_and_ic_dataset(full_ds[chosen_flow_label], time_vector)
            
            ## DATA 
            xt_data[chosen_flow_label], u_data[chosen_flow_label] = build_stratum_dataset(dataPrm[chosen_flow_label], time_vector, label = 'DATA')

            ## VALIDATION DATASET
            xt_val[chosen_flow_label], u_val[chosen_flow_label] = craft_validation_dataset(val_ds[chosen_flow_label], time_vector)


        # """The coordinate points are the same between the different datasets, so we can choose a geometry among the options"""
        coordinates, time_min, time_max = get_coordinates_for_generator(full_ds[self.config['train']['training_param_label'][0]], time_vector)


        # Workflow
        self.custom_bar = trange(self.last_iter + 1)

        # Branch
        branch_input = {}
        for key in self.config['branches_control']['branch_input_ID']:
            branch_input.update({key:[]})

        ## Random Sampling - Get Fixed Indexes
        # BC 
        idx_bc_fixed = self.random_sampling(xt_bc[chosen_flow_label], label = 'bc')
        
        for chosen_flow_label in self.config['train']['training_param_label']:
                
            # INLET   
            u_bc_sample = u_bc[chosen_flow_label][idx_bc_fixed]    
            
            ## Branch
            # for i in range(N_vel_branch_inlet):
                # index = self.config['branches_control']['axis_indexes'][self.config['branches_control']['vel_axis_ID'][i]]
                # branch_input[self.config['branches_control']['branch_input_ID'][i]].append(vel_bc_inlet_sample[:,index].T)
            branch_input[self.config['branches_control']['branch_input_ID'][0]].append(u_bc_sample.T)
            
        
        for regular_iter in self.custom_bar:

            self.regular_iter = regular_iter

            # Trunk
            xt_ic_sample_all = []
            u_ic_target = []
            
            xt_bc_sample_all = []
            u_bc_target = []
           
            xt_data_sample_all = []
            u_data_target = []

           
            # Val
            xt_val_sample_all = []
            u_val_target = []
           

            ## Random Sampling - Get Indexes
            # IC 
            idx_ic = self.random_sampling(xt_ic[chosen_flow_label], label = 'ic')
            
            # BC 
            idx_bc = self.random_sampling(xt_bc[chosen_flow_label],  label = 'bc')
            
            # DATA 
            idx_data = self.random_sampling(xt_data[chosen_flow_label],  label = 'data')

            # VAL 
            idx_val = self.random_sampling(xt_val[chosen_flow_label],  label = 'val')

            for chosen_flow_label in self.config['train']['training_param_label']:
                
                # IC 
                xt_ic_sample =  xt_ic[chosen_flow_label][idx_ic]
                u_ic_sample = u_ic[chosen_flow_label][idx_ic]
                
                # BC 
                xt_bc_sample =  xt_bc[chosen_flow_label][idx_bc]
                u_bc_sample = u_bc[chosen_flow_label][idx_bc]
                
                # DATA 
                xt_data_sample =  xt_data[chosen_flow_label][idx_data]
                u_data_sample = u_data[chosen_flow_label][idx_data]

                # VAL 
                xt_val_sample = xt_val[chosen_flow_label][idx_val]
                u_val_sample = u_val[chosen_flow_label][idx_val]

                ## Trunk
                xt_ic_sample_all.append(xt_ic_sample)
                u_ic_target.append(u_ic_sample)
                
                xt_bc_sample_all.append(xt_bc_sample)
                u_bc_target.append(u_bc_sample)
                
                xt_data_sample_all.append(xt_data_sample)
                u_data_target.append(u_data_sample)

                xt_val_sample_all.append(xt_val_sample)
                u_val_target.append(u_val_sample)

            # Trunk
            xt_ic_sample_all = np.array(xt_ic_sample_all)
            u_ic_target = np.array(u_ic_target)
            
            xt_bc_sample_all = np.array(xt_bc_sample_all)
            u_bc_target = np.array(u_bc_target)
            
            xt_data_sample_all = np.array(xt_data_sample_all)
            u_data_target = np.array(u_data_target)

            xt_val_sample_all = np.array(xt_val_sample_all)
            u_val_target = np.array(u_val_target)
            
            # IC
            # N = 6, P = ?
            self.xt_ic_tensor = []
            xt_ic_tensor = torch.tensor(np.swapaxes(xt_ic_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            self.xt_ic_tensor.append(xt_ic_tensor)
          
            self.f_ic_tensor = []
            num_samples = xt_ic_sample.shape[0]  # 51850
            target_device = self.config['device']

            for key in branch_input.keys():
                
                base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]),dtype=torch.float32,device=target_device)
                
                final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
                
                self.f_ic_tensor.append(final_view)
            
            self.u_ic_tensor = torch.tensor(np.swapaxes(u_ic_target,0,1).reshape(-1,self.output_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            # BC
            self.xt_bc_tensor = []
            xt_bc_tensor = torch.tensor(np.swapaxes(xt_bc_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            self.xt_bc_tensor.append(xt_bc_tensor)
            
            self.f_bc_tensor = []
            num_samples = xt_bc_sample.shape[0]  
            target_device = self.config['device']
            
            for key in branch_input.keys():
                
                base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]),dtype=torch.float32,device=target_device)
                
                final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
                
                self.f_bc_tensor.append(final_view)
            
            self.u_bc_tensor = torch.tensor(np.swapaxes(u_bc_target,0,1).reshape(-1,self.output_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            # DATA
            self.xt_data_tensor = []
            xt_data_tensor = torch.tensor(np.swapaxes(xt_data_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            self.xt_data_tensor.append(xt_data_tensor)
            
            self.f_data_tensor = []
            num_samples = xt_data_sample.shape[0]  
            target_device = self.config['device']

            for key in branch_input.keys():
                
                base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]),dtype=torch.float32,device=target_device)
                
                final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
                
                self.f_data_tensor.append(final_view)
            
            self.u_data_tensor = torch.tensor(np.swapaxes(u_data_target,0,1).reshape(-1,self.output_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            
            
           
            # Validation dataset
            # N = 6, P = xyzt_val_fixed.shape[0]
            self.xt_val_tensor = []
            xt_val_tensor = torch.tensor(np.swapaxes(xt_val_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            self.xt_val_tensor.append(xt_val_tensor)
            
            
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
                
                beta = []
                for i, xt_val_tensor_i in enumerate(self.xt_val_tensor):
                    beta.append(self.model.trunk_list[i](xt_val_tensor_i))

                tau[0] = tau[0].reshape(-1, self.branch_out_dim)

                u_pred = torch.sum(tau[0] * beta[0], axis = 1)[:, None]
               
            
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
        self.has_exponential_decay = self.config['optim1']['exponential_decay']['enabled']
        self.has_loss_balancing = self.config['loss_balancing']['enabled']

        # Initialize the SummaryWriter
        if self.config['mode']=='train':
            self.writer = SummaryWriter(log_dir=config['logs_folder_path'])

        self.model = model
        self.branch_in_dim = config['branch1']['neuralNet']['in_dim']
        self.branch_out_dim = config['branch1']['neuralNet']['out_dim']
        self.trunk_in_dim = config['trunk1']['neuralNet']['in_dim']
        self.output_dim = 1

        # PDE parameters
        self.c = config['dataset']['c'] 

         # Optimizers
        if config['optim1']['optimizer'] == 'Adam':

            # Retrieve parameter groups from each model
            param_groups_branch_sum = []
            for branch_i in self.model.branch_list:
                param_groups_branch_i = list(branch_i.param_groups)
                param_groups_branch_sum += param_groups_branch_i

            param_groups_trunk_sum = []
            for trunk_i in self.model.trunk_list:
                param_groups_trunk_i = list(trunk_i.param_groups)
                param_groups_trunk_sum += param_groups_trunk_i
            
            # Combine the parameter groups into a single list
            combined_param_groups = param_groups_branch_sum + param_groups_trunk_sum

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
            all_params = [param for group in self.model.branch_list[0].param_groups + self.model.trunk_list[0].param_groups for param in group['params']]
           
            self.optimizer_LBFGS = torch.optim.LBFGS(all_params, max_iter=config['optim2']['max_iter'], max_eval=config['optim2']['max_eval'], tolerance_grad=1.0 * torch.finfo(torch.float32).eps, history_size=50)

        
        # Loss function
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        

        # Logging
        self.total_loss = None
        self.l2_error_u = None

        # Loss terms
        self.term_loss_tensor_dict = {key:[] for key in self.config['loss_terms']}

        # Predicted output terms
        self.term_s_tensor_dict = {key:[] for key in self.config['loss_terms']}

        # lambdas initialization
        self.term_lambdas_tensor = torch.tensor(
                                    [self.config['lambda_weights_init'][i] for i in range(len(self.config['loss_terms']))], 
                                    dtype=torch.float32, requires_grad=False).to(self.config['device'])

        self.term_lambdas_tensor_dict = {key: self.term_lambdas_tensor[i] for i,key in enumerate(self.config['loss_terms'])}

          # Loss balancing
        scheme = self.config['loss_balancing']['scheme']
        if self.has_loss_balancing and scheme == 'ntk_guided_weights':
            
            self.K = {key:[] for key in self.config['loss_terms']}

            self.config['logger'].info(f"Using {scheme} for loss balancing.")
            self.loss_balancing_log = f"Loss Balancing - {scheme}\n"


        self.log_log = """LOG\n"""
        self.min_error_for_checkpoint = np.Infinity
        self.runtime0 = self.config['tick_start']
        self.best_iter = 0

    def loss_boundary_condition(self):

        tau = []
        for i, f_bc_tensor_i in enumerate(self.f_bc_tensor):
            tau.append(self.model.branch_list[i](f_bc_tensor_i))

        beta = []
        for i, xt_bc_tensor_i in enumerate(self.xt_bc_tensor):
            beta.append(self.model.trunk_list[i](xt_bc_tensor_i))

        ## np.tile
        tau[0] = tau[0].reshape(-1, self.branch_out_dim)

        # Predicted
        u_pred = torch.sum(tau[0] * beta[0], axis = 1)[:, None]
       

        # Ground Trust
        u_GT = self.u_bc_tensor
        
        # Compute losses
        u_loss = self.loss_fn(u_pred, u_GT)

    
        return u_loss, u_pred
    
    def loss_initial_condition(self):

        tau = []
        for i, f_ic_tensor_i in enumerate(self.f_ic_tensor):
            tau.append(self.model.branch_list[i](f_ic_tensor_i))
 
        beta = []
        for i, xt_ic_tensor_i in enumerate(self.xt_ic_tensor):
            beta.append(self.model.trunk_list[i](xt_ic_tensor_i))

        ## np.tile
        tau[0] = tau[0].reshape(-1, self.branch_out_dim)

        # Predicted
        u_pred = torch.sum(tau[0] * beta[0], axis = 1)[:, None]
       

        # Ground Trust
        u_GT = self.u_ic_tensor
        
        # Compute losses
        u_loss = self.loss_fn(u_pred, u_GT)

    
        return u_loss, u_pred
    
    def loss_physics(self):

        # forward pass
        tau = []
        for i, f_phy_tensor_i in enumerate(self.f_phy_tensor):
            tau.append(self.model.branch_list[i](f_phy_tensor_i))

        beta = []
        beta.append(self.model.trunk_list[0](torch.cat([self.x_phy, self.t_phy], 1)))

        ## np.tile
        tau[0] = tau[0].reshape(-1, self.branch_out_dim)

        # Predicted
        if len(self.config['branches_control']['branch_list_ID']) == 1:
            u = torch.sum(tau[0] * beta[0], axis = 1)[:, None]

        # Autodiff
        u_x = torch.autograd.grad(
            u, self.x_phy, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        u_t = torch.autograd.grad(
            u, self.t_phy, grad_outputs=torch.ones_like(u), create_graph=True)[0]


        EQ1 = u_t + self.c *  u_x 
       

        # Compute losses
        loss = self.loss_fn(EQ1, self.residual_target)
       
        return loss, u
        
    def loss(self):

        # Forward pass and compute the losses per terms
        loss_ic, pred_ic = self.loss_initial_condition()
        loss_bc, pred_bc = self.loss_boundary_condition()
        loss_p, pred_phy = self.loss_physics()

        self.term_loss_tensor_dict['ic'] = loss_ic
        self.term_loss_tensor_dict['bc'] = loss_bc
        self.term_loss_tensor_dict['phy'] = loss_p

        self.term_s_tensor_dict['ic'] = pred_ic
        self.term_s_tensor_dict['bc'] = pred_bc
        self.term_s_tensor_dict['phy'] = pred_phy
        
        
        # Apply adaptive lambdas
        self.loss_balancing_call()
        
    
        # Compute total loss
        loss_total = self.term_lambdas_tensor_dict['ic'] * loss_ic + self.term_lambdas_tensor_dict['bc'] * loss_bc + self.term_lambdas_tensor_dict['phy'] * loss_p 


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

        if self.config['loss_balancing']['scheme'] == 'no_weights':
            pass

        elif self.config['loss_balancing']['scheme'] == 'fixed_weights':
            pass
        
        elif self.config['loss_balancing']['scheme'] == 'data_guided_weights':
            
            # Create a copy of the tensor values to avoid gradient issues
            term_s_tensor_dict_copy = {
                key: tensor.detach().clone() 
                for key, tensor in self.term_s_tensor_dict.items()
            }

            for idx, key in enumerate(self.config['loss_terms']):
                # Use detached tensor to compute max
                max_val = torch.max(torch.abs(term_s_tensor_dict_copy[key]))
                self.term_lambdas_tensor[idx] = 1/max_val.detach()

        elif self.config['loss_balancing']['scheme'] == 'ntk_guided_weights':
           
            # 0. grab *exactly* the parameters used by the optimiser -------------
            all_params = [param for group in self.model.branch_list[0].param_groups + self.model.trunk_list[0].param_groups for param in group['params']]
            
            # 1. gather losses --------------------------------------------------
            losses = [self.term_loss_tensor_dict[k] for k in self.config["loss_terms"]]

             # 2a. build gradient matrix  (n_terms × n_params) ------------------
            grad_mat = torch.stack([
                self.flat_grad(loss, all_params).detach()
                for loss in losses
            ]) #                ↑ detach: NTK weights should not be back-proped

            # 2b. NTK diagonal --------------------------------------------------
            k_diag = self.ntk_diag_from_grads(grad_mat)     # shape (n_terms,)

            # 3. λ update -------------------------------------------------------
            lambdas = self.normalise_ntk(
                k_diag,                                # stays on-device
                self.config["loss_balancing"]["type"]
            ).to(self.config["device"])

            # 4. # update lambdas -----------------------------------------
            self.term_lambdas_tensor = lambdas
        
        else:
            pass
    
    def normalise_ntk(self, k_vec: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Vectorised λ update:   λ_k = (‖H‖_∞ / H_kk)^α   with α ∈ {1,½}.
        """
        if mode == "global_NTK_weights":      # α = 1, global sum variant
            return k_vec.sum() / k_vec
        if mode == "local_NTK_weights":       # α = 1, max variant
            return k_vec.max() / k_vec
        if mode == "moderate_local_NTK_weights":  # α = ½
            return torch.sqrt(k_vec.max() / k_vec)

        raise ValueError(f"Unknown NTK scheme: {mode}")


    def flat_grad(self, loss: torch.Tensor,
                params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
        """
        Return a 1-D tensor containing ∂loss/∂θ for *all* parameters,
        inserting zeros where autograd returns None.
        """
        grads = torch.autograd.grad(
            loss, params,
            retain_graph=True, create_graph=False, allow_unused=True
        )
        vec = [
            (g if g is not None else torch.zeros_like(p)).flatten()
            for g, p in zip(grads, params)
        ]
        return torch.cat(vec)                      # shape: (n_params,)


    def ntk_diag_from_grads(self, grad_mat: torch.Tensor) -> torch.Tensor:
        """
        Given the gradient matrix of shape (n_terms, n_params),
        return the vector of NTK diagonal entries.
        """
        return (grad_mat ** 2).sum(dim=1)          # row-wise ‖·‖₂²
   
    def logger_call(self):

        # Monitoring resources 
        if self.config['logging']['monitoring_resources'] and self.regular_iter == 0:

            self.log_log += "\n---------\n"

            # CPU / system-RAM
            self.log_log += get_cpu_memory_usage("") + "\n"

            # GPU (if any)
            if torch.cuda.is_available():
                cuda_device = torch.device("cuda")  # default current device
                self.log_log += get_cuda_memory_usage(cuda_device,"") + "\n"
            else:
                self.log_log += "No GPU detected\n"
        
        # Total and term losses
        loss_train = self.total_loss
        l2_error_u = self.l2_error_u
        
        
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
     
                
        # Save into a .txt file
        with open(self.config['train_progress_file_path'], 'w') as file:
            file.write(self.log_log)
        
        # Tensorboard
        self.writer.add_scalar('total_loss', self.total_loss, self.regular_iter)
        self.writer.add_scalar('l2_error_u', self.l2_error_u, self.regular_iter)
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
            
            for i in range(len(self.model.trunk_list)):
                end_string = '_state_dict'
                key = self.config['trunk_control']['trunk_list_ID'][i] + end_string
                aux_dict = {key: self.model.trunk_list[i].state_dict()}
                model_weights.update(aux_dict)
            
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

    def random_sampling(self, xt, label = 'val'):
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
        
        xt_bc = {} 
        u_bc = {} 
    
        
        xt_val = {} 
        u_val = {}

        for chosen_flow_label in self.config['train']['training_param_label']:


            #---------------------
            # Fixed dataset
            #---------------------
            ## BOUNDARY CONDITIONS AND INITIAL CONDITION
            xt_bc[chosen_flow_label], u_bc[chosen_flow_label], xt_ic[chosen_flow_label], u_ic[chosen_flow_label]= craft_bc_and_ic_dataset(full_ds[chosen_flow_label], time_vector)

            ## VALIDATION DATASET
            xt_val[chosen_flow_label], u_val[chosen_flow_label] = craft_validation_dataset(val_ds[chosen_flow_label], time_vector)


        # """The coordinate points are the same between the different datasets, so we can choose a geometry among the options"""
        coordinates, time_min, time_max = get_coordinates_for_generator(full_ds[self.config['train']['training_param_label'][0]], time_vector)


        # Workflow
        self.custom_bar = trange(self.last_iter + 1)

        # Branch
        branch_input = {}
        for key in self.config['branches_control']['branch_input_ID']:
            branch_input.update({key:[]})

        ## Random Sampling - Get Fixed Indexes
        # BC 
        idx_bc_fixed = self.random_sampling(xt_bc[chosen_flow_label], label = 'bc')
        
        for chosen_flow_label in self.config['train']['training_param_label']:
                
            # INLET   
            u_bc_sample = u_bc[chosen_flow_label][idx_bc_fixed]    
            
            ## Branch
            # for i in range(N_vel_branch_inlet):
                # index = self.config['branches_control']['axis_indexes'][self.config['branches_control']['vel_axis_ID'][i]]
                # branch_input[self.config['branches_control']['branch_input_ID'][i]].append(vel_bc_inlet_sample[:,index].T)
            branch_input[self.config['branches_control']['branch_input_ID'][0]].append(u_bc_sample.T)
            
        
        for regular_iter in self.custom_bar:

            self.regular_iter = regular_iter

            # Trunk
            xt_ic_sample_all = []
            u_ic_target = []
            
            xt_bc_sample_all = []
            u_bc_target = []
           
           
            # Val
            xt_val_sample_all = []
            u_val_target = []
           

            ## Random Sampling - Get Indexes
            # IC 
            idx_ic = self.random_sampling(xt_ic[chosen_flow_label], label = 'ic')
            
            # BC 
            idx_bc = self.random_sampling(xt_bc[chosen_flow_label],  label = 'bc')
        

            # VAL 
            idx_val = self.random_sampling(xt_val[chosen_flow_label],  label = 'val')

            for chosen_flow_label in self.config['train']['training_param_label']:
                
                # IC 
                xt_ic_sample =  xt_ic[chosen_flow_label][idx_ic]
                u_ic_sample = u_ic[chosen_flow_label][idx_ic]
                
                # BC 
                xt_bc_sample =  xt_bc[chosen_flow_label][idx_bc]
                u_bc_sample = u_bc[chosen_flow_label][idx_bc]
                

                # VAL 
                xt_val_sample = xt_val[chosen_flow_label][idx_val]
                u_val_sample = u_val[chosen_flow_label][idx_val]

                ## Trunk
                xt_ic_sample_all.append(xt_ic_sample)
                u_ic_target.append(u_ic_sample)
                
                xt_bc_sample_all.append(xt_bc_sample)
                u_bc_target.append(u_bc_sample)
                

                xt_val_sample_all.append(xt_val_sample)
                u_val_target.append(u_val_sample)

            # Trunk
            xt_ic_sample_all = np.array(xt_ic_sample_all)
            u_ic_target = np.array(u_ic_target)
            
            xt_bc_sample_all = np.array(xt_bc_sample_all)
            u_bc_target = np.array(u_bc_target)

            xt_val_sample_all = np.array(xt_val_sample_all)
            u_val_target = np.array(u_val_target)
            
            # IC
            # N = 6, P = ?
            self.xt_ic_tensor = []
            xt_ic_tensor = torch.tensor(np.swapaxes(xt_ic_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            self.xt_ic_tensor.append(xt_ic_tensor)
          
            self.f_ic_tensor = []
            num_samples = xt_ic_sample.shape[0]  # 51850
            target_device = self.config['device']

            for key in branch_input.keys():
                
                base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]),dtype=torch.float32,device=target_device)
                
                final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
                
                self.f_ic_tensor.append(final_view)
            
            self.u_ic_tensor = torch.tensor(np.swapaxes(u_ic_target,0,1).reshape(-1,self.output_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            # BC
            self.xt_bc_tensor = []
            xt_bc_tensor = torch.tensor(np.swapaxes(xt_bc_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            self.xt_bc_tensor.append(xt_bc_tensor)
            
            self.f_bc_tensor = []
            num_samples = xt_bc_sample.shape[0]  
            target_device = self.config['device']
            
            for key in branch_input.keys():
                
                base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]),dtype=torch.float32,device=target_device)
                
                final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
                
                self.f_bc_tensor.append(final_view)
            
            self.u_bc_tensor = torch.tensor(np.swapaxes(u_bc_target,0,1).reshape(-1,self.output_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
           
           
            # Validation dataset
            # N = 6, P = xyzt_val_fixed.shape[0]
            self.xt_val_tensor = []
            xt_val_tensor = torch.tensor(np.swapaxes(xt_val_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            self.xt_val_tensor.append(xt_val_tensor)
            
            
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
                
                beta = []
                for i, xt_val_tensor_i in enumerate(self.xt_val_tensor):
                    beta.append(self.model.trunk_list[i](xt_val_tensor_i))

                tau[0] = tau[0].reshape(-1, self.branch_out_dim)

                u_pred = torch.sum(tau[0] * beta[0], axis = 1)[:, None]
               
            
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

        self.branch_in_dim = self.config['branch1']['neuralNet']['in_dim']
        self.branch_out_dim = self.config['branch1']['neuralNet']['out_dim']
            
    def test_full(self, dataloader, N_batches, branch_input, subset = 'test', param_label = None):

        history_log = f"""LOG TEST ON {param_label} - {subset}_dataset\n"""
       
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

                beta = []
                beta.append(self.model.trunk_list[0](inputs))

                ## np.tile
                tau[0] = tau[0].reshape(-1, self.branch_out_dim)

                # Predicted
                u_pred = torch.sum(tau[0] * beta[0], axis = 1)[:, None]


            l2_relative_error_u.append(metric_l2_relative_error(exact = u_ref, pred = u_pred).cpu().numpy())

            
            # Save into a .txt file
            history_log = history_log + \
            "\n-------------------------\n" + \
            f"""Batch: {(batch_iter + 1)}/{(N_batches)} - 'l2_relative_error_vel': {l2_relative_error_u[batch_iter]} \n"""
         
            
        # Save into a .txt file
        with open(self.config['test_progress_file_path'], 'a') as file:
            file.write(history_log + '\n\n\n')

        self.config['logger'].info(f"Accuracy in {param_label} - {subset}_dataset")
        self.config['logger'].info(f"L2 relative error in vel: {np.mean(np.array(l2_relative_error_u))}")
        
        print(f"Accuracy in {param_label} - {subset}dataset")
        print(f"L2 relative error in vel: {np.mean(np.array(l2_relative_error_u))}")

    def test_value(self, k_i, t_i, c, branch_input, label = 'InflowBC_K056'):
        
        # Exact
        x_array = np.linspace(0, 1, 1000)[:,None]
        phi_i = calculate_phi(k_i)
        u_exact = u_close_form(x_array, t_i, c, k_i, phi_i)
        
        u_exact_tensor = torch.tensor(u_exact, dtype=torch.float32, requires_grad = False).to(self.config['device'])

    
        # N = 1, P = batch_size
        t_array = np.repeat(np.array([t_i]), x_array.shape[0], axis=0)[:,None]

        xt_array = np.hstack([x_array, t_array])

        xt_tensor = torch.tensor(xt_array, dtype = torch.float32, requires_grad= False).to(self.config['device'])
        
        
        f_tensor = []
        num_samples = xt_tensor.shape[0]
        target_device = self.config['device']

        for key in branch_input.keys():
                
            base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]),dtype=torch.float32,device=target_device)
                
            final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
                
            f_tensor.append(final_view)
        
        
        
        

        with torch.no_grad():
            tau = self.model.branch_list[0](f_tensor[0])
            beta = self.model.trunk_list[0](xt_tensor)
            ## np.tile
            tau = tau.reshape(-1, self.branch_out_dim)
        u_pred_tensor = torch.sum(beta * tau, axis = 1)[:,None]
            

        
        l2_relative_error = torch.linalg.norm((u_exact_tensor-u_pred_tensor), 2)/torch.linalg.norm(u_exact_tensor, 2)
        print(f'l2_relative_error={l2_relative_error.item()} for t = {t_i} and {label}')

        u_pred = u_pred_tensor.cpu().numpy()
        

        return x_array, u_exact, u_pred, l2_relative_error.item()
    
    def visualize_comparison_per_value(self, t_all, f_a_i, x_final, u_exact_final, u_pred_final, label = 'InflowBC_K056'):
       

        fig, ax = plt.subplots(figsize=(5, 5))

        # Create a color gradient for lines and points
        line_colors_exact = plt.cm.Blues(np.linspace(0.5, 1, len(t_all)))
        line_colors_pred = plt.cm.Reds(np.linspace(0.5, 1, len(t_all)))
        point_colors = plt.cm.Greens(np.linspace(0.5, 1, len(t_all)))

        for i,value in enumerate(t_all):
            
            ax.plot(x_final[i], u_exact_final[i], color=line_colors_exact[i], linewidth=3, label=f't = {value} and {f_a_i}')
            ax.plot(x_final[i], u_pred_final[i], '--',color=line_colors_pred[i], linewidth=3)
           

        # Setting up the plot
        ax.set_xlabel('x',fontsize=18)
        ax.set_ylabel('u',fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax.set_xlim(0, 1)
        # ax.set_ylim(0.8, 4)
        ax.set_ylim(-0.5, 1.5)
        # ax.set_ylim(0.4, 2.5)
        ax.legend()
        # ax.set_title('Exact Solution for Different a Values')

        # plt.show()

        # Save the figure
        fig_path = self.config['charts_folder_path'].joinpath(f'comparison_exact_vs_predicted_{label}.png')
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
      
    def visualize_comparison_fulldomain(self, branch_input, k_i, c, label):
        

        # ------------------------------------------------------------------
        # 1. Create a *regular space-time grid* with a single, unambiguous rule:
        #    - first axis  →  time  (Nt points)
        #    - second axis →  space (Nx points)
        # ------------------------------------------------------------------
        t_star = np.linspace(0.0, 1.0, 100)      # Nt
        x_star = np.linspace(0.0, 1.0, 1000)     # Nx

        TT, XX = np.meshgrid(t_star, x_star, indexing="ij")  # TT,XX ∈ ℝ[Nt,Nx]

        t_flat = TT.reshape(-1, 1)               # ℝ[Nt·Nx,1]
        x_flat = XX.reshape(-1, 1)

        # ------------------------------------------------------------------
        # 2. Exact solution on the same grid
        # ------------------------------------------------------------------
        phi_i = calculate_phi(k_i)  
        u_ref = u_close_form(x_flat, t_flat, c, k_i, phi_i)  # expects (x,t,c)

        # ------------------------------------------------------------------
        # 3. Convert to torch & push through the network  (xt order)
        # ------------------------------------------------------------------
        device = self.config["device"]

        t = torch.as_tensor(t_flat, dtype=torch.float32, device=device)
        x = torch.as_tensor(x_flat, dtype=torch.float32, device=device)

        # >>> trunk expects (x , t)  <<<  so concatenate in that order
        xt = torch.cat([x, t], dim=1)        # shape: (N, 2)

        # == Branch inputs (unchanged) =====================================
        f_tensor = []
        num_samples = xt.shape[0]
        for i, key in enumerate(branch_input.keys()):
            base = torch.as_tensor(np.vstack(branch_input[key]),
                                dtype=torch.float32,
                                device=device)
            f_tensor.append(base.unsqueeze(0).expand(num_samples, -1, -1))

        # == PINN prediction ===============================================
        with torch.no_grad():
            tau = [self.model.branch_list[i](f_tensor_i) for i, f_tensor_i in enumerate(f_tensor)]
            beta = self.model.trunk_list[0](xt) 
            tau[0] = tau[0].view(-1, self.branch_out_dim)
            u_pred = (tau[0] * beta).sum(dim=1, keepdim=True)

            
        # ------------------------------------------------------------------
        # 4. Error metrics
        # ------------------------------------------------------------------
        l2_relative_error = metric_l2_relative_error(
            exact=torch.as_tensor(u_ref, device=device, dtype=torch.float32),
            pred=u_pred
        )

        self.config['logger'].info(f'l2_relative_error={l2_relative_error} for {label}')
        print(f'l2_relative_error={l2_relative_error} for {label}')

        # ------------------------------------------------------------------
        # 5. Reshape back to (Nt,Nx) – NO transposes, the axes are correct
        # ------------------------------------------------------------------
        Nt, Nx = TT.shape
        u_pred = u_pred.cpu().numpy().reshape(Nt, Nx)
        u_ref  = u_ref.reshape(Nt, Nx)
        abs_err = np.abs(u_ref - u_pred)

        # ------------------------------------------------------------------
        # 6. Plot
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

        titles = [f"Exact u(t,x) ({label})", "Predicted u(t,x)", "Absolute error"]
        data   = [u_ref, u_pred, abs_err]

        for ax, z, title in zip(axes, data, titles):
            pcm = ax.pcolormesh(t_star, x_star, z.T, cmap="jet", shading="auto")
            fig.colorbar(pcm, ax=ax)
            ax.set_xlabel("t")
            ax.set_ylabel("x")
            ax.set_title(title)

        # ------------------------------------------------------------------
        # 7. Save & close
        # ------------------------------------------------------------------
        fig_path = self.config["charts_folder_path"] / f"comparison_fulldomain_{label}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # (optional) return the relative L2 error for logging
        return float(l2_relative_error)
        
#--------------------------
class modified_NeuralNetwork(torch.nn.Module):

    def __init__(self, config):

        super().__init__()

        self.config = config
      
        # NN Architectures
        if config['branch1']['neuralNet']['architecture'] == "Modified Deeponet":
            self.mdona = ModifiedDeepONetArch(config)
        else:
            raise ValueError("Invalid architecture")
       
class Trainer_PIDeepONetLdata_modified(torch.nn.Module):

    def __init__(self, config, model):

        super().__init__()
        
        self.config = config
        self.has_exponential_decay = self.config['optim1']['exponential_decay']['enabled']
        self.has_loss_balancing = self.config['loss_balancing']['enabled']

        # Initialize the SummaryWriter
        if self.config['mode']=='train':
            self.writer = SummaryWriter(log_dir=config['logs_folder_path'])

        self.model = model
        self.branch_in_dim = config['branch1']['neuralNet']['in_dim']
        self.branch_out_dim = config['branch1']['neuralNet']['out_dim']
        self.trunk_in_dim = config['trunk1']['neuralNet']['in_dim']
        self.output_dim = 1

        # PDE parameters
        self.c = config['dataset']['c']   

        # Optimizers
        if config['optim1']['optimizer'] == 'Adam':
            
            param_groups = list(self.model.mdona.param_groups)

            self.optimizer_Adam = torch.optim.Adam(param_groups, lr=config['optim1']['learning_rate'], \
                                                    betas=(config['optim1']['beta1'], config['optim1']['beta2']), \
                                                    eps=config['optim1']['eps'] )
            if self.has_exponential_decay:
                def lr_lambda_func(current_step):
                    """Custom function for the exponential decay"""
                    
                    decay_rate = self.config['optim1']['exponential_decay']['decay_rate']
                    decay_steps = self.config['optim1']['exponential_decay']['decay_steps']

                    factor = decay_rate ** (current_step / decay_steps)

                    return factor

                self.scheduler_Adam = torch.optim.lr_scheduler.LambdaLR(self.optimizer_Adam, lr_lambda_func)
       
        if config['optim2']['optimizer'] == 'LBFGS':

            all_params = [param for group in self.model.mdona.param_groups for param in group['params']]

            self.optimizer_LBFGS = torch.optim.LBFGS(all_params)
           
        
        # Loss function
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        

        # Logging
        self.total_loss = None
        self.l2_error_u = None

        # Loss terms
        self.term_loss_tensor_dict = {key:[] for key in self.config['loss_terms']}

        # Predicted output terms
        self.term_s_tensor_dict = {key:[] for key in self.config['loss_terms']}

        # lambdas initialization
        self.term_lambdas_tensor = torch.tensor(
                                    [self.config['lambda_weights_init'][i] for i in range(len(self.config['loss_terms']))], 
                                    dtype=torch.float32, requires_grad=False).to(self.config['device'])

        self.term_lambdas_tensor_dict = {key: self.term_lambdas_tensor[i] for i,key in enumerate(self.config['loss_terms'])}

         # Loss balancing
        scheme = self.config['loss_balancing']['scheme']
        if self.has_loss_balancing and scheme == 'ntk_guided_weights':
            
            self.K = {key:[] for key in self.config['loss_terms']}

            self.config['logger'].info(f"Using {scheme} for loss balancing.")
            self.loss_balancing_log = f"Loss Balancing - {scheme}\n"


        self.log_log = """LOG\n"""
        self.min_error_for_checkpoint = np.Infinity
        self.runtime0 = self.config['tick_start']
        self.best_iter = 0
     
    def loss_boundary_condition(self):

        u_pred = self.model.mdona(self.f_bc_tensor[0], self.xt_bc_tensor)
       
        # Ground Trust
        u_GT = self.u_bc_tensor
        
        # Compute losses
        u_loss = self.loss_fn(u_pred, u_GT)

        return u_loss, u_pred
    
    def loss_data(self):

        u_pred = self.model.mdona(self.f_data_tensor[0], self.xt_data_tensor)
       
        # Ground Trust
        u_GT = self.u_data_tensor
        
        # Compute losses
        u_loss = self.loss_fn(u_pred, u_GT)

    
        return u_loss, u_pred
    
    def loss_initial_condition(self):

        u_pred = self.model.mdona(self.f_ic_tensor[0], self.xt_ic_tensor)
       
        # Ground Trust
        u_GT = self.u_ic_tensor
        
        # Compute losses
        u_loss = self.loss_fn(u_pred, u_GT)

    
        return u_loss, u_pred
    
    def loss_physics(self):

        u = self.model.mdona(self.f_phy_tensor[0], torch.cat([self.x_phy, self.t_phy], 1))

        # Autodiff
        u_x = torch.autograd.grad(
            u, self.x_phy, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        u_t = torch.autograd.grad(
            u, self.t_phy, grad_outputs=torch.ones_like(u), create_graph=True)[0]


        EQ1 = u_t + self.c *  u_x 
       

        # Compute losses
        loss = self.loss_fn(EQ1, self.residual_target)
       
        return loss, u
        
    def loss(self):

        # Forward pass and compute the losses per terms
        loss_ic, pred_ic = self.loss_initial_condition()
        loss_bc, pred_bc = self.loss_boundary_condition()
        loss_data, pred_data = self.loss_data()
        loss_p, pred_phy = self.loss_physics()

        self.term_loss_tensor_dict['ic'] = loss_ic
        self.term_loss_tensor_dict['bc'] = loss_bc
        self.term_loss_tensor_dict['data'] = loss_data
        self.term_loss_tensor_dict['phy'] = loss_p

        self.term_s_tensor_dict['ic'] = pred_ic
        self.term_s_tensor_dict['bc'] = pred_bc
        self.term_s_tensor_dict['data'] = pred_data
        self.term_s_tensor_dict['phy'] = pred_phy
        
        
        # Apply adaptive lambdas
        self.loss_balancing_call()
        
    
        # Compute total loss
        loss_total = self.term_lambdas_tensor_dict['ic'] * loss_ic + self.term_lambdas_tensor_dict['bc'] * loss_bc + self.term_lambdas_tensor_dict['data'] * loss_data + self.term_lambdas_tensor_dict['phy'] * loss_p 


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

        if self.config['loss_balancing']['scheme'] == 'no_weights':
            pass

        elif self.config['loss_balancing']['scheme'] == 'fixed_weights':
            pass
        
        elif self.config['loss_balancing']['scheme'] == 'data_guided_weights':
            
            # Create a copy of the tensor values to avoid gradient issues
            term_s_tensor_dict_copy = {
                key: tensor.detach().clone() 
                for key, tensor in self.term_s_tensor_dict.items()
            }

            for idx, key in enumerate(self.config['loss_terms']):
                # Use detached tensor to compute max
                max_val = torch.max(torch.abs(term_s_tensor_dict_copy[key]))
                self.term_lambdas_tensor[idx] = 1/max_val.detach()

        elif self.config['loss_balancing']['scheme'] == 'ntk_guided_weights':
                
            # 0. grab *exactly* the parameters used by the optimiser -------------
            all_params = [param for group in self.model.mdona.param_groups for param in group['params']]

            # 1. gather losses --------------------------------------------------
            losses = [self.term_loss_tensor_dict[k] for k in self.config["loss_terms"]]

            # 2a. build gradient matrix  (n_terms × n_params) ------------------
            grad_mat = torch.stack([
                self.flat_grad(loss, all_params).detach()
                for loss in losses
            ]) #                ↑ detach: NTK weights should not be back-proped

            # 2b. NTK diagonal --------------------------------------------------
            k_diag = self.ntk_diag_from_grads(grad_mat)     # shape (n_terms,)

            # 3. λ update -------------------------------------------------------
            lambdas = self.normalise_ntk(
                k_diag,                                # stays on-device
                self.config["loss_balancing"]["type"]
            ).to(self.config["device"])

            # 4. # update lambdas -----------------------------------------
            self.term_lambdas_tensor = lambdas
        
        else:
            pass
    
   
    def normalise_ntk(self, k_vec: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Vectorised λ update:   λ_k = (‖H‖_∞ / H_kk)^α   with α ∈ {1,½}.
        """
        if mode == "global_NTK_weights":      # α = 1, global sum variant
            return k_vec.sum() / k_vec
        if mode == "local_NTK_weights":       # α = 1, max variant
            return k_vec.max() / k_vec
        if mode == "moderate_local_NTK_weights":  # α = ½
            return torch.sqrt(k_vec.max() / k_vec)

        raise ValueError(f"Unknown NTK scheme: {mode}")


    def flat_grad(self, loss: torch.Tensor,
                params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
        """
        Return a 1-D tensor containing ∂loss/∂θ for *all* parameters,
        inserting zeros where autograd returns None.
        """
        grads = torch.autograd.grad(
            loss, params,
            retain_graph=True, create_graph=False, allow_unused=True
        )
        vec = [
            (g if g is not None else torch.zeros_like(p)).flatten()
            for g, p in zip(grads, params)
        ]
        return torch.cat(vec)                      # shape: (n_params,)


    def ntk_diag_from_grads(self, grad_mat: torch.Tensor) -> torch.Tensor:
        """
        Given the gradient matrix of shape (n_terms, n_params),
        return the vector of NTK diagonal entries.
        """
        return (grad_mat ** 2).sum(dim=1)          # row-wise ‖·‖₂²
    
    
    def logger_call(self):

        # Monitoring resources 
        if self.config['logging']['monitoring_resources'] and self.regular_iter == 0:

            self.log_log += "\n---------\n"

            # CPU / system-RAM
            self.log_log += get_cpu_memory_usage("") + "\n"

            # GPU (if any)
            if torch.cuda.is_available():
                cuda_device = torch.device("cuda")  # default current device
                self.log_log += get_cuda_memory_usage(cuda_device,"") + "\n"
            else:
                self.log_log += "No GPU detected\n"
        
        # Total and term losses
        loss_train = self.total_loss
        l2_error_u = self.l2_error_u
        
        
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
     
                
        # Save into a .txt file
        with open(self.config['train_progress_file_path'], 'w') as file:
            file.write(self.log_log)
        
        # Tensorboard
        self.writer.add_scalar('total_loss', self.total_loss, self.regular_iter)
        self.writer.add_scalar('l2_error_u', self.l2_error_u, self.regular_iter)
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
           
            model_weights = {
                'state_dict': self.model.mdona.state_dict(),
            }

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

    def random_sampling(self, xt, label = 'val'):
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
        
        xt_bc = {} 
        u_bc = {} 
        
        xt_data = {} 
        u_data = {} 
        
        xt_val = {} 
        u_val = {}
      

        for chosen_flow_label in self.config['train']['training_param_label']:


            #---------------------
            # Fixed dataset
            #---------------------
            ## BOUNDARY CONDITIONS AND INITIAL CONDITION
            xt_bc[chosen_flow_label], u_bc[chosen_flow_label], xt_ic[chosen_flow_label], u_ic[chosen_flow_label]= craft_bc_and_ic_dataset(full_ds[chosen_flow_label], time_vector)
            
            ## DATA 
            xt_data[chosen_flow_label], u_data[chosen_flow_label] = build_stratum_dataset(dataPrm[chosen_flow_label], time_vector, label = 'DATA')

            ## VALIDATION DATASET
            xt_val[chosen_flow_label], u_val[chosen_flow_label] = craft_validation_dataset(val_ds[chosen_flow_label], time_vector)


        # """The coordinate points are the same between the different datasets, so we can choose a geometry among the options"""
        coordinates, time_min, time_max = get_coordinates_for_generator(full_ds[self.config['train']['training_param_label'][0]], time_vector)


        # Workflow
        self.custom_bar = trange(self.last_iter + 1)
        
         # Branch
        branch_input = {}
        for key in self.config['branches_control']['branch_input_ID']:
            branch_input.update({key:[]})

        ## Random Sampling - Get Fixed Indexes
        # BC 
        idx_bc_fixed = self.random_sampling(xt_bc[chosen_flow_label], label = 'bc')
        
        for chosen_flow_label in self.config['train']['training_param_label']:
                
            # INLET  
            u_bc_sample = u_bc[chosen_flow_label][idx_bc_fixed]    
            
            ## Branch
            # for i in range(N_vel_branch_inlet):
                # index = self.config['branches_control']['axis_indexes'][self.config['branches_control']['vel_axis_ID'][i]]
                # branch_input[self.config['branches_control']['branch_input_ID'][i]].append(vel_bc_inlet_sample[:,index].T)
            branch_input[self.config['branches_control']['branch_input_ID'][0]].append(u_bc_sample.T)
            
        

        for regular_iter in self.custom_bar:

            self.regular_iter = regular_iter
            
            # Trunk
            xt_ic_sample_all = []
            u_ic_target = []
            
            xt_bc_sample_all = []
            u_bc_target = []
           
            xt_data_sample_all = []
            u_data_target = []

           
            # Val
            xt_val_sample_all = []
            u_val_target = []
           

            ## Random Sampling - Get Indexes
            # IC 
            idx_ic = self.random_sampling(xt_ic[chosen_flow_label], label = 'ic')
            
            # BC 
            idx_bc = self.random_sampling(xt_bc[chosen_flow_label], label = 'bc')
            
            # DATA 
            idx_data = self.random_sampling(xt_data[chosen_flow_label], label = 'data')

            # VAL 
            idx_val = self.random_sampling(xt_val[chosen_flow_label], label = 'val')

            for chosen_flow_label in self.config['train']['training_param_label']:
                
                # IC 
                xt_ic_sample =  xt_ic[chosen_flow_label][idx_ic]
                u_ic_sample = u_ic[chosen_flow_label][idx_ic]
                
                # BC 
                xt_bc_sample =  xt_bc[chosen_flow_label][idx_bc]
                u_bc_sample = u_bc[chosen_flow_label][idx_bc]
                
                # DATA 
                xt_data_sample =  xt_data[chosen_flow_label][idx_data]
                u_data_sample = u_data[chosen_flow_label][idx_data]

                # VAL 
                xt_val_sample = xt_val[chosen_flow_label][idx_val]
                u_val_sample = u_val[chosen_flow_label][idx_val]

                ## Trunk
                xt_ic_sample_all.append(xt_ic_sample)
                u_ic_target.append(u_ic_sample)
                
                xt_bc_sample_all.append(xt_bc_sample)
                u_bc_target.append(u_bc_sample)
                
                xt_data_sample_all.append(xt_data_sample)
                u_data_target.append(u_data_sample)

                xt_val_sample_all.append(xt_val_sample)
                u_val_target.append(u_val_sample)

            # Trunk
            xt_ic_sample_all = np.array(xt_ic_sample_all)
            u_ic_target = np.array(u_ic_target)
            
            xt_bc_sample_all = np.array(xt_bc_sample_all)
            u_bc_target = np.array(u_bc_target)
            
            xt_data_sample_all = np.array(xt_data_sample_all)
            u_data_target = np.array(u_data_target)

            xt_val_sample_all = np.array(xt_val_sample_all)
            u_val_target = np.array(u_val_target)
            
            # IC
            # N = 6, P = ? 
            self.xt_ic_tensor = []
            xt_ic_tensor = torch.tensor(np.swapaxes(xt_ic_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            self.xt_ic_tensor.append(xt_ic_tensor)

            self.f_ic_tensor = []
            num_samples = xt_ic_sample.shape[0]  # 51850
            target_device = self.config['device']

            for key in branch_input.keys():
                
                base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]),dtype=torch.float32,device=target_device)
                
                final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
                
                self.f_ic_tensor.append(final_view)
            
            self.u_ic_tensor = torch.tensor(np.swapaxes(u_ic_target,0,1).reshape(-1,self.output_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            # BC
            self.xt_bc_tensor = []
            xt_bc_tensor = torch.tensor(np.swapaxes(xt_bc_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            self.xt_bc_tensor.append(xt_bc_tensor)

            self.f_bc_tensor = []
            num_samples = xt_bc_sample.shape[0]  
            target_device = self.config['device']
            
            for key in branch_input.keys():
                
                base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]),dtype=torch.float32,device=target_device)
                
                final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
                
                self.f_bc_tensor.append(final_view)
            
            self.u_bc_tensor = torch.tensor(np.swapaxes(u_bc_target,0,1).reshape(-1,self.output_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            # DATA
            self.xt_data_tensor = []
            xt_data_tensor = torch.tensor(np.swapaxes(xt_data_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            self.xt_data_tensor.append(xt_data_tensor)

            self.f_data_tensor = []
            num_samples = xt_data_sample.shape[0]  
            target_device = self.config['device']

            for key in branch_input.keys():
                
                base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]),dtype=torch.float32,device=target_device)
                
                final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
                
                self.f_data_tensor.append(final_view)
            
            self.u_data_tensor = torch.tensor(np.swapaxes(u_data_target,0,1).reshape(-1,self.output_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            
            
           
            # Validation dataset
            # N = 6, P = xyzt_val_fixed.shape[0]
            self.xt_val_tensor = []
            xt_val_tensor = torch.tensor(np.swapaxes(xt_val_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            self.xt_val_tensor.append(xt_val_tensor)
            
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
            

            with torch.no_grad():
                u_pred = self.model.mdona(self.f_val_tensor[0], self.xt_val_tensor)
               
            
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
   

class Trainer_PIDeepONet_modified(torch.nn.Module):

    def __init__(self, config, model):

        super().__init__()
        
        self.config = config
        self.has_exponential_decay = self.config['optim1']['exponential_decay']['enabled']
        self.has_loss_balancing = self.config['loss_balancing']['enabled']

        # Initialize the SummaryWriter
        if self.config['mode']=='train':
            self.writer = SummaryWriter(log_dir=config['logs_folder_path'])

        self.model = model
        self.branch_in_dim = config['branch1']['neuralNet']['in_dim']
        self.branch_out_dim = config['branch1']['neuralNet']['out_dim']
        self.trunk_in_dim = config['trunk1']['neuralNet']['in_dim']
        self.output_dim = 1

        # PDE parameters
        self.c = config['dataset']['c']   

        # Optimizers
        if config['optim1']['optimizer'] == 'Adam':
            
            param_groups = list(self.model.mdona.param_groups)

            self.optimizer_Adam = torch.optim.Adam(param_groups, lr=config['optim1']['learning_rate'], \
                                                    betas=(config['optim1']['beta1'], config['optim1']['beta2']), \
                                                    eps=config['optim1']['eps'] )
            if self.has_exponential_decay:
                def lr_lambda_func(current_step):
                    """Custom function for the exponential decay"""
                    
                    decay_rate = self.config['optim1']['exponential_decay']['decay_rate']
                    decay_steps = self.config['optim1']['exponential_decay']['decay_steps']

                    factor = decay_rate ** (current_step / decay_steps)

                    return factor

                self.scheduler_Adam = torch.optim.lr_scheduler.LambdaLR(self.optimizer_Adam, lr_lambda_func)
       
        if config['optim2']['optimizer'] == 'LBFGS':

            all_params = [param for group in self.model.mdona.param_groups for param in group['params']]

            self.optimizer_LBFGS = torch.optim.LBFGS(all_params)
           
        
        # Loss function
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        

        # Logging
        self.total_loss = None
        self.l2_error_u = None

        # Loss terms
        self.term_loss_tensor_dict = {key:[] for key in self.config['loss_terms']}

        # Predicted output terms
        self.term_s_tensor_dict = {key:[] for key in self.config['loss_terms']}

        # lambdas initialization
        self.term_lambdas_tensor = torch.tensor(
                                    [self.config['lambda_weights_init'][i] for i in range(len(self.config['loss_terms']))], 
                                    dtype=torch.float32, requires_grad=False).to(self.config['device'])

        self.term_lambdas_tensor_dict = {key: self.term_lambdas_tensor[i] for i,key in enumerate(self.config['loss_terms'])}

         # Loss balancing
        scheme = self.config['loss_balancing']['scheme']
        if self.has_loss_balancing and scheme == 'ntk_guided_weights':
            
            self.K = {key:[] for key in self.config['loss_terms']}

            self.config['logger'].info(f"Using {scheme} for loss balancing.")
            self.loss_balancing_log = f"Loss Balancing - {scheme}\n"


        self.log_log = """LOG\n"""
        self.min_error_for_checkpoint = np.Infinity
        self.runtime0 = self.config['tick_start']
        self.best_iter = 0
     
    def loss_boundary_condition(self):

        u_pred = self.model.mdona(self.f_bc_tensor[0], self.xt_bc_tensor)
       
        # Ground Trust
        u_GT = self.u_bc_tensor
        
        # Compute losses
        u_loss = self.loss_fn(u_pred, u_GT)

        return u_loss, u_pred
    
    
    def loss_initial_condition(self):

        u_pred = self.model.mdona(self.f_ic_tensor[0], self.xt_ic_tensor)
       
        # Ground Trust
        u_GT = self.u_ic_tensor
        
        # Compute losses
        u_loss = self.loss_fn(u_pred, u_GT)

    
        return u_loss, u_pred
    
    def loss_physics(self):

        u = self.model.mdona(self.f_phy_tensor[0], torch.cat([self.x_phy, self.t_phy], 1))

        # Autodiff
        u_x = torch.autograd.grad(
            u, self.x_phy, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        u_t = torch.autograd.grad(
            u, self.t_phy, grad_outputs=torch.ones_like(u), create_graph=True)[0]


        EQ1 = u_t + self.c *  u_x 
       

        # Compute losses
        loss = self.loss_fn(EQ1, self.residual_target)
       
        return loss, u
        
    def loss(self):

        # Forward pass and compute the losses per terms
        loss_ic, pred_ic = self.loss_initial_condition()
        loss_bc, pred_bc = self.loss_boundary_condition()
        loss_p, pred_phy = self.loss_physics()

        self.term_loss_tensor_dict['ic'] = loss_ic
        self.term_loss_tensor_dict['bc'] = loss_bc
        self.term_loss_tensor_dict['phy'] = loss_p

        self.term_s_tensor_dict['ic'] = pred_ic
        self.term_s_tensor_dict['bc'] = pred_bc
        self.term_s_tensor_dict['phy'] = pred_phy
        
        
        # Apply adaptive lambdas
        self.loss_balancing_call()
        
    
        # Compute total loss
        loss_total = self.term_lambdas_tensor_dict['ic'] * loss_ic + self.term_lambdas_tensor_dict['bc'] * loss_bc + self.term_lambdas_tensor_dict['phy'] * loss_p 


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

        if self.config['loss_balancing']['scheme'] == 'no_weights':
            pass

        elif self.config['loss_balancing']['scheme'] == 'fixed_weights':
            pass
        
        elif self.config['loss_balancing']['scheme'] == 'data_guided_weights':
            
            # Create a copy of the tensor values to avoid gradient issues
            term_s_tensor_dict_copy = {
                key: tensor.detach().clone() 
                for key, tensor in self.term_s_tensor_dict.items()
            }

            for idx, key in enumerate(self.config['loss_terms']):
                # Use detached tensor to compute max
                max_val = torch.max(torch.abs(term_s_tensor_dict_copy[key]))
                self.term_lambdas_tensor[idx] = 1/max_val.detach()

        elif self.config['loss_balancing']['scheme'] == 'ntk_guided_weights':
                
            # 0. grab *exactly* the parameters used by the optimiser -------------
            all_params = [param for group in self.model.mdona.param_groups for param in group['params']]

            # 1. gather losses --------------------------------------------------
            losses = [self.term_loss_tensor_dict[k] for k in self.config["loss_terms"]]

            # 2a. build gradient matrix  (n_terms × n_params) ------------------
            grad_mat = torch.stack([
                self.flat_grad(loss, all_params).detach()
                for loss in losses
            ]) #                ↑ detach: NTK weights should not be back-proped

            # 2b. NTK diagonal --------------------------------------------------
            k_diag = self.ntk_diag_from_grads(grad_mat)     # shape (n_terms,)

            # 3. λ update -------------------------------------------------------
            lambdas = self.normalise_ntk(
                k_diag,                                # stays on-device
                self.config["loss_balancing"]["type"]
            ).to(self.config["device"])

            # 4. # update lambdas -----------------------------------------
            self.term_lambdas_tensor = lambdas
        
        else:
            pass
    
   
    def normalise_ntk(self, k_vec: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Vectorised λ update:   λ_k = (‖H‖_∞ / H_kk)^α   with α ∈ {1,½}.
        """
        if mode == "global_NTK_weights":      # α = 1, global sum variant
            return k_vec.sum() / k_vec
        if mode == "local_NTK_weights":       # α = 1, max variant
            return k_vec.max() / k_vec
        if mode == "moderate_local_NTK_weights":  # α = ½
            return torch.sqrt(k_vec.max() / k_vec)

        raise ValueError(f"Unknown NTK scheme: {mode}")


    def flat_grad(self, loss: torch.Tensor,
                params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
        """
        Return a 1-D tensor containing ∂loss/∂θ for *all* parameters,
        inserting zeros where autograd returns None.
        """
        grads = torch.autograd.grad(
            loss, params,
            retain_graph=True, create_graph=False, allow_unused=True
        )
        vec = [
            (g if g is not None else torch.zeros_like(p)).flatten()
            for g, p in zip(grads, params)
        ]
        return torch.cat(vec)                      # shape: (n_params,)


    def ntk_diag_from_grads(self, grad_mat: torch.Tensor) -> torch.Tensor:
        """
        Given the gradient matrix of shape (n_terms, n_params),
        return the vector of NTK diagonal entries.
        """
        return (grad_mat ** 2).sum(dim=1)          # row-wise ‖·‖₂²
    
    
    def logger_call(self):

        # Monitoring resources 
        if self.config['logging']['monitoring_resources'] and self.regular_iter == 0:

            self.log_log += "\n---------\n"

            # CPU / system-RAM
            self.log_log += get_cpu_memory_usage("") + "\n"

            # GPU (if any)
            if torch.cuda.is_available():
                cuda_device = torch.device("cuda")  # default current device
                self.log_log += get_cuda_memory_usage(cuda_device,"") + "\n"
            else:
                self.log_log += "No GPU detected\n"
        
        # Total and term losses
        loss_train = self.total_loss
        l2_error_u = self.l2_error_u
        
        
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
     
                
        # Save into a .txt file
        with open(self.config['train_progress_file_path'], 'w') as file:
            file.write(self.log_log)
        
        # Tensorboard
        self.writer.add_scalar('total_loss', self.total_loss, self.regular_iter)
        self.writer.add_scalar('l2_error_u', self.l2_error_u, self.regular_iter)
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
           
            model_weights = {
                'state_dict': self.model.mdona.state_dict(),
            }

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

    def random_sampling(self, xt, label = 'val'):
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
        
        xt_bc = {} 
        u_bc = {} 
        
       
        xt_val = {} 
        u_val = {}
      

        for chosen_flow_label in self.config['train']['training_param_label']:


            #---------------------
            # Fixed dataset
            #---------------------
            ## BOUNDARY CONDITIONS AND INITIAL CONDITION
            xt_bc[chosen_flow_label], u_bc[chosen_flow_label], xt_ic[chosen_flow_label], u_ic[chosen_flow_label]= craft_bc_and_ic_dataset(full_ds[chosen_flow_label], time_vector)
            

            ## VALIDATION DATASET
            xt_val[chosen_flow_label], u_val[chosen_flow_label] = craft_validation_dataset(val_ds[chosen_flow_label], time_vector)


        # """The coordinate points are the same between the different datasets, so we can choose a geometry among the options"""
        coordinates, time_min, time_max = get_coordinates_for_generator(full_ds[self.config['train']['training_param_label'][0]], time_vector)


        # Workflow
        self.custom_bar = trange(self.last_iter + 1)
        
         # Branch
        branch_input = {}
        for key in self.config['branches_control']['branch_input_ID']:
            branch_input.update({key:[]})

        ## Random Sampling - Get Fixed Indexes
        # BC 
        idx_bc_fixed = self.random_sampling(xt_bc[chosen_flow_label], label = 'bc')
        
        for chosen_flow_label in self.config['train']['training_param_label']:
                
            # INLET  
            u_bc_sample = u_bc[chosen_flow_label][idx_bc_fixed]    
            
            ## Branch
            # for i in range(N_vel_branch_inlet):
                # index = self.config['branches_control']['axis_indexes'][self.config['branches_control']['vel_axis_ID'][i]]
                # branch_input[self.config['branches_control']['branch_input_ID'][i]].append(vel_bc_inlet_sample[:,index].T)
            branch_input[self.config['branches_control']['branch_input_ID'][0]].append(u_bc_sample.T)
            
        

        for regular_iter in self.custom_bar:

            self.regular_iter = regular_iter
            
            # Trunk
            xt_ic_sample_all = []
            u_ic_target = []
            
            xt_bc_sample_all = []
            u_bc_target = []
           

           
            # Val
            xt_val_sample_all = []
            u_val_target = []
           

            ## Random Sampling - Get Indexes
            # IC 
            idx_ic = self.random_sampling(xt_ic[chosen_flow_label], label = 'ic')
            
            # BC 
            idx_bc = self.random_sampling(xt_bc[chosen_flow_label], label = 'bc')
            

            # VAL 
            idx_val = self.random_sampling(xt_val[chosen_flow_label], label = 'val')

            for chosen_flow_label in self.config['train']['training_param_label']:
                
                # IC 
                xt_ic_sample =  xt_ic[chosen_flow_label][idx_ic]
                u_ic_sample = u_ic[chosen_flow_label][idx_ic]
                
                # BC 
                xt_bc_sample =  xt_bc[chosen_flow_label][idx_bc]
                u_bc_sample = u_bc[chosen_flow_label][idx_bc]
                

                # VAL 
                xt_val_sample = xt_val[chosen_flow_label][idx_val]
                u_val_sample = u_val[chosen_flow_label][idx_val]

                ## Trunk
                xt_ic_sample_all.append(xt_ic_sample)
                u_ic_target.append(u_ic_sample)
                
                xt_bc_sample_all.append(xt_bc_sample)
                u_bc_target.append(u_bc_sample)

                xt_val_sample_all.append(xt_val_sample)
                u_val_target.append(u_val_sample)

            # Trunk
            xt_ic_sample_all = np.array(xt_ic_sample_all)
            u_ic_target = np.array(u_ic_target)
            
            xt_bc_sample_all = np.array(xt_bc_sample_all)
            u_bc_target = np.array(u_bc_target)


            xt_val_sample_all = np.array(xt_val_sample_all)
            u_val_target = np.array(u_val_target)
            
            # IC
            # N = 6, P = ? 
            self.xt_ic_tensor = []
            xt_ic_tensor = torch.tensor(np.swapaxes(xt_ic_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            self.xt_ic_tensor.append(xt_ic_tensor)

            self.f_ic_tensor = []
            num_samples = xt_ic_sample.shape[0]  # 51850
            target_device = self.config['device']

            for key in branch_input.keys():
                
                base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]),dtype=torch.float32,device=target_device)
                
                final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
                
                self.f_ic_tensor.append(final_view)
            
            self.u_ic_tensor = torch.tensor(np.swapaxes(u_ic_target,0,1).reshape(-1,self.output_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
            # BC
            self.xt_bc_tensor = []
            xt_bc_tensor = torch.tensor(np.swapaxes(xt_bc_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            self.xt_bc_tensor.append(xt_bc_tensor)

            self.f_bc_tensor = []
            num_samples = xt_bc_sample.shape[0]  
            target_device = self.config['device']
            
            for key in branch_input.keys():
                
                base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]),dtype=torch.float32,device=target_device)
                
                final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
                
                self.f_bc_tensor.append(final_view)
            
            self.u_bc_tensor = torch.tensor(np.swapaxes(u_bc_target,0,1).reshape(-1,self.output_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            
           
           
            # Validation dataset
            # N = 6, P = xyzt_val_fixed.shape[0]
            self.xt_val_tensor = []
            xt_val_tensor = torch.tensor(np.swapaxes(xt_val_sample_all,0,1).reshape(-1,self.trunk_in_dim), dtype = torch.float32, requires_grad= False).to(self.config['device'])
            self.xt_val_tensor.append(xt_val_tensor)
            
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
            

            with torch.no_grad():
                u_pred = self.model.mdona(self.f_val_tensor[0], self.xt_val_tensor)
               
            
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
   
   
class modified_Tester(torch.nn.Module):

    def __init__(self, config, model):
        
        super().__init__()
        
        self.config = config
        self.model = model

        self.branch_in_dim = self.config['branch1']['neuralNet']['in_dim']
        self.branch_out_dim = self.config['branch1']['neuralNet']['out_dim']
         
    def test_full(self, dataloader, N_batches, branch_input, subset = 'test', param_label = None):

        history_log = f"""LOG TEST ON {param_label} - {subset}_dataset\n"""
       
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
            
                u_pred = self.model.mdona(f_tensor[0], inputs)


            l2_relative_error_u.append(metric_l2_relative_error(exact = u_ref, pred = u_pred).cpu().numpy())

            
            # Save into a .txt file
            history_log = history_log + \
            "\n-------------------------\n" + \
            f"""Batch: {(batch_iter + 1)}/{(N_batches)} - 'l2_relative_error_vel': {l2_relative_error_u[batch_iter]} \n"""
         
            
        # Save into a .txt file
        with open(self.config['test_progress_file_path'], 'a') as file:
            file.write(history_log + '\n\n\n')

        self.config['logger'].info(f"Accuracy in {param_label} - {subset}_dataset")
        self.config['logger'].info(f"L2 relative error in vel: {np.mean(np.array(l2_relative_error_u))}")
        
        print(f"Accuracy in {param_label} - {subset}dataset")
        print(f"L2 relative error in vel: {np.mean(np.array(l2_relative_error_u))}")

    def test_value(self, k_i, t_i, c, branch_input, label = 'InflowBC_K056'):
        

        # Exact
        x_array = np.linspace(0, 1, 1000)[:,None]
        phi_i = calculate_phi(k_i)
        u_exact = u_close_form(x_array, t_i, c, k_i, phi_i)
        
        u_exact_tensor = torch.tensor(u_exact, dtype=torch.float32, requires_grad = False).to(self.config['device'])

    
        # N = 1, P = batch_size
        t_array = np.repeat(np.array([t_i]), x_array.shape[0], axis=0)[:,None]

        xt_array = np.hstack([x_array, t_array])

        xt_tensor = torch.tensor(xt_array, dtype = torch.float32, requires_grad= False).to(self.config['device'])
        
        
        f_tensor = []
        num_samples = xt_tensor.shape[0]
        target_device = self.config['device']

        for key in branch_input.keys():
                
            base_tensor_dev = torch.as_tensor(np.vstack(branch_input[key]),dtype=torch.float32,device=target_device)
                
            final_view = base_tensor_dev.contiguous().unsqueeze(0).expand(num_samples, -1, -1) 
                
            f_tensor.append(final_view)
        
        
        
        

        with torch.no_grad():
        #     tau = self.model.branch_list[0](f_a_tensor)
        #     beta = self.model.trunk(xt_tensor)
        # u_pred_tensor = torch.sum(beta * tau, axis = 1)[:,None]
            u_pred_tensor = self.model.mdona(f_tensor[0], xt_tensor)

        
        l2_relative_error = torch.linalg.norm((u_exact_tensor-u_pred_tensor), 2)/torch.linalg.norm(u_exact_tensor, 2)
        print(f'l2_relative_error={l2_relative_error.item()} for t = {t_i} and {label}')

        u_pred = u_pred_tensor.cpu().numpy()
        

        return x_array, u_exact, u_pred, l2_relative_error.item()
    
    def visualize_comparison_per_value(self, t_all, f_a_i, x_final, u_exact_final, u_pred_final, label = 'InflowBC_K056'):
       

        fig, ax = plt.subplots(figsize=(5, 5))

        # Create a color gradient for lines and points
        line_colors_exact = plt.cm.Blues(np.linspace(0.5, 1, len(t_all)))
        line_colors_pred = plt.cm.Reds(np.linspace(0.5, 1, len(t_all)))
        point_colors = plt.cm.Greens(np.linspace(0.5, 1, len(t_all)))

        for i,value in enumerate(t_all):
            
            ax.plot(x_final[i], u_exact_final[i], color=line_colors_exact[i], linewidth=3, label=f't = {value} and {f_a_i}')
            ax.plot(x_final[i], u_pred_final[i], '--',color=line_colors_pred[i], linewidth=3)
           

        # Setting up the plot
        ax.set_xlabel('x',fontsize=18)
        ax.set_ylabel('u',fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax.set_xlim(0, 1)
        # ax.set_ylim(0.8, 4)
        ax.set_ylim(-0.5, 1.5)
        # ax.set_ylim(0.4, 2.5)
        ax.legend()
        # ax.set_title('Exact Solution for Different a Values')

        # plt.show()

        # Save the figure
        fig_path = self.config['charts_folder_path'].joinpath(f'comparison_exact_vs_predicted_{label}.png')
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    
    def visualize_comparison_fulldomain(self, branch_input, k_i, c, label):
        

        # ------------------------------------------------------------------
        # 1. Create a *regular space-time grid* with a single, unambiguous rule:
        #    - first axis  →  time  (Nt points)
        #    - second axis →  space (Nx points)
        # ------------------------------------------------------------------
        t_star = np.linspace(0.0, 1.0, 100)      # Nt
        x_star = np.linspace(0.0, 1.0, 1000)     # Nx

        TT, XX = np.meshgrid(t_star, x_star, indexing="ij")  # TT,XX ∈ ℝ[Nt,Nx]

        t_flat = TT.reshape(-1, 1)               # ℝ[Nt·Nx,1]
        x_flat = XX.reshape(-1, 1)

        # ------------------------------------------------------------------
        # 2. Exact solution on the same grid
        # ------------------------------------------------------------------
        phi_i = calculate_phi(k_i) 
        u_ref = u_close_form(x_flat, t_flat, c, k_i, phi_i)  # expects (x,t,c)

        # ------------------------------------------------------------------
        # 3. Convert to torch & push through the network  (xt order)
        # ------------------------------------------------------------------
        device = self.config["device"]

        t = torch.as_tensor(t_flat, dtype=torch.float32, device=device)
        x = torch.as_tensor(x_flat, dtype=torch.float32, device=device)

        # >>> trunk expects (x , t)  <<<  so concatenate in that order
        xt = torch.cat([x, t], dim=1)        # shape: (N, 2)

        # == Branch inputs (unchanged) =====================================
        f_tensor = []
        num_samples = xt.shape[0]
        for i, key in enumerate(branch_input.keys()):
            base = torch.as_tensor(np.vstack(branch_input[key]),
                                dtype=torch.float32,
                                device=device)
            f_tensor.append(base.unsqueeze(0).expand(num_samples, -1, -1))

        # == PINN prediction ===============================================
        with torch.no_grad():
            # tau = [self.model.branch_list[i](f_tensor_i) for i, f_tensor_i in enumerate(f_tensor)]
            # beta = self.model.trunk(xt)      # <<<<<<<< uses xt
            # tau[0] = tau[0].view(-1, self.branch_out_dim)
            # u_pred = (tau[0] * beta).sum(dim=1, keepdim=True)

            u_pred = self.model.mdona(f_tensor[0], xt)
        # ------------------------------------------------------------------
        # 4. Error metrics
        # ------------------------------------------------------------------
        l2_relative_error = metric_l2_relative_error(
            exact=torch.as_tensor(u_ref, device=device, dtype=torch.float32),
            pred=u_pred
        )

        self.config['logger'].info(f'l2_relative_error={l2_relative_error} for {label}')
        print(f'l2_relative_error={l2_relative_error} for {label}')

        # ------------------------------------------------------------------
        # 5. Reshape back to (Nt,Nx) – NO transposes, the axes are correct
        # ------------------------------------------------------------------
        Nt, Nx = TT.shape
        u_pred = u_pred.cpu().numpy().reshape(Nt, Nx)
        u_ref  = u_ref.reshape(Nt, Nx)
        abs_err = np.abs(u_ref - u_pred)

        # ------------------------------------------------------------------
        # 6. Plot
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

        titles = [f"Exact u(t,x) ({label})", "Predicted u(t,x)", "Absolute error"]
        data   = [u_ref, u_pred, abs_err]

        for ax, z, title in zip(axes, data, titles):
            pcm = ax.pcolormesh(t_star, x_star, z.T, cmap="jet", shading="auto")
            fig.colorbar(pcm, ax=ax)
            ax.set_xlabel("t")
            ax.set_ylabel("x")
            ax.set_title(title)

        # ------------------------------------------------------------------
        # 7. Save & close
        # ------------------------------------------------------------------
        fig_path = self.config["charts_folder_path"] / f"comparison_fulldomain_{label}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # (optional) return the relative L2 error for logging
        return float(l2_relative_error)


            

