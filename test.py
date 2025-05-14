import sys
import os
from pathlib import Path
import time
import json 


import torch
import numpy as np

from collections import namedtuple


# Local modules
from models import NeuralNetwork, Tester
from utils import convert_to_json, get_subset_ds_deeponet, create_dataset_for_test, build_stratum_dataset



def testing_call(config):

     # Recreate the model instance
    config['logger'].info("Recreate the model instance")
    model = NeuralNetwork(config = config)
    
    # Recover the best model - Load weights
    config['logger'].info("Load model weights")
    load_path_weights = config['checkpoints_folder_path'].joinpath(f"model_weights.pt")
    weights = torch.load(load_path_weights)
    # checkpoint_path = config['checkpoints_folder_path'].joinpath(f"model_weights.pt")
    # checkpoint = torch.load(checkpoint_path)

    # weights = checkpoint['mlp_state_dict']

    # load_path_idx_inlet = config['checkpoints_folder_path'].joinpath(f"idx_inlet.npy")
    # idx_inlet = np.load(load_path_idx_inlet)
    # idx_outlet = np.load(load_path_outlet_outlet)

    # Recover the best model - Update weights
    config['logger'].info("Update model weights")
    for i, key in enumerate(config['branches_control']['branch_list_ID']):
        model.branch_list[i].load_state_dict(weights[f'{key}_state_dict'])
    model.trunk.load_state_dict(weights['trunk_state_dict'])

    model.to(config['device'])

    for idx, ID in enumerate(config['test']['test_param_label']):

        # Load data
        timePrm, full_ds = get_subset_ds_deeponet(config, ID, 'full')
        # timePrm, max_inlet_velocity, full_ds = get_subset_ds_deeponet(config, ID, 'temporaltest')

        # # Update parameters based on the max inlet velocity
        # config['pde_param']['nu_f'] = config['pde_param']['mu_f']/config['pde_param']['rho_f']
        # config['pde_param']['omega'] = 2*np.pi/config['pde_param']['T']
        # config['pde_param']['V_inlet_max'] = max_inlet_velocity
        # config['logger'].info(f"Maximum inlet velocity magnitude: {max_inlet_velocity}")
        # # config['pde_param']['V_inlet_max'] = config[mode]['vel_max_inlet']
        # config['pde_param']['Re'] = config['pde_param']['2R_pipe']*config['pde_param']['V_inlet_max']/config['pde_param']['nu_f'] 
        # config['pde_param']['Wo'] = np.sqrt(config['pde_param']['omega']*config['pde_param']['2R_pipe']**2/config['pde_param']['nu_f']) 


        config['logger'].info("Create dataset for test")
        dataloader_full, N_batches_full = create_dataset_for_test(config, full_ds, timePrm)


        # N_vel_branch_inlet = len(config['branches_control']['vel_axis_ID'])
        # has_outlet_pressure = len(config['branches_control']['branch_input_ID']) > len(config['branches_control']['vel_axis_ID'])



        # Branch
        branch_input = {}
        for key in config['branches_control']['branch_input_ID']:
                branch_input.update({key:[]})
        
        
        
        # # INLET
        # _, v_inlet, _ =  build_stratum_dataset(full_ds.INLET, timePrm.time_vector[:,None])

        # # OUTLET
        # _, _, pressure_outlet =  build_stratum_dataset(full_ds.OUTLET, timePrm.time_vector[:,None])
        # # Data transformation: From dimension to dimensionless
        # v_inlet = v_inlet/config['pde_param']['V_inlet_max']
        # pressure_outlet = pressure_outlet/(config['pde_param']['rho_f'] * config['pde_param']['V_inlet_max']**2)
       
         ## Branch
        # for i in range(N_vel_branch_inlet):
            # index = config['branches_control']['axis_indexes'][config['branches_control']['vel_axis_ID'][i]]
            # branch_input[config['branches_control']['branch_input_ID'][i]].append(v_inlet[idx_inlet,index].T)
            # branch_input[config['branches_control']['branch_input_ID'][i]].append(v_inlet[:,index].T)
        branch_input[config['branches_control']['branch_input_ID'][0]].append(np.repeat(np.array([config['test']['test_param'][idx]]),90)[None,:])
        
        
        config['logger'].info("### Testing ...")
        tester = Tester(config, model)
        
        # Testing
        # ID = config['dataset']['chosen_flow_label']
        config['logger'].info("#-----------------------------------------#")
        config['logger'].info(f"TESTING ON {ID} - 'full'")
        config['logger'].info("#-----------------------------------------#")
        tester.test(dataloader_full, N_batches_full, branch_input, subset = 'full')
        
        config['logger'].info("#-----------------------------------------#")
        config['logger'].info(f"TESTING ON {ID} - 'temporaltest'")
        config['logger'].info("#-----------------------------------------#")
        # tester.test(dataloader_temporaltest, N_batches_temporaltest, subset = 'temporaltest')

        # config['logger'].info("#-----------------------------------------#")
        # config['logger'].info(f"TESTING ON {ID} - 'full'")
        # config['logger'].info("#-----------------------------------------#")
        # tester.test(dataloader_full, N_batches_full, subset = 'full')

        # # #--------------------------
        # # config['logger'].info("#-----------------------------------------#")
        # # tester.visualize_absolute_error(ID, branch_input)
        
        # # #--------------------------
        # # config['logger'].info("#-----------------------------------------#")
        # t_values = config[config['mode']]['t_values']
        # y_values = config[config['mode']]['y_values_for_cross_sections']
        # for t_value in t_values:
        #     for y_value in y_values:
        #         tester.visualize_absolute_error_in_cross_sections(full_ds, t0= t_value, y0 = y_value, subset = 'full')
        
        # #--------------------------
        # config['logger'].info("#-----------------------------------------#")
        # z_value = config[config['mode']]['z_value_for_longitudinal_section']
        # for t_value in t_values:
        #     tester.visualize_absolute_error_in_longitudinal_section(full_ds, t0 = t_value, z0 = z_value, subset = 'full')
        
        
        #--------------------------
        config['logger'].info("#-----------------------------------------#")
        runtime_final = (time.time() - config['tick_start'])/60
        config['runtime_final_in_min_for_test'] = runtime_final

        config['logger'].info(f'### EXECUTION TIME ### = {runtime_final} min')
        config['logger'].info(f'### END ###')

        method = config['method']
        config_file = f'config_test_{method}.json'
        config_file_path = config['files_folder_path'].joinpath(config_file)
        with open(config_file_path, 'w') as f: json.dump(config, f, indent=4, default=convert_to_json)

    #---
    config['logger'].info("#-----------------------------------------#")
    runtime_final = (time.time() - config['tick_start'])/60

    config['logger'].info(f'### EXECUTION TIME - TEST ### = {runtime_final} min')
    config['logger'].info(f'### END ###')
    print(f'### EXECUTION TIME - TEST ### = {runtime_final} min')
    print(f'### END ###')