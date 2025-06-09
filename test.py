import sys
import os
from pathlib import Path
import time
import json 


import torch
import numpy as np

from collections import namedtuple


# Local modules
from models import NeuralNetwork, modified_NeuralNetwork, Tester, modified_Tester
from utils import convert_to_json, get_subset_ds_deeponet, create_dataset_for_test, build_stratum_dataset



def testing_call(config):

    # Recreate the model instance
    config['logger'].info("Recreate the model instance")
    if config['method'] == 'modified_pi_deeponet_Ldata' or config['method'] == 'modified_pi_deeponet':
        model = modified_NeuralNetwork(config = config)
    else:
        model = NeuralNetwork(config = config)
    
    # Recover the best model - Load weights
    config['logger'].info("Load model weights")
    load_path_weights = config['checkpoints_folder_path'].joinpath(f"model_weights.pt")
    weights = torch.load(load_path_weights)
    

    # Recover the best model - Update weights
    config['logger'].info("Update model weights")
    if config['method'] == 'modified_pi_deeponet_Ldata' or config['method'] == 'modified_pi_deeponet':
        model.mdona.load_state_dict(weights['state_dict'])
    else:
        for i, key in enumerate(config['branches_control']['branch_list_ID']):
            model.branch_list[i].load_state_dict(weights[f'{key}_state_dict'])
        for i, key in enumerate(config['trunk_control']['trunk_list_ID']):
            model.trunk_list[i].load_state_dict(weights[f'{key}_state_dict'])

    model.to(config['device'])
    c = config['dataset']['c']

    # load_path_idx_ic = config['checkpoints_folder_path'].joinpath(f"idx_ic.npy")
    # idx_ic = np.load(load_path_idx_ic)

    for idx, ID in enumerate(config['test']['test_param_label']):

        # Load data
        config['logger'].info("Loading dataset - full")
        timePrm, full_ds = get_subset_ds_deeponet(config, ID, 'full')

        config['logger'].info("Create dataset for test - dataloader")
        dataloader_full, N_batches_full = create_dataset_for_test(config, full_ds, timePrm)

        # Branch
        branch_input = {}
        for key in config['branches_control']['branch_input_ID']:
                branch_input.update({key:[]})
        
        # BC
        _, u_bc = build_stratum_dataset(full_ds.BC, timePrm.time_vector[:,None], label = 'BC')
        # PHY
        xt_phy, u_phy = build_stratum_dataset(full_ds.PHY, timePrm.time_vector[:,None], label = 'PHY')
        
        ## IC - PHY
        idx_aux = np.where(xt_phy[:,1] == 0)
        # xt_ic = xt_phy[idx]
        u_ic = u_phy[idx_aux]

        # IC
        # u_ic_sample = u_ic[idx_ic]

        branch_input[config['branches_control']['branch_input_ID'][0]].append(u_bc.T)
        # branch_input[config['branches_control']['branch_input_ID'][1]].append(u_ic.T)

        print("### Testing ... ###")
        config['logger'].info("### Testing ...")
        if config['method'] == 'modified_pi_deeponet_Ldata' or config['method'] == 'modified_pi_deeponet':
            tester = modified_Tester(config, model)
        else:
            tester = Tester(config, model)
        
        config['logger'].info("#-----------------------------------------#")
        config['logger'].info("#-----------------------------------------#")
        config['logger'].info(f"TESTING ON {ID} - 'full'")
        config['logger'].info("#-----------------------------------------#")
        config['logger'].info("#-----------------------------------------#")
        tester.test_full(dataloader_full, N_batches_full, branch_input, subset = 'full', param_label = ID)
        
    
        x_array_final = []
        u_exact_final = []
        u_pred_final = []
       
        for t_i in config['test']['t_values']:

            k_i = config['test']['test_param'][idx]
            config['logger'].info(f"TESTING ON {ID} - value {t_i}")
            x_array, u_exact, u_pred, l2_relative_error = tester.test_value(k_i, t_i, c, branch_input, label = ID)
            
            config['logger'].info(f'l2_relative_error={l2_relative_error} for t = {t_i} and {ID}')
            
            x_array_final.append(x_array)
            u_exact_final.append(u_exact)
            u_pred_final.append(u_pred)
            
        
        tester.visualize_comparison_per_value(config['test']['t_values'], ID, x_array_final, u_exact_final, u_pred_final, label = ID)
        
        print("### Visualize comparison full domain ###")
        tester.visualize_comparison_fulldomain(branch_input, k_i, c, ID)
        
        

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