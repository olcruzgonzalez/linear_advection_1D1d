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
    

    # Recover the best model - Update weights
    config['logger'].info("Update model weights")
    for i, key in enumerate(config['branches_control']['branch_list_ID']):
        model.branch_list[i].load_state_dict(weights[f'{key}_state_dict'])
    model.trunk.load_state_dict(weights['trunk_state_dict'])

    model.to(config['device'])

    for idx, ID in enumerate(config['test']['test_param_label']):

        # Load data
        config['logger'].info("Loading dataset - full")
        timePrm, full_ds = get_subset_ds_deeponet(config, ID, 'full')
        # timePrm, max_inlet_velocity, full_ds = get_subset_ds_deeponet(config, ID, 'temporaltest')

        config['logger'].info("Create dataset for test - dataloader")
        dataloader_full, N_batches_full = create_dataset_for_test(config, full_ds, timePrm)

        # Branch
        branch_input = {}
        for key in config['branches_control']['branch_input_ID']:
                branch_input.update({key:[]})
        
        
        branch_input[config['branches_control']['branch_input_ID'][0]].append(np.repeat(np.array([config['test']['test_param'][idx]]),90)[None,:])
        
        print("### Testing ... ###")
        config['logger'].info("### Testing ...")
        tester = Tester(config, model)
        
        # Testing
        # ID = config['dataset']['chosen_flow_label']
        config['logger'].info("#-----------------------------------------#")
        config['logger'].info("#-----------------------------------------#")
        config['logger'].info(f"TESTING ON {ID} - 'full'")
        config['logger'].info("#-----------------------------------------#")
        config['logger'].info("#-----------------------------------------#")
        tester.test_full(dataloader_full, N_batches_full, branch_input, subset = 'full', param_label = ID)
        
        config['logger'].info("#-----------------------------------------#")
        config['logger'].info(f"TESTING ON {ID} - 'temporaltest'")
        config['logger'].info("#-----------------------------------------#")
        # tester.test(dataloader_temporaltest, N_batches_temporaltest, subset = 'temporaltest')

        
        
        x_array_final = []
        u_exact_final = []
        u_pred_final = []
       
        for t_i in config['test']['t_values']:

            c = config['test']['test_param'][idx]
            config['logger'].info(f"TESTING ON {ID} - value {t_i}")
            x_array, u_exact, u_pred, l2_relative_error = tester.test_value(t_i, c)
            
            config['logger'].info(f'l2_relative_error={l2_relative_error} for t = {t_i} and c = {c}')
            
            x_array_final.append(x_array)
            u_exact_final.append(u_exact)
            u_pred_final.append(u_pred)
            
        
        tester.visualize_comparison_per_value(config['test']['t_values'], c, x_array_final, u_exact_final, u_pred_final)
        
        print("### Visualize comparison full domain ###")
        tester.visualize_comparison_fulldomain(branch_input, c)
        
        

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