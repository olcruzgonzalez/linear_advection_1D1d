import sys
import os
from pathlib import Path
import time
import json 

import torch
import numpy as np

# Local modules
from models import NeuralNetwork, modified_NeuralNetwork, Trainer_PIDeepONetLdata, Trainer_PIDeepONet, Trainer_PIDeepONetLdata_modified, Trainer_PIDeepONet_modified
from utils import convert_to_json, all_data_transformations_deeponet, random_data_extraction_deeponet


def training_call(config):

    # Data transformations
    timePrm, full_ds = all_data_transformations_deeponet(config, subset = 'full')  # Only full
    timePrm, val_ds = all_data_transformations_deeponet(config, subset = 'val')

    # Extract data for the Loss_data
    if  config['train']['data_extraction']['mode'] == 'random':
        
        dataPrm, config['data_percent'] = random_data_extraction_deeponet(config, full_ds)
    
    else: pass

  
    # Training
    if config['method'] == 'pi_deeponet_Ldata':

        config['logger'].info("Model definition")
        model = NeuralNetwork(config = config)
        model.to(config['device'])

        config['logger'].info("### Training ...")
        trainer = Trainer_PIDeepONetLdata(config, model)
        trainer.train(timePrm, full_ds, dataPrm, val_ds)
    

    elif config['method'] == 'pi_deeponet':
        
        config['logger'].info("Model definition")
        model = NeuralNetwork(config = config)
        model.to(config['device'])
        
        config['logger'].info("### Training ...")
        trainer = Trainer_PIDeepONet(config, model)
        trainer.train(timePrm, full_ds, val_ds)

    
    elif config['method'] == 'modified_pi_deeponet_Ldata':

        config['logger'].info("Model definition")
        model = modified_NeuralNetwork(config = config)
        model.to(config['device'])

        config['logger'].info("### Training ...")
        trainer = Trainer_PIDeepONetLdata_modified(config, model)
        trainer.train(timePrm, full_ds, dataPrm, val_ds)

    
    elif config['method'] == 'modified_pi_deeponet':

        config['logger'].info("Model definition")
        model = modified_NeuralNetwork(config = config)
        model.to(config['device'])
        
        config['logger'].info("### Training ...")
        trainer = Trainer_PIDeepONet_modified(config, model)
        trainer.train(timePrm, full_ds, val_ds)

    
    else: pass


    runtime_final = (time.time() - config['tick_start'])/60
    config['runtime_final_in_min_for_train'] = runtime_final
    
    config['logger'].info(f'### EXECUTION TIME ### = {runtime_final} min')
    config['logger'].info(f'### END ###')

    method = config['method']
    config_file = f'config_train_{method}.json'
    config_file_path = config['files_folder_path'].joinpath(config_file)
    with open(config_file_path, 'w') as f: json.dump(config, f, indent=4, default=convert_to_json)


    print(f'### EXECUTION TIME ### = {runtime_final} min')
    print(f'### END ###')

