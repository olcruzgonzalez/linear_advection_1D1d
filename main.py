import sys
import os
import logging
import pickle 
import argparse
import time, shutil

from pathlib import Path
from ruamel.yaml import YAML

import numpy as np
import torch


# Local modules
from generate_dataset.generate import generate_call
from train import training_call
from test import testing_call
from utils import processing_call, named_tuple_definition, initialize_mode, run_command


def run(config):
    
    if config['mode'] == 'train': training_call(config)
    else: testing_call(config)

if __name__ == "__main__":

    tick_start  = time.time()

    # Setting parameters from the command line
    parser = argparse.ArgumentParser(description="Execution Mode")
    parser.add_argument("--mode", type=str, default='train', help="Execution modes are 'train' or 'test'.")
    parser.add_argument("--method", type=str, default='pi_deeponet_Ldata', help="Available methods are  'pi_deeponet_Ldata', 'pi_deeponet'.")
    parser.add_argument("--run_ID", type=int, default=-1, help="The 'run_ID' is a unique identifier for the calculation.")
    parser.add_argument("--input_file_name", type=str, default='input.yaml', help="The input file contains the parameters necessary to launch the calculations. Here, we use YAML format.")
    parser.add_argument("--relative_path", type=str, default='./input.yaml', help="This is the relative path to the YAML input file. Only applies when run_ID == -1.")
    args = parser.parse_args()
    
    print(f'### {args.mode.upper()} MODE ###')
    config = {} 
    config = initialize_mode(config, mode = args.mode, method = args.method, run_ID = args.run_ID)
    
    # Loading input.yaml file
    config['logger'].info("Loading input.yaml file")
    if args.run_ID == -1: 
        config['input_file_name'] = args.input_file_name
        config['relative_path_to_input_file'] = args.relative_path
        config['input_file_path'] = config['cwd'].joinpath(config['relative_path_to_input_file'])
    else: 
        config['input_file_path'] = config['cwd'].joinpath(f'results/run-ID{args.run_ID}/files/{args.input_file_name}')

    yaml = YAML(typ='rt')  
    with open(config['input_file_path'], 'r') as f:
        config.update(yaml.load(f))
    
    # Dataset
    config['data_path'] = config['cwd'].joinpath(config['dataset']['path'])
    
    if args.run_ID == -1:
       
        # Save input.yaml file
        source = config['input_file_path']
        destination = config['files_folder_path'].joinpath(config['input_file_name'])
        shutil.copy2(source, destination)

        # Exporting env requirements
        try:
            config['logger'].info(f"Exporting pip requirements")
            run_command(f"pip freeze > {config['files_folder_path'].joinpath('pip_requirements.txt')}")
        except:
            config['logger'].info(f"Unable to export pip requirements")
        try:
            config['logger'].info(f"Exporting conda requirements")
            run_command(f"conda list --export > {config['files_folder_path'].joinpath('conda_requirements.txt')}")
        except:
            config['logger'].info(f"Unable to export conda requirements")
    
    #--------------------------
    # Mimic dataset coming from simulations !
    # Generate Dataset - Academic problems
    if config['dataset']['generate']:
        print("### Generating dataset...")
        generate_call(config)
    #--------------------------

    # NamedTuples
    Collection = named_tuple_definition(config['dataset']['strata_labels'], config['dataset']['state_vector_labels'])

    config['logger'].info(f"Setting seed={config['seed']} for Pytorch and numpy reproducibility")
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.determinist = True

    if config['dataset']['processing']:
        print("### Processing dataset...")
        processing_call(config, Collection)
    
    config['mode'] = args.mode
    config['method'] = args.method
    config['tick_start'] = tick_start

    run(config)
