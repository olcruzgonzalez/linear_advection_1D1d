from pathlib import Path

import numpy as np
from tqdm import trange

from generate_dataset.ground_trust import governing_equation, boundary_condition


def generate_call(config):
    
    cwd = config['cwd']
    L = config['dataset']['L']
    c = config['dataset']['c']
    varying_param = config['dataset']['varying_param'] # parameter k
    varying_param_label = config['dataset']['varying_param_label']
    strata_ID = config['dataset']['strata_labels']
    N_samples_per_stratum = config['dataset']['N_samples_per_stratum']
    N_time_step = config['dataset']['N_time_step']
    initial_time = config['dataset']['initial_time']
    total_time = config['dataset']['total_time']
    time_step = config['dataset']['time_step']
    seed = config['seed']


    np.random.seed(seed)
    N_coll = N_samples_per_stratum['N_coll']
    N = len(varying_param_label)
    pbar = trange(N)
    time_vector = np.linspace(initial_time, total_time, N_time_step)
    

    for i in pbar:

        # Folder Structure
        dataset_folder = f".dataset/STRATA_{varying_param_label[i]}"
        results_folder_path = cwd.joinpath(dataset_folder)
        results_folder_path.mkdir(parents=True, exist_ok=True)
        
        for stratum in strata_ID: 
            dataset_folder = f".dataset/STRATA_{varying_param_label[i]}/{stratum}"
            results_folder_path = cwd.joinpath(dataset_folder)
            results_folder_path.mkdir(parents=True, exist_ok=True)
            
            for t_i in time_vector:
              
                if stratum == 'PHY':
                    governing_equation(cwd, L, c, varying_param[i], varying_param_label[i], stratum, N_coll, t_i)
                elif stratum == 'BC':
                    boundary_condition(cwd, varying_param[i], varying_param_label[i], stratum, t_i)

    print("### Dataset successfully generated!")
        
        
    
    