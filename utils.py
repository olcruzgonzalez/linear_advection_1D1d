import sys
import os
import subprocess

from pathlib import Path
import pathlib

import random
import datetime
import time
import logging

import psutil
import GPUtil

# import pickle
import dill as pickle
import copy

import torch
import scipy.io
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


plt.ioff()  # Turn interactive plotting off

from collections import namedtuple
from torch.utils.data import Dataset, DataLoader



# General purpose

def get_cpu_memory_usage(description: str = "") -> str:
    mem = psutil.virtual_memory()

    total_gib = mem.total / 1024 ** 3
    used_gib = mem.used / 1024 ** 3
    avail_gib = mem.available / 1024 ** 3
    percent_used = mem.percent

    msg = (
        f"CPU-RAM{f' ({description})' if description else ''}: "
        f"Used {used_gib:.2f} GiB / Total {total_gib:.2f} GiB | "
        f"Usage {percent_used:.1f}% | "
        f"Avail {avail_gib:.2f} GiB"
    )
    # print(msg)
    return msg

def get_cuda_memory_usage(device: torch.device, description: str = "") -> str:
    if device.type != "cuda":
        msg = f"Device {device} is not CUDA."
        print(msg)
        return msg

    allocated_gib = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved_gib = torch.cuda.memory_reserved(device) / 1024 ** 3
    total_gib = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

    # “Available” = total physical capacity minus memory reserved by the CUDA caching allocator
    avail_gib = total_gib - reserved_gib
    percent_used = (allocated_gib / total_gib) * 100 if total_gib else 0.0

    msg = (
        f"CUDA{f' ({description})' if description else ''} on {device}: "
        f"Used {allocated_gib:.2f} GiB / Total {total_gib:.2f} GiB | "
        f"Usage {percent_used:.1f}% | "
        f"Avail {avail_gib:.2f} GiB | "
        f"Reserv {reserved_gib:.2f} GiB"
    )
    # print(msg)
    return msg

def run_command(command):
    """Execute shell commands and capture output.
    """
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        raise Exception("Error executing command '{}': {}".format(command, result.stderr))
    
def convert_to_json(item):
    """Convert items from a python dictionary to JSON serializable.
    """
    if isinstance(item, Path):
        return str(item)
    elif isinstance(item, logging.Logger):
        return item.name
    elif isinstance(item, (list, tuple)):
        return [convert_to_json(item) for item in item]
    elif isinstance(item, torch.device):
        return str(item)
    elif isinstance(item, (np.ndarray, np.generic)):
        return str(item)
    else:
        raise TypeError(f"Item of type {type(item)} is not JSON serializable")

def create_folder_structure(cwd, method=None, run_ID=None):
    """Create the structure of results folder.
    """

    # (1) Folder
    results_folder = "results"
    results_folder_path = cwd.joinpath(results_folder)
    results_folder_path.mkdir(parents=True, exist_ok=True)
    print(f"Folder '{results_folder}' has been created in the current working directory.")

    # |_(A) Subfolder
    now = datetime.datetime.now()
    if run_ID == -1: 
        run_ID = now.strftime("%Y%m%d%H%M%S")

        #---
        # Create file 'run_ID_log.txt'
        log_file_path = results_folder_path.joinpath('run_ID_log.txt')

        # Create the file if it doesn't exist
        if not log_file_path.exists():
            log_file_path.touch()

        print(f"Log file created at: {log_file_path}.")

        # Add ID to the log file
        log_entry = run_ID

        with open(log_file_path, 'a') as file:
            file.write(log_entry + '\n')

        print(f'Log entry added to: {log_file_path}')
        #---

    else: pass
    
    

    run_ID_folder = f"run-ID{run_ID}"
    run_ID_path = results_folder_path.joinpath(run_ID_folder)
    run_ID_path.mkdir(parents=True, exist_ok=True)
    print(f"Subfolder '{run_ID_folder}' has been created inside '{results_folder}'.")
    
    # |_(a) 2xSubFolders
    method_folder = f"method-{method}"
    method_folder_path = run_ID_path.joinpath(method_folder)
    method_folder_path.mkdir(parents=True, exist_ok=True)
    print(f"2xSubfolder '{method_folder}' has been created inside '{run_ID_folder}'.")

    # |_(b) 2xSubFolders
    files_folder = "files"
    files_folder_path = run_ID_path.joinpath(files_folder)
    files_folder_path.mkdir(parents=True, exist_ok=True)
    print(f"3xSubfolder '{files_folder}' has been created inside '{run_ID_folder}'.")
    
    # |_(i) 3xSubFolders
    checkpoints_folder = "checkpoints"
    logs_folder = "logs"
    charts_folder = "charts"
   
    checkpoints_folder_path = method_folder_path.joinpath(checkpoints_folder)
    logs_folder_path = method_folder_path.joinpath(logs_folder)
    charts_folder_path = method_folder_path.joinpath(charts_folder)

    checkpoints_folder_path.mkdir(parents=True, exist_ok=True)
    print(f"3xSubfolder '{checkpoints_folder}' has been created inside '{method_folder}'.")
    
    logs_folder_path.mkdir(parents=True, exist_ok=True)
    print(f"3xSubfolder '{logs_folder}' has been created inside '{method_folder}'.")
    
    charts_folder_path.mkdir(parents=True, exist_ok=True)
    print(f"3xSubfolder '{charts_folder}' has been created inside '{method_folder}'.")


    return results_folder_path, run_ID_path, method_folder_path, checkpoints_folder_path, files_folder_path, logs_folder_path, charts_folder_path, run_ID
   
def check_for_gpus(logger):
    """Check if cuda is available for Pytorch computations.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    logger.info(f"Running on device:{device}!")
    return device

def initialize_mode(config, mode, method, run_ID):
    
    # Transfer code usage from Linux to Windows
    # temp = pathlib.PosixPath
    # pathlib.PosixPath = pathlib.WindowsPath
    
    # Project directory.
    config['cwd'] = Path(os.getcwd()) 

    print('### Building necessary folders ... ###')
    config['results_folder_path'], config['run_ID_path'], config['method_folder_path'], config['checkpoints_folder_path'], config['files_folder_path'],\
            config['logs_folder_path'], config['charts_folder_path'], config['run_ID'] = create_folder_structure(config['cwd'], method, run_ID)

    # Log
    config[f'log_file_in_{mode}'] = f'verbose_{mode}.log'
    config[f'log_file_path_in_{mode}'] = config['logs_folder_path'].joinpath(config[f'log_file_in_{mode}'])
    logging.basicConfig(
        format='%(levelname)s: %(filename)s, line %(lineno)d -> %(message)s', filename = config[f'log_file_path_in_{mode}'], level=logging.INFO)
    config['logger'] = logging.getLogger(__name__)
    config['logger'].info(f"### {mode.upper()} MODE - LOG ###")

    config[f'{mode}_progress_file'] = f'{mode}_progress.txt'
    config[f'{mode}_progress_file_path'] = config['logs_folder_path'].joinpath(config[f'{mode}_progress_file'])
    config['logger'].info("Building necessary folders ...")
    config['logger'].info("Folder 'results' and corresponding subfolder have been created in the current working directory.")
    print(f'### See logs/verbose_{mode}.log file for log details! ###')

    # Checking the GPU availability 
    config['logger'].info("Checking the GPU availability")
    config['device'] = check_for_gpus(config['logger'])

    return config


# Dataset Processing 

def create_named_tuple(name, state_vector_labels):
    return namedtuple(name, state_vector_labels)

def named_tuple_definition(strata_labels, state_vector_labels):

    Collection = {}
    for stratum in strata_labels:
        Collection[stratum] = create_named_tuple(f"Collection_{stratum}", state_vector_labels)
    
    return Collection

def import_data_per_stratum_all(stratum_path, N_time_step):
    
    # Get a list of files
    files = os.listdir(stratum_path)

    for filename in files:
        data0 = np.loadtxt(stratum_path.joinpath(filename), skiprows=1)

        dims1 = (data0.shape[0],1,N_time_step)

        break

    coordinates = np.zeros((dims1))
    u = np.zeros((dims1))

    files.sort()
    i = 0
    # Wrap the loop with tqdm for a progress bar
    for filename in tqdm(files, desc="Processing Files "):
        data = np.loadtxt(stratum_path.joinpath(filename), skiprows=1)
       
        coordinates[:,:,i] = data[:,1:2]
        
        u[:,:,i] = data[:,2:3]

        i+=1

    return np.hstack((coordinates, u,))

def import_data(strata_path, N_time_step):

    state_vector_data = {}
    for key,value in strata_path.items():
        print(f"{key}")
        state_vector_data[key] = import_data_per_stratum_all(value, N_time_step)
       
    return state_vector_data

def build_stratified_dataset_based_on_path(strata_path, N_time_step):
    
    state_vector_data = import_data(strata_path, N_time_step)
  
    stratified_dataset = []
    stratified_dataset_labels = []

    for key, value in state_vector_data.items():
        stratified_dataset.append(value)
        stratified_dataset_labels.extend([key] * value.shape[0])

    stratified_dataset = np.vstack(stratified_dataset)
    stratified_dataset_labels = np.array(stratified_dataset_labels)

    return stratified_dataset, stratified_dataset_labels

def split_dataset(stratified_dataset, stratified_dataset_labels, seed, train_fraction, test_fraction):

    # Proportional Split -> stratified into train_aux/test
    train_dataset_aux, test_dataset, train_dataset_labels_aux, test_dataset_labels = train_test_split(
        stratified_dataset, stratified_dataset_labels, test_size=test_fraction, stratify=stratified_dataset_labels, random_state=seed)

    # Proportional Split -> train_aux into val/train
    train_dataset, val_dataset, train_dataset_labels, val_dataset_labels = train_test_split(
        train_dataset_aux, train_dataset_labels_aux, train_size=train_fraction/(1-test_fraction), stratify=train_dataset_labels_aux, random_state=seed)
    
    
    return train_dataset, train_dataset_labels, val_dataset, val_dataset_labels, test_dataset, test_dataset_labels

def recover_strata(strata_labels, subset_dataset, subset_dataset_labels, collection, N_time_step):
    
    strata_data = {}
    for stratum in strata_labels:  
        idx = np.where(subset_dataset_labels == stratum)[0]
        dataset = subset_dataset[idx]
        # Split into column vectors
        dataset_tuple = tuple(dataset[:, i,:].reshape(-1,1,N_time_step) for i in range(dataset.shape[1]))#
        strata_data[stratum] = collection[stratum](*dataset_tuple)
        
    return strata_data

def save_data_call(cwd, strata_data, flow_label, subset):

    for key, value in strata_data.items():
        with open(cwd.joinpath(f".dataset/STRATA_{flow_label}/{key}_stratum_data_{subset}.pkl"), "wb") as file:
            pickle.dump(value, file)
    
def generate_splitted_datasets(cwd, strata_labels, flow_label, seed, collection, N_time_step, train_fraction, test_fraction):

    strata_path = {}
    for stratum in strata_labels:
        strata_path[stratum] = cwd.joinpath(f'.dataset/STRATA_{flow_label}/{stratum}/')

    # Stratified dataset
    stratified_dataset, stratified_dataset_labels = build_stratified_dataset_based_on_path(strata_path, N_time_step)
    

    # Split dataset
    train_dataset, train_dataset_labels, val_dataset,val_dataset_labels,\
    test_dataset, test_dataset_labels = split_dataset(stratified_dataset, stratified_dataset_labels, seed, train_fraction, test_fraction)

    # Recover stratum
    strata_data_train = recover_strata(strata_labels, train_dataset, train_dataset_labels, collection, N_time_step)
    strata_data_val = recover_strata(strata_labels, val_dataset, val_dataset_labels, collection, N_time_step)
    strata_data_test = recover_strata(strata_labels, test_dataset, test_dataset_labels, collection, N_time_step)
    

    # Save data using pickle
    save_data_call(cwd, strata_data_train, flow_label, subset = 'train')
    save_data_call(cwd, strata_data_val, flow_label, subset = 'val')
    save_data_call(cwd, strata_data_test, flow_label, subset = 'test')

def generate_full_dataset(cwd, strata_labels, flow_label, collection, N_time_step):
    
    strata_path = {}
    for stratum in strata_labels:
        strata_path[stratum] = cwd.joinpath(f'.dataset/STRATA_{flow_label}/{stratum}/')
        
    # Import data 
    state_vector_data = import_data(strata_path, N_time_step)

    strata_data_full = {}
    for stratum in strata_labels:  
        dataset = state_vector_data[stratum]
        # Split into column vectors
        dataset_tuple = tuple(dataset[:, i, :].reshape(-1,1,N_time_step) for i in range(dataset.shape[1])) #
        strata_data_full[stratum] = collection[stratum](*dataset_tuple)

    # Save data using pickle
    save_data_call(cwd, strata_data_full, flow_label, subset = 'full')

def processing_call(config, collection):

    cwd = config['cwd']
    strata_labels = config['dataset']['strata_labels']
    varying_param_label = config['dataset']['varying_param_label']
    
    train_fraction = config['dataset']['train_fraction']
    test_fraction = config['dataset']['test_fraction']
   
    N_time_step = config['dataset']['N_time_step']
    initial_time = config['dataset']['initial_time']
    total_time = config['dataset']['total_time']
    time_step = config['dataset']['time_step']
    
    seed = config['seed']

    for param_label in varying_param_label:

        print("#-----------------------------------------#")
        print(f"### Working on {param_label} ###")
        print("#-----------------------------------------#")
        print("#-------------#")
        print(f"# Splitted datasets")
        print("#-------------#")
        generate_splitted_datasets(cwd, strata_labels, param_label, seed, collection, N_time_step, train_fraction, test_fraction)
        print("#--------------#")
        print(f"# Full dataset")
        print("#--------------#")
        generate_full_dataset(cwd, strata_labels, param_label, collection, N_time_step)


        Collection_Time = namedtuple(
        "Collection_Time", ['time_vector', 'N_time_step', 'total_time', 'time_step']
        )

        ## SAVE TIME DATA
        # TO UPDATE (based on the considered time interval and time step) 
        time_vector = np.linspace(initial_time, total_time, N_time_step)

        # collection initialization
        timePrm = Collection_Time(time_vector, N_time_step, total_time, time_step) 

        with open(cwd.joinpath(f".dataset/STRATA_{param_label}/TIME_data.pkl"), "wb") as file:
            pickle.dump(timePrm, file)

    print("### Done ! ###")



# Data Manipulation
def data_dict_to_ds(state_vector_labels,N_time_step, data):
    
    datas_tuple = tuple(data[:, i,:].reshape(-1,1,N_time_step) for i in range(data.shape[1]))#
    Collection = create_named_tuple(f"Collection_Data", state_vector_labels)
    data_ds = Collection(*datas_tuple)
    return data_ds

def load_data(data_path,ID,strata_labels, subset):
    """Load time and strata data from pickle files.
    """
    # Load time data
    with open(data_path.joinpath(f'STRATA_{ID}/TIME_data'+'.pkl'), 'rb') as inp:
        timePrm = pickle.load(inp)
    
    # Load strata data
    strata_data = {}
    for key in strata_labels:
        with open(data_path.joinpath(f'STRATA_{ID}/{key}_stratum_data_{subset}'+'.pkl'), 'rb') as inp:
            strata_data[key] = pickle.load(inp)

    
    return timePrm, strata_data

def get_subset_ds_deeponet(config, chosen_flow_label, subset):

    mode =  config['mode']
    Collection = namedtuple("Collection", config['dataset']['strata_labels'])
    config['logger'].info(f"Loading dataset  - {subset}")
    timePrm, strata_data = load_data(config['data_path'], chosen_flow_label, config['dataset']['strata_labels'], subset)

    ds = Collection(**strata_data)
   
    return timePrm, ds

def all_data_transformations_deeponet(config, subset):

    logger = config['logger']
    mode =  config['mode']
    
    ds = {}
    Collection = namedtuple("Collection", config['dataset']['strata_labels'])


    for idx, chosen_flow_label in enumerate(config['train']['training_param_label']):
       
        # Load data
        config['logger'].info(f"Loading dataset  - {subset}")
        timePrm, strata_data = load_data(config['data_path'], chosen_flow_label, config['dataset']['strata_labels'], subset)

      
        config['pde_param']['c'][chosen_flow_label] = config['train']['training_param'][idx]
        
        ds[chosen_flow_label] = Collection(**strata_data)

    return timePrm, ds

def random_data_extraction_deeponet(config, full_ds):

    decimal_fraction = config['train']['data_extraction']['decimal_fraction']
    data_ds = {} 
    data_percent = {}

    # Coordinates are identical for all IDs
    chosen_flow_label_0 = config['train']['training_param_label'][0]
    volumePrm = full_ds[chosen_flow_label_0].PHY
    coordinates = volumePrm.coord_x
    N_coordinates = coordinates.shape[0]
    x_idx = np.random.choice(N_coordinates,  round(decimal_fraction*N_coordinates), replace=False)

    for chosen_flow_label in config['train']['training_param_label']:
        
        volumePrm = full_ds[chosen_flow_label].PHY
         # Coordinates
        x = coordinates[x_idx]
        
        # Velocities
        velocity = volumePrm.u
        u = velocity[x_idx]
        
        data = np.hstack([x,u])
        data_percent[chosen_flow_label] = x.shape[0]/N_coordinates * 100

        data_ds[chosen_flow_label] = data_dict_to_ds(config['dataset']['state_vector_labels'],config['dataset']['N_time_step'], data)

        config['logger'].info(f"The data percentage with respect to VOLUME for the Loss_data is {data_percent[chosen_flow_label]}, from a total of {volumePrm.coord_x.shape[0]}.")
    
    return data_ds, data_percent


# Training

def build_stratum_dataset(stratumPrm, time_vector, label):
    # Stratum 
    # Coordinates
    if label == 'BC':
        coordinates = stratumPrm.coord_x[0,:,:][None,:]
    else:
        coordinates = stratumPrm.coord_x
    N_coord = coordinates.shape[0]

    X = coordinates[:,0,:]  # N_coord x N_timeSteps
    x = X.T.flatten()[:, None]  # N_coord * N_timeSteps x 1
    t = np.repeat(time_vector, N_coord)[:, None]  # N_coord * N_timeSteps x 1
    
    xt = np.hstack((x,t))
    

    # Velocities
    if label == 'BC':
        u = stratumPrm.u[0,:,:][None,:]
    else:
        u = stratumPrm.u
    U1 = u[:,0,:]
    v = U1.T.flatten()[:, None] 
    
    return xt, v

def craft_bc_and_ic_dataset(train_ds, time_vector):
    
    # BC
    xt_bc, u_bc = build_stratum_dataset(train_ds.BC, time_vector, label = 'BC')
    # PHY
    xt_phy, u_phy = build_stratum_dataset(train_ds.PHY, time_vector, label = 'PHY')
    
    ## IC - PHY
    idx = np.where(xt_phy[:,1] == 0)
    xt_ic = xt_phy[idx]
    u_ic = u_phy[idx]

    return xt_bc, u_bc, xt_ic, u_ic


def craft_validation_dataset(val_ds, time_vector):
    
    # BC
    xt_bc, u_bc = build_stratum_dataset(val_ds.BC, time_vector, label = 'BC')
    # PHY
    xt_phy, u_phy = build_stratum_dataset(val_ds.PHY, time_vector, label = 'PHY')

    xt_star = np.vstack((xt_bc,xt_phy))
    u_ref = np.vstack((u_bc,u_phy))

    return xt_star, u_ref

def get_coordinates_for_generator(ds, time_vector):

    
    phyPrm = ds.PHY
    time = 0
    
    time_min = time_vector.min()
    time_max = time_vector.max()

    # We will consider the spatial points x,y,z provided for the mesh 
    coordinates = phyPrm.coord_x[:,:,time]

    return coordinates,time_min,time_max

def from_time_to_index(t_value):

    time = [f"{np.round(i, 6):.6f}" for i in np.linspace(0, 0.98,50)]

    index = time.index(t_value)

    return index

# Testing

class DataLabel(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        label = self.labels[idx]
        return data_point, label
    
def build_dataloader(test_dataset, test_dataset_labels, batch_size):

    # Initialize custom dataset
    custom_dataset_test = DataLabel(test_dataset, test_dataset_labels)

    # Create DataLoaders for test sets
    test_dataloader = DataLoader(custom_dataset_test, batch_size=batch_size, shuffle=True, drop_last= True)
    
    return test_dataloader

def build_stratified_dataset_based_on_data(ds, time_vector):

    # BC
    xt_bc, u_bc =  build_stratum_dataset(ds.BC, time_vector, 'BC')
    # PHY
    xt_phy, u_phy =  build_stratum_dataset(ds.PHY, time_vector, 'PHY')
    

    bc_dataset = np.hstack(
        (xt_bc, u_bc))
    phy_dataset = np.hstack(
        (xt_phy, u_phy))
    
    stratified_dataset = np.vstack((bc_dataset,phy_dataset))


    stratified_dataset_labels = np.array(['BC'] * bc_dataset.shape[0] + ['PHY'] * phy_dataset.shape[0])


    return stratified_dataset, stratified_dataset_labels

def create_dataset_for_test(config, test_ds, timePrm):
    
    # Consider dataset as stratified
    stratified_dataset, stratified_dataset_labels = build_stratified_dataset_based_on_data(test_ds, timePrm.time_vector[:,None])
    
    
    # Build mini-batches
    dataloader_test = build_dataloader(stratified_dataset, stratified_dataset_labels,
                                        batch_size=config['test']['batch_size'])

    # Determine number of mini-batches
    N_batches_test = len(dataloader_test)

    return dataloader_test, N_batches_test

def plot_only_magnitude(config, star, ref, idx1, idx2, xlabel, ylabel, plane_name, timestep, ID='Vel', label='Planes'):
    
    if ID == 'Vel':
        vmin = 0
        vmax = config['pde_param']['V_inlet_max']
    else:
        vmin = np.min(ref)
        vmax = np.max(ref)

    # Plot
    fig = plt.figure(figsize=(18, 5))
   
    scatter_ref = plt.scatter(star[:,idx1], star[:,idx2], c= ref, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(scatter_ref, label=f'{ID}_ref')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ID}- {label} - Exact in {plane_name} and time={timestep}")
    plt.tight_layout()


    # Save the figure
    fig_path = config['charts_folder_path'].joinpath(f'{ID}_absolute_error_in_{plane_name}_time_{timestep}_{label}.png')
    fig.savefig(fig_path, bbox_inches="tight", dpi=200)
    plt.close()

def plot_magnitude_and_save_absolute_error(config, star, ref, pred, idx1, idx2, xlabel, ylabel, plane_name, timestep, ID='Vel'):
    
    if ID == 'Vel':
        vmin = 0
        vmax = config['pde_param']['V_inlet_max']
    else:
        vmin = np.min(ref)
        vmax = np.max(ref)

    # Plot
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    
    scatter_ref = plt.scatter(star[:,idx1], star[:,idx2], c= ref, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(scatter_ref, label=f'{ID}_ref')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ID} - Exact in {plane_name} and time={timestep}")
    plt.tight_layout()

    plt.subplot(1, 3, 2)

    scatter_pred = plt.scatter(star[:,idx1], star[:,idx2], c= pred, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(scatter_pred, label=f'{ID}_pred')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ID} - Predicted in {plane_name} and time={timestep}")
    plt.tight_layout()

    
    abs_error = abs(ref - pred)
    ae_vmax = np.max(abs_error)
    ae_vmin = np.min(abs_error)
    plt.subplot(1, 3, 3)

    ae_scatter = plt.scatter(star[:,idx1], star[:,idx2], c= abs_error, cmap="jet", vmin=ae_vmin, vmax=ae_vmax)
    plt.colorbar(ae_scatter, label='abs_error')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ID} - Absolute error in {plane_name} and time={timestep}")
    plt.tight_layout()

    # Save the figure
    fig_path = config['charts_folder_path'].joinpath(f'{ID}_absolute_error_in_{plane_name}_time_{timestep}.png')
    fig.savefig(fig_path, bbox_inches="tight", dpi=200)
    plt.close()

    config['logger'].info("#-----------------------------------------#")
    config['logger'].info(f"Accuracy in {plane_name} and time={timestep} - full_dataset")
            
    config['logger'].info(f"max absolute error in magnitude of {ID}: {np.max(abs_error)}")
    config['logger'].info(f"min absolute error in magnitude of {ID}: {np.min(abs_error)}")
    config['logger'].info(f"mean absolute error in magnitude of {ID}: {np.mean(abs_error)}")
  
    print("#-----------------------------------------#")
    print(f"Accuracy in {plane_name} and time={timestep}  - full_dataset")
    
    print(f"max absolute error in magnitude of {ID}: {np.max(abs_error)}")
    print(f"min absolute error in magnitude of {ID}: {np.min(abs_error)}")
    print(f"mean absolute error in magnitude of {ID}: {np.mean(abs_error)}")