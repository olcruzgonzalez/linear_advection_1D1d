import os
from pathlib import Path

import numpy as np



def governing_equation(cwd, varying_param_i, varying_param_label_i, stratum, N_coll, t):
    
    c = varying_param_i
    x = np.linspace(0,1,N_coll)
    # Close form solution
    u = np.sin(x)*np.cos(c*t) - np.sin(c*t)*np.cos(x)
    
    ds = np.stack([x,u]).T
    
    label = f'time.{t:.2f}'.replace('.','_')
    
    ds_folder = f".dataset/STRATA_{varying_param_label_i}/{stratum}"
    ds_path = cwd.joinpath(ds_folder)

    with open(ds_path.joinpath(f'data_{label}.txt'), 'w') as file:
        file.write("nodes\tx\tu\n")
        for i in range(ds.shape[0]):
            file.write(f"{i+1}\t{ds[i,0]:.6f}\t{ds[i,1]:.6f}\n")

