from pathlib import Path
import numpy as np

def u_close_form(x, t, c):
    return np.where(x >= c*t,
                    np.sin(np.pi * (x - c*t)),   # from initial line
                    np.sin(np.pi/2 * (t - x/c)))   # from inflow boundary

def governing_equation(cwd, varying_param_i, varying_param_label_i, stratum, N_coll, t):
    
    c = varying_param_i
    x = np.linspace(0,1,N_coll)
    # Close form solution
    u = u_close_form(x, t, c)
    
    ds = np.stack([x,u]).T
    
    label = f'time.{t:.2f}'.replace('.','_')
    
    ds_folder = f".dataset/STRATA_{varying_param_label_i}/{stratum}"
    ds_path = cwd.joinpath(ds_folder)

    with open(ds_path.joinpath(f'data_{label}.txt'), 'w') as file:
        file.write("nodes\tx\tu\n")
        for i in range(ds.shape[0]):
            file.write(f"{i+1}\t{ds[i,0]:.6f}\t{ds[i,1]:.6f}\n")


def boundary_condition(cwd, varying_param_label_i, stratum, t):
    
    N_bc = 100
    # Boundary condition
    x = 0.0
    u = np.sin(np.pi/2 * t )
    
    ds = np.repeat(np.stack([[x,u]]), N_bc, axis=0)
    
    label = f'time.{t:.2f}'.replace('.','_')
    
    ds_folder = f".dataset/STRATA_{varying_param_label_i}/{stratum}"
    ds_path = cwd.joinpath(ds_folder)

    with open(ds_path.joinpath(f'data_{label}.txt'), 'w') as file:
        file.write("nodes\tx\tu\n")
        for i in range(ds.shape[0]):
            file.write(f"{i+1}\t{ds[i,0]:.6f}\t{ds[i,1]:.6f}\n")

