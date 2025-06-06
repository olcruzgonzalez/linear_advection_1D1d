from pathlib import Path
import numpy as np

def f_ic(x,k):
    return np.sin(np.pi * x)

def g_bc(t, k):
    return np.sin(k*np.pi/2 * t) + k*np.pi/2*t  # from inflow boundary condition

def u_close_form(x, t, c, k):
    return np.where(x >= c*t,
                    f_ic(x - c*t,k),   # from initial condition line
                    g_bc(t - x/c,k))  # from inflow boundary condition


def governing_equation(cwd, L, c, varying_param_i, varying_param_label_i, stratum, N_coll, t_i):
    
    k_i = varying_param_i
    x = np.linspace(0,L,N_coll)
    # Close form solution
    u = u_close_form(x, t_i, c, k_i)
    
    ds = np.stack([x,u]).T
    
    label = f'time.{t_i:.2f}'.replace('.','_')
    
    ds_folder = f".dataset/STRATA_{varying_param_label_i}/{stratum}"
    ds_path = cwd.joinpath(ds_folder)

    with open(ds_path.joinpath(f'data_{label}.txt'), 'w') as file:
        file.write("nodes\tx\tu\n")
        for i in range(ds.shape[0]):
            file.write(f"{i+1}\t{ds[i,0]:.6f}\t{ds[i,1]:.6f}\n")


def boundary_condition(cwd, varying_param_i, varying_param_label_i, stratum, t_i):
    
    k_i = varying_param_i
    N_bc = 100
    # Boundary condition
    x = 0.0
    u = g_bc(t_i, k_i)
    
    ds = np.repeat(np.stack([[x,u]]), N_bc, axis=0)
    
    label = f'time.{t_i:.2f}'.replace('.','_')
    
    ds_folder = f".dataset/STRATA_{varying_param_label_i}/{stratum}"
    ds_path = cwd.joinpath(ds_folder)

    with open(ds_path.joinpath(f'data_{label}.txt'), 'w') as file:
        file.write("nodes\tx\tu\n")
        for i in range(ds.shape[0]):
            file.write(f"{i+1}\t{ds[i,0]:.6f}\t{ds[i,1]:.6f}\n")

