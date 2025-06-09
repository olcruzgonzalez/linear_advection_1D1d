from pathlib import Path
import numpy as np

def calculate_phi(k: float) -> float:
  """
  Converts a k parameter within the range [0.5, 2] to a phi parameter
  in the range [0, 2π] through a linear correspondence.

  Args:
    k: A float between 0.5 and 2.0.

  Returns:
    The corresponding phi value as a float between 0 and 2π.

  Raises:
    ValueError: If the input k is outside the specified range [0.5, 2].
  """
  if not (0.5 <= k <= 2):
    raise ValueError("Input k must be within the range [0.5, 2].")

  # Normalize k from [0.5, 2] to [0, 1]
  normalized_k = (k - 0.5) / (2 - 0.5)

  # Scale the normalized value to the [0, 2π] range
  phi = normalized_k * 2 * np.pi

  return phi

def u_0(x,k,phi):
    # return np.sin(k*np.pi * x + phi) - np.sin(phi)# from initial condition line
    return np.sin(np.pi * x + phi) - np.sin(phi)# from initial condition line

def g_bc(t, k):
    return np.sin(k*np.pi/2 * t) + k*np.pi/2*t  # from inflow boundary condition

def u_close_form(x, t, c, k, phi):
    return np.where(x >= c*t,
                    u_0(x - c*t,k, phi),   # from initial condition line
                    g_bc(t - x/c,k))  # from inflow boundary condition


def governing_equation(cwd, L, c, varying_param_i, varying_param_label_i, stratum, N_coll, t_i):
    
    k_i = varying_param_i
    x = np.linspace(0,L,N_coll)
    phi_i = calculate_phi(k_i)  # Convert k to phi
    # Close form solution
    u = u_close_form(x, t_i, c, k_i, phi_i)

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

