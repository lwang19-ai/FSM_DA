#!/usr/bin/env python3
"""
Simplified EnKF for testing numerical stability
"""

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from model import ForwardParams, run_forward
from enkf_helper import *
import numpy.linalg as LA

def robust_solve(M, vec, max_jitter=1e-6):
    """Ultra-robust matrix solver"""
    try:
        # Try direct solve first
        return LA.solve(M, vec)
    except LA.LinAlgError:
        # Progressive regularization
        for jitter in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
            try:
                M_reg = M + np.eye(M.shape[0]) * jitter
                return LA.solve(M_reg, vec)
            except LA.LinAlgError:
                continue
        
        # Ultimate fallback: pseudo-inverse
        try:
            return LA.pinv(M) @ vec
        except:
            # If everything fails, return zero
            return np.zeros_like(vec)

def simulate_member_simple(args):
    """Simplified simulation function"""
    i, nx, ny, current_time, hSL_val, hSS_val, param_values, z0_i, p0_i = args
    
    try:
        from model import ForwardParams
        
        params_i = ForwardParams(
            nx=nx, ny=ny, times=current_time,
            diffusion_coeff=param_values[0],
            thickness_scale=param_values[1],
            erosion_rate=0.1,
            subsidence_scalar=0.1,
            diffusion_iters=6,
            bowl_amp=0.5,
            bowl_center_east=0.20,
            bowl_center_north=0.20,
            bowl_width_east=0.22,
            bowl_width_north=0.50
        )

        z_layers, p_layers, deposits, compositions = run_forward(
            z0_i, p0_i, current_time, [hSL_val], current_time, [hSS_val], params_i
        )

        return z_layers[-1], p_layers[-1], deposits[-1], compositions[-1]
        
    except Exception as e:
        print(f"Error in ensemble member {i}: {e}")
        return z0_i, p0_i, np.zeros_like(z0_i), np.zeros((z0_i.shape[0], z0_i.shape[1], 4))

def main():
    # Load data
    print("Loading reference data...")
    data = np.load("./data/reference_fsm_results.npz", allow_pickle=True)
    nx, ny, nt = data["nx"], data["ny"], data["nt"]
    times = data["times"]
    z0 = data["z0"]
    p0 = data["p0"]
    hSL_vals = data["hSL_vals"]
    hSS_vals = data["hSS_vals"]
    z_layers = data["z_layers"]
    p_layers = data["p_layers"]
    deposits = data["deposits"]
    compositions = data["compositions"]
    sigma_z = float(data["sigma_z"])
    sigma_p = float(data["sigma_p"])
    sigma_d = float(data["sigma_d"])
    sigma_c = float(data["sigma_c"])
    z_layers_meas = data["z_layers_meas"]
    p_layers_meas = data["p_layers_meas"]
    deposits_meas = data["deposits_meas"]
    compositions_meas = data["compositions_meas"]
    params = data["params"].item()

    # Simple settings
    N_ens = 30  # Smaller ensemble for stability
    num_params = 2
    param_names = ["diffusion_coeff", "thickness_scale"]
    true_params = np.array([params.diffusion_coeff, params.thickness_scale])
    param_stds = np.array([0.05, 5.0])  # Conservative uncertainties
    
    print(f"True values: {true_params}")
    
    # Initialize ensemble
    param_ensemble = np.empty((N_ens, num_params))
    for i in range(num_params):
        param_ensemble[:, i] = np.random.normal(true_params[i], param_stds[i], size=N_ens)
    
    # Constrain initial ensemble
    param_ensemble[:, 0] = np.clip(param_ensemble[:, 0], 0.1, 0.5)
    param_ensemble[:, 1] = np.clip(param_ensemble[:, 1], 20.0, 60.0)
    
    z0_ensemble = np.empty((N_ens, ny, nx))
    p0_ensemble = np.empty((N_ens, ny, nx, 4))
    
    for i in range(N_ens):
        z0_ensemble[i] = np.random.normal(z0, 5.0)
        rnd = np.random.normal(p0, 0.02)
        rnd = np.clip(rnd, 0, 1)
        rnd /= rnd.sum(axis=-1, keepdims=True) + 1e-12
        p0_ensemble[i] = rnd

    # Observation setup
    obs_idx = build_obs_index(ny, nx, stride_z=16, stride_p=16, stride_d=16, stride_c=16,
                              use_z=True, use_p=True, use_d=True, use_c=True)
    
    # Simple observation error
    R_var = np.full(len(obs_idx), 1.0)  # Uniform observation error
    R_inv = 1.0 / R_var
    
    # Storage
    param_history = np.empty((min(50, nt), N_ens, num_params))  # Only store first 50 steps
    
    print("Starting simplified EnKF...")
    
    for t in range(min(50, nt)):  # Only run first 50 steps
        if t % 10 == 0:
            print(f"Processing time step {t}/{min(50, nt)}")
            
        current_time = times[t:t+1]
        
        # Run ensemble
        args_list = [
            (i, nx, ny, current_time, hSL_vals[t], hSS_vals[t],
             param_ensemble[i].copy(), z0_ensemble[i].copy(), p0_ensemble[i].copy())
            for i in range(N_ens)
        ]
        
        with ProcessPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(simulate_member_simple, args_list))

        # Build state vector (only parameters + elevation)
        X = np.empty((N_ens, num_params + nx*ny))
        for i in range(N_ens):
            X[i, :num_params] = param_ensemble[i]
            X[i, num_params:] = results[i][0].ravel()  # Only elevation

        # Get observations (only elevation for simplicity)
        obs_full = np.concatenate([
            z_layers_meas[t].ravel(),
            p_layers_meas[t].ravel(),
            deposits_meas[t].ravel(),
            compositions_meas[t].ravel()
        ])
        obs = obs_full[obs_idx]

        # EnKF update
        X_mean = X.mean(axis=0)
        X_pert = X - X_mean
        
        Y_all = X[:, num_params:][:, obs_idx[obs_idx < nx*ny]]  # Only elevation observations
        Y_mean = Y_all.mean(axis=0)
        Y_pert = Y_all - Y_mean
        
        if Y_pert.shape[1] == 0:  # No valid observations
            print(f"No valid observations at time {t}")
            continue
            
        Nm1 = float(N_ens - 1)
        S = Y_pert / np.sqrt(Nm1)
        
        # Simple update
        try:
            SR = S / 1.0  # Simple observation error
            M = np.eye(N_ens) + SR @ S.T
            
            # Parameter update only
            Cxy_param = (X_pert[:, :num_params].T @ S) / np.sqrt(Nm1)
            
            for i in range(N_ens):
                d = obs[:Y_pert.shape[1]] - Y_all[i]
                v = d / 1.0
                Sv = S @ v
                alpha = robust_solve(M, Sv)
                w = v - (S.T @ alpha) / 1.0
                delta_param = Cxy_param @ w
                
                # Very conservative update
                param_ensemble[i] += 0.01 * delta_param
            
            # Constrain parameters
            param_ensemble[:, 0] = np.clip(param_ensemble[:, 0], 0.1, 0.5)
            param_ensemble[:, 1] = np.clip(param_ensemble[:, 1], 20.0, 60.0)
            
        except Exception as e:
            print(f"Update failed at time {t}: {e}")
            continue
        
        # Store results
        param_history[t] = param_ensemble
        
        # Update initial conditions for next step
        for i in range(N_ens):
            z0_ensemble[i] = results[i][0]
            p0_ensemble[i] = results[i][1]
    
    print("EnKF completed!")
    
    # Simple plotting
    plt.figure(figsize=(12, 5))
    
    for i, name in enumerate(param_names):
        plt.subplot(1, 2, i+1)
        
        # Plot ensemble mean
        param_mean = param_history[:, :, i].mean(axis=1)
        param_std = param_history[:, :, i].std(axis=1)
        
        time_range = range(len(param_mean))
        plt.plot(time_range, param_mean, 'r-', linewidth=2, label='Ensemble mean')
        plt.fill_between(time_range, param_mean - param_std, param_mean + param_std, 
                        color='r', alpha=0.2, label='±1σ')
        plt.axhline(y=true_params[i], color='b', linestyle='--', label='True value')
        
        plt.xlabel('Time step')
        plt.ylabel(name)
        plt.title(f'Parameter: {name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./plots/simple_enkf_test.png', dpi=300)
    plt.show()
    
    print("Test completed!")

if __name__ == '__main__':
    main()