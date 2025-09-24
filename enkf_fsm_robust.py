# %%
import numpy as np
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
import numpy.linalg as LA
from enkf_helper import *
import matplotlib.pyplot as plt
from numba import jit, prange
from scipy.linalg import cho_factor, cho_solve, LinAlgError, LinAlgWarning
from sensitivity_analysis import SensitivityAnalyzer
import copy
from model import run_forward

# Optimized Gaspari-Cohn function with Numba
@jit(nopython=True, parallel=True)
def fast_gaspari_cohn(distances, c):
    """Numba-optimized Gaspari-Cohn function"""
    n = distances.size
    rho = np.zeros(n)
    
    for i in prange(n):
        x = abs(distances[i]) / c
        if x <= 1.0:
            rho[i] = 1 + x*x*(-5/3 + 0.5*x + 0.25*x*x) + x**5*(-0.1)
        elif x <= 2.0:
            rho[i] = 4 - 5*x + (5/3)*x*x + 0.5*x**3 - (1/6)*x**4 + (1/12)*x**5
    
    return rho

class RobustEnKF:
    def __init__(self, N_ens, num_params, nx, ny):
        self.N_ens = N_ens
        self.num_params = num_params
        self.nx, self.ny = nx, ny
        
        # Pre-allocate arrays to avoid repeated allocation
        self.state_dim = num_params + nx*ny + nx*ny*4 + nx*ny + nx*ny*4
        self.ensemble_vectors = np.empty((N_ens, self.state_dim), dtype=np.float64)
        
        # Pre-compute offsets for efficient array slicing
        self.offsets = {
            'params': 0,
            'z': num_params,
            'p': num_params + nx*ny,
            'deposits': num_params + nx*ny + nx*ny*4,
            'compositions': num_params + nx*ny + nx*ny*4 + nx*ny
        }
        
    def pack_ensemble_vectors(self, param_ensemble, z_outputs, p_outputs, d_outputs, c_outputs):
        """Efficiently pack ensemble data using pre-allocated arrays"""
        # Use array views for efficient packing
        self.ensemble_vectors[:, self.offsets['params']:self.offsets['z']] = param_ensemble
        self.ensemble_vectors[:, self.offsets['z']:self.offsets['p']] = z_outputs.reshape(self.N_ens, -1)
        self.ensemble_vectors[:, self.offsets['p']:self.offsets['deposits']] = p_outputs.reshape(self.N_ens, -1)
        self.ensemble_vectors[:, self.offsets['deposits']:self.offsets['compositions']] = d_outputs.reshape(self.N_ens, -1)
        self.ensemble_vectors[:, self.offsets['compositions']:] = c_outputs.reshape(self.N_ens, -1)
        return self.ensemble_vectors
    
    def robust_cholesky_solve(self, M, regularization=1e-6):
        """Robust solver with regularization and fallbacks"""
        n = M.shape[0]
        
        # Check for NaN/Inf
        if np.any(~np.isfinite(M)):
            print("Warning: Non-finite values in covariance matrix, using identity")
            return lambda x: x / n
        
        # Add regularization
        M_reg = M + regularization * np.eye(n)
        
        try:
            # Try Cholesky first
            cho_fac = cho_factor(M_reg, check_finite=False)
            def solve_M(vec):
                return cho_solve(cho_fac, vec, check_finite=False)
            return solve_M
        except:
            try:
                # Fallback to SVD-based pseudoinverse
                U, s, Vt = np.linalg.svd(M_reg, full_matrices=False)
                # Threshold small singular values
                s_thresh = np.maximum(s, regularization * s[0])
                s_inv = 1.0 / s_thresh
                M_inv = (Vt.T * s_inv) @ U.T
                def solve_M(vec):
                    return M_inv @ vec
                return solve_M
            except:
                # Final fallback - return scaled identity
                def solve_M(vec):
                    return vec / (np.trace(M_reg) / n + regularization)
                return solve_M

class ObservationManager:
    """Cache observation data and operations"""
    def __init__(self, ny, nx, obs_idx, sigma_z, sigma_p, sigma_d, sigma_c):
        self.ny, self.nx = ny, nx
        self.obs_idx = obs_idx
        
        # Pre-compute R matrix
        off_z = 0
        off_p = off_z + ny*nx
        off_d = off_p + ny*nx*4
        off_c = off_d + ny*nx
        
        R_full = np.empty(ny*nx*(1+4+1+4), dtype=float)
        R_full[off_z:off_z+ny*nx] = sigma_z**2
        R_full[off_p:off_p+ny*nx*4] = sigma_p**2
        R_full[off_d:off_d+ny*nx] = sigma_d**2
        R_full[off_c:off_c+ny*nx*4] = sigma_c**2
        
        self.R_var = R_full[obs_idx] + 1e-8
        self.R_inv = 1.0 / self.R_var
        
    def get_observations(self, z_meas, p_meas, d_meas, c_meas):
        """Efficiently extract observations"""
        obs_full = np.concatenate([
            z_meas.ravel(),
            p_meas.ravel(), 
            d_meas.ravel(),
            c_meas.ravel()
        ])
        return obs_full[self.obs_idx]

def simulate_member_sensitivity_robust(args):
    """Robust simulation with better error handling"""
    i, nx, ny, current_time, hSL_val, hSS_val, selected_params, param_names, z0_i, p0_i, base_params = args
    
    try:
        # Create a copy of the base params
        params_i = copy.deepcopy(base_params)
        
        # Update parameters based on selection with bounds checking
        for j, param_name in enumerate(param_names):
            param_val = selected_params[j]
            
            if param_name.startswith('supply_comp_'):
                comp_idx = int(param_name.split('_')[-1])
                # Ensure valid composition values
                param_val = np.clip(param_val, 0.01, 0.8)
                params_i.supply_composition[comp_idx] = param_val
                # Renormalize supply composition
                total = params_i.supply_composition.sum()
                if total > 0:
                    params_i.supply_composition /= total
                else:
                    # Fallback to default composition
                    params_i.supply_composition = np.array([0.5, 0.3, 0.15, 0.05])
            elif param_name == 'diffusion_coeff':
                # Ensure positive diffusion coefficient
                param_val = np.clip(param_val, 0.01, 2.0)
                setattr(params_i, param_name, param_val)
            elif param_name == 'thickness_scale':
                # Ensure positive thickness scale
                param_val = np.clip(param_val, 1.0, 200.0)
                setattr(params_i, param_name, param_val)
            else:
                setattr(params_i, param_name, param_val)

        # Run forward model with error checking
        z_layers, p_layers, deposits, compositions = run_forward(
            z0_i, p0_i, current_time, [hSL_val], current_time, [hSS_val], params_i
        )

        z_last = z_layers[-1]
        p_last = p_layers[-1]
        d_last = deposits[-1]
        c_last = compositions[-1]

        # Check for finite outputs
        if not (np.all(np.isfinite(z_last)) and
                np.all(np.isfinite(p_last)) and
                np.all(np.isfinite(d_last)) and
                np.all(np.isfinite(c_last))):
            raise ValueError("Forward model produced non-finite outputs")

        return z_last, p_last, d_last, c_last
        
    except Exception as e:
        print(f"Error in ensemble member {i}: {e}")
        # Return fallback values with small perturbations to maintain ensemble spread
        noise_z = np.random.normal(0, 1.0, z0_i.shape)
        noise_p = np.random.normal(0, 0.01, p0_i.shape)
        noise_p = np.clip(noise_p, -0.05, 0.05)  # Limit perturbation size
        
        p_perturbed = p0_i + noise_p
        p_perturbed = np.clip(p_perturbed, 0.01, 0.99)
        p_perturbed = p_perturbed / p_perturbed.sum(axis=-1, keepdims=True)
        
        return (z0_i + noise_z, p_perturbed, 
                np.abs(np.random.normal(0, 1.0, z0_i.shape)), 
                np.random.dirichlet([1, 1, 1, 1], size=z0_i.shape))

# %% load the initial settings, parameters, and results from a reference run.
if __name__ == '__main__':
    data = np.load("./data/reference_fsm_results.npz", allow_pickle=True)

    # Access variables by their keys
    nx = data["nx"]
    ny = data["ny"]
    nt = data["nt"]
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
    
    # Store as base_params for passing to simulation function
    base_params = params

    # create initial ensembles for parameters and state
    N_ens = 50
    
    # Vectorized initialization
    z0_std = 10.0
    z0_ensemble = np.random.normal(z0[None, :, :], z0_std, size=(N_ens, ny, nx))

    p0_std = np.array([0.05, 0.05, 0.05, 0.05])
    p0_ensemble = np.random.normal(p0[None, None, None, :], p0_std[None, None, None, :], 
                                   size=(N_ens, ny, nx, 4))
    p0_ensemble = np.clip(p0_ensemble, 0, 1)
    p0_ensemble /= p0_ensemble.sum(axis=-1, keepdims=True) + 1e-12

    # Pre-allocate all arrays
    deposits_ensemble = np.empty((N_ens, ny, nx))
    compositions_ensemble = np.empty((N_ens, ny, nx, 4))
    
    z_mean_history = np.empty((nt, ny, nx))
    z_std_history = np.empty((nt, ny, nx))
    
    # Keep full history only for final analysis
    z0_history = np.empty((nt, N_ens, ny, nx))
    p0_history = np.empty((nt, N_ens, ny, nx, 4))
    
    # Use less dense observation sampling to reduce numerical issues
    obs_idx = build_obs_index(ny, nx,
                              stride_z=8, stride_p=8, stride_d=8, stride_c=8,  # Reduced density
                              use_z=True, use_p=True, use_d=False, use_c=False)  # Use fewer variables
    
    obs_manager = ObservationManager(ny, nx, obs_idx, sigma_z, sigma_p, sigma_d, sigma_c)

    # Simplified - skip localization initially to test global EnKF
    inflation = 1.10  # Increased inflation to maintain ensemble spread

    #  Sensitivity Analysis
    print("Performing sensitivity analysis...")
    analyzer = SensitivityAnalyzer(params, nx, ny, times, hSL_vals, hSS_vals)

    # Run sensitivity analysis with fewer samples for speed
    samples, param_names_sa, outputs, output_names = analyzer.run_sensitivity_analysis(
        z0, p0, n_samples=50
    )

    # Compute sensitivity indices
    sensitivities = analyzer.compute_sobol_indices(samples, outputs)

    # Plot results and get recommendations
    sensitivity_matrix, overall_sensitivity = analyzer.plot_sensitivity_results(
        param_names_sa, output_names, sensitivities
    )

    # Get recommended parameters for EnKF with lower threshold
    recommended_params = analyzer.recommend_parameters(
        param_names_sa, overall_sensitivity, threshold=0.08  # Lower threshold
    )

    print(f"\nUsing {len(recommended_params)} most sensitive parameters for EnKF:")
    for param in recommended_params:
        print(f"  - {param}")

    # Create parameter mapping for EnKF
    param_values = []
    param_stds = []
    true_params = []

    for param_name in recommended_params:
        if param_name.startswith('supply_comp_'):
            # Handle supply composition components
            comp_idx = int(param_name.split('_')[-1])
            if hasattr(params, 'supply_composition') and comp_idx < len(params.supply_composition):
                true_val = params.supply_composition[comp_idx]
                param_values.append(true_val)
                # Use larger std for composition parameters
                param_stds.append(max(0.02, 0.15 * true_val))
                true_params.append(true_val)
            else:
                print(f"Warning: {param_name} not found in params, skipping")
                continue
        elif hasattr(params, param_name):
            true_val = getattr(params, param_name)
            param_values.append(true_val)
            # Set appropriate standard deviations
            if 'coeff' in param_name:
                param_stds.append(0.08 * true_val)  # 8% for diffusion
            elif 'scale' in param_name:
                param_stds.append(0.25 * true_val)  # 25% for thickness scale
            else:
                param_stds.append(0.15 * true_val)  # 15% for others
            true_params.append(true_val)
        else:
            print(f"Warning: {param_name} not found in params, skipping")
            continue

    # Update arrays to match actual found parameters
    param_values = np.array(param_values)
    param_stds = np.array(param_stds)
    true_params = np.array(true_params)

    # Update num_params to reflect actually found parameters
    num_params = len(true_params)

    # Update param_rw_std with larger values to maintain spread
    param_rw_std = np.array([
        0.005 * true_params[i] if 'coeff' in recommended_params[i] else
        0.01 * true_params[i] if 'scale' in recommended_params[i] else
        0.005 * true_params[i] for i in range(num_params)
    ])

    print(f"Selected parameters: {recommended_params[:num_params]}")
    print(f"True values: {true_params}")
    print(f"Standard deviations: {param_stds}")
    print(f"Random walk stds: {param_rw_std}")

    # Initialize ensemble with better parameter bounds
    param_ensemble = np.random.normal(true_params[None, :], param_stds[None, :], size=(N_ens, num_params))
    
    # Apply parameter bounds
    for i, param_name in enumerate(recommended_params[:num_params]):
        if param_name.startswith('supply_comp_'):
            param_ensemble[:, i] = np.clip(param_ensemble[:, i], 0.02, 0.8)
        elif 'coeff' in param_name:
            param_ensemble[:, i] = np.clip(param_ensemble[:, i], 0.01, 1.0)
        elif 'scale' in param_name:
            param_ensemble[:, i] = np.clip(param_ensemble[:, i], 5.0, 150.0)

    # Initialize robust EnKF
    enkf = RobustEnKF(N_ens, num_params, nx, ny)

    # Update history arrays
    param_mean_history = np.empty((nt, num_params))
    param_std_history = np.empty((nt, num_params))
    param_history = np.empty((nt, N_ens, num_params))

    # Update recommended_params to match actually used parameters
    recommended_params = recommended_params[:num_params]

    print("Starting robust EnKF iterations...")
    
    for t in range(min(nt, 20)):  # Test with fewer time steps first
        if t % 5 == 0:
            print(f"Processing time step {t}/{min(nt, 20)}")
            
        current_time = times[t:t+1]
        
        # Prepare arguments for parallel execution
        args_list = [
            (i, nx, ny, current_time, hSL_vals[t], hSS_vals[t],
             param_ensemble[i].copy(), recommended_params, z0_ensemble[i].copy(), p0_ensemble[i].copy(), base_params)
            for i in range(N_ens)
        ]
        
        # Run parallel simulations
        max_workers = min(os.cpu_count(), N_ens)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(simulate_member_sensitivity_robust, args_list))

        # Efficiently unpack results
        model_outputs_z = np.array([r[0] for r in results])
        model_outputs_p = np.array([r[1] for r in results])
        model_outputs_d = np.array([r[2] for r in results])
        model_outputs_c = np.array([r[3] for r in results])

        # Use optimized ensemble vector packing
        X = enkf.pack_ensemble_vectors(param_ensemble, model_outputs_z, 
                                      model_outputs_p, model_outputs_d, model_outputs_c)

        # Get observations efficiently
        obs = obs_manager.get_observations(z_layers_meas[t], p_layers_meas[t],
                                         deposits_meas[t], compositions_meas[t])

        # **SIMPLIFIED GLOBAL ENKF UPDATE - NO LOCALIZATION**
        Y_full = X[:, num_params:]
        Y_all = Y_full[:, obs_idx]

        # Check for ensemble collapse
        X_std = np.std(X, axis=0)
        if np.any(X_std[:num_params] < 1e-12):
            print(f"Warning: Ensemble collapse detected at time {t}, applying perturbations")
            # Add perturbations to maintain diversity
            for i in range(N_ens):
                X[i, :num_params] += np.random.normal(0, param_rw_std * 2)

        # Vectorized anomaly computation
        X_mean = X.mean(axis=0)
        X_pert = X - X_mean
        Y_mean = Y_all.mean(axis=0)
        Y_pert = Y_all - Y_mean
        Nm1 = float(N_ens - 1)

        # Check for zero variance in observations
        Y_std = np.std(Y_all, axis=0)
        if np.any(Y_std < 1e-12):
            print(f"Warning: Zero observation variance detected at time {t}")
            continue

        S = Y_pert / np.sqrt(Nm1)

        # Robust matrix operations with regularization
        SR = S * obs_manager.R_inv
        M = np.eye(N_ens) + SR @ S.T

        # Pre-compute cross-covariance for parameters only
        Cxy_param = (X_pert[:, :num_params].T @ S) / np.sqrt(Nm1)

        # Use robust solver
        solve_M = enkf.robust_cholesky_solve(M, regularization=1e-4)

        # Parameter update with bounds checking
        for i in range(N_ens):
            d = obs - Y_all[i]
            v = obs_manager.R_inv * d
            Sv = S @ v
            alpha = solve_M(Sv)
            w = v - obs_manager.R_inv * (S.T @ alpha)
            delta_param = Cxy_param @ w
            
            # Check for NaN values
            if np.any(~np.isfinite(delta_param)):
                print(f"Warning: Non-finite parameter update at time {t}, member {i}")
                delta_param = np.zeros_like(delta_param)
            
            # Apply update with random walk
            new_params = X[i, :num_params] + delta_param + np.random.normal(0.0, param_rw_std)
            
            # Apply parameter bounds
            for j, param_name in enumerate(recommended_params):
                if param_name.startswith('supply_comp_'):
                    new_params[j] = np.clip(new_params[j], 0.02, 0.8)
                elif 'coeff' in param_name:
                    new_params[j] = np.clip(new_params[j], 0.01, 1.0)
                elif 'scale' in param_name:
                    new_params[j] = np.clip(new_params[j], 5.0, 150.0)
            
            X[i, :num_params] = new_params

        # Apply inflation to maintain ensemble spread
        X_mean = X.mean(axis=0)
        X_pert = X - X_mean
        X = X_mean + inflation * X_pert

        # Write back to arrays
        param_ensemble = X[:, :num_params]
        
        z0_ensemble = X[:, enkf.offsets['z']:enkf.offsets['p']].reshape(N_ens, ny, nx)
        
        p0_ensemble = X[:, enkf.offsets['p']:enkf.offsets['deposits']].reshape(N_ens, ny, nx, 4)
        p0_ensemble = np.clip(p0_ensemble, 0.01, 0.99)
        p0_ensemble /= p0_ensemble.sum(axis=-1, keepdims=True) + 1e-12
        
        deposits_ensemble = X[:, enkf.offsets['deposits']:enkf.offsets['compositions']].reshape(N_ens, ny, nx)
        
        compositions_ensemble = X[:, enkf.offsets['compositions']:].reshape(N_ens, ny, nx, 4)
        compositions_ensemble = np.clip(compositions_ensemble, 0.01, 0.99)
        compositions_ensemble /= compositions_ensemble.sum(axis=-1, keepdims=True) + 1e-12

        # Store history
        param_history[t] = param_ensemble
        z0_history[t] = z0_ensemble
        p0_history[t] = p0_ensemble
        
        param_mean_history[t] = param_ensemble.mean(axis=0)
        param_std_history[t] = param_ensemble.std(axis=0)
        z_mean_history[t] = z0_ensemble.mean(axis=0)
        z_std_history[t] = z0_ensemble.std(axis=0)

        # Print diagnostics
        if t % 5 == 0:
            param_mean = param_ensemble.mean(axis=0)
            param_std = param_ensemble.std(axis=0)
            print(f"Time {t}: Params = {param_mean}, Stds = {param_std}")

    print("Robust EnKF iterations completed!")

    # Plotting code
    plt.figure(figsize=(15, 5))
    param_names_plot = recommended_params

    for i, name in enumerate(param_names_plot):
        plt.subplot(1, len(param_names_plot), i+1)
        
        # Plot ensemble members (reduced for performance)
        step = max(1, N_ens // 10)
        for j in range(0, N_ens, step):
            plt.plot(times[:min(nt, 20)], param_history[:min(nt, 20), j, i], 'k-', alpha=0.1)
        
        # Plot ensemble mean
        mean_param = param_mean_history[:min(nt, 20), i]
        plt.plot(times[:min(nt, 20)], mean_param, 'r-', linewidth=2, label='Ensemble mean')
        
        # Plot true value
        plt.axhline(y=true_params[i], color='b', linestyle='--', label='True value')
        
        # Plot uncertainty bounds
        std_param = param_std_history[:min(nt, 20), i]
        plt.fill_between(times[:min(nt, 20)], mean_param - 2*std_param, mean_param + 2*std_param, 
                        color='r', alpha=0.2, label='±2σ')
        
        plt.xlabel('Time step')
        plt.ylabel(name)
        plt.title(f'Parameter: {name}')
        plt.legend()

    plt.tight_layout()
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/robust_parameter_convergence.png', dpi=300)
    plt.show()