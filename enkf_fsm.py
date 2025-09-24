# %%
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import numpy.linalg as LA
from enkf_helper import *
import matplotlib.pyplot as plt
from numba import jit, prange
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


class HybridEnKF:
    """Hybrid EnKF: Simple approach with minimal robustness improvements"""
    
    def __init__(self, N_ens, num_params, nx, ny):
        self.N_ens = N_ens
        self.num_params = num_params
        self.nx, self.ny = nx, ny
        
        # Pre-allocate main arrays (simplified)
        self.state_dim = num_params + nx*ny + nx*ny*4 + nx*ny + nx*ny*4
        
    def simple_cholesky_solve(self, M, max_jitter=1e-6):
        """Robust solver with multiple fallbacks"""
        # Check matrix condition first
        if np.any(np.isnan(M)) or np.any(np.isinf(M)):
            print("Warning: Invalid values in matrix M, using identity")
            def solve_M(vec):
                return vec
            return solve_M
        
        # Try progressively stronger regularization
        jitter_levels = [0, max_jitter, max_jitter*10, max_jitter*100, max_jitter*1000]
        
        for jitter in jitter_levels:
            try:
                if jitter == 0:
                    M_reg = M
                else:
                    M_reg = M + np.eye(M.shape[0]) * jitter
                
                # Try Cholesky decomposition
                L = LA.cholesky(M_reg)
                def solve_M(vec):
                    y = LA.solve(L, vec)
                    return LA.solve(L.T, y)
                if jitter > 0:
                    print(f"Warning: Used regularization {jitter:.2e}")
                return solve_M
                
            except LA.LinAlgError:
                continue
        
        # If all else fails, use pseudo-inverse
        print("Warning: Using pseudo-inverse solver due to singular matrix")
        def solve_M(vec):
            try:
                return LA.pinv(M) @ vec
            except:
                # Ultimate fallback: return zero update
                print("Warning: Complete solver failure, returning zero update")
                return np.zeros_like(vec)
        return solve_M
    
    def check_ensemble_health(self, X):
        """Simple ensemble health check"""
        # Check for NaN
        if np.any(np.isnan(X)):
            print("Warning: NaN detected in ensemble")
            return False
        
        # Check for collapse (more reasonable threshold)
        stds = X.std(axis=0)
        # Only check parameters for collapse, not all state variables
        param_stds = stds[:2]  # First 2 are parameters
        if np.any(param_stds < 1e-6):
            print(f"Warning: Parameter ensemble collapse detected. Stds: {param_stds}")
            return False
            
        return True



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

# -------------------- ETKF (ensemble-space) helper --------------------

def etkf_update_full_state(X,                 # (N_ens, n_state)
                           obs_vals,          # (p_subset,)
                           obs_col_idx,       # (p_subset,) indices into the **state part** [num_params:]
                           num_params,        # int
                           R_var_subset,      # (p_subset,) observation variances for the subset
                           inflation=1.0):
    """Deterministic ETKF in ensemble space, updating the *entire* state.
    - X contains [params | state] laid out exactly as assembled below.
    - obs_col_idx indexes columns of X[:, num_params:] to form Y.
    Returns updated X with same shape.
    """
    import numpy as _np
    from numpy.linalg import inv as _inv
    from scipy.linalg import sqrtm as _sqrtm

    Ne, n = X.shape
    # Forecast mean/anomalies in (n, Ne)
    Xf = X.T                        # (n, Ne)
    xbar_f = Xf.mean(axis=1, keepdims=True)
    Af = Xf - xbar_f                # (n, Ne)

    if inflation != 1.0:
        Af *= inflation

    # Build observation ensemble Y = H(X) using selected columns of the state part
    # Y_raw has shape (Ne, p)
    Y_raw = X[:, num_params:][:, obs_col_idx]
    ybar = Y_raw.mean(axis=0, keepdims=True)    # (1, p)
    Df = (Y_raw - ybar).T                       # (p, Ne) anomalies in obs space

    # Ensemble-space matrices
    # C = (Ne-1) I + Df^T R^{-1} Df   [shape (Ne, Ne)]
    Nm1 = float(Ne - 1)
    Rinvd = 1.0 / (R_var_subset + 1e-12)        # diagonal R^{-1}
    # scale Df rows by sqrt(R^{-1}) without forming big matrices
    Df_scaled = Df * Rinvd[:, None]            # (p, Ne)
    C = (Nm1) * _np.eye(Ne) + Df.T @ Df_scaled  # (Ne, Ne)

    # Innovation of the mean
    innov = (obs_vals - ybar.ravel())          # (p,)
    rhs = Df.T @ (Rinvd * innov)               # (Ne,)

    # Solve for weights and transform
    try:
        Cinv = _inv(C)
    except _np.linalg.LinAlgError:
        # mild ridge if needed
        C = C + 1e-6 * _np.eye(Ne)
        Cinv = _inv(C)

    w_bar = Cinv @ rhs[:, None]                 # (Ne,1)
    T = _sqrtm(Nm1 * Cinv).real                 # (Ne,Ne)

    # Analysis mean/anomalies in state space
    xa_bar = xbar_f + Af @ w_bar                # (n,1)
    Aa = Af @ T                                 # (n,Ne)

    Xa = (xa_bar + Aa).T                        # back to (Ne, n)
    return Xa

# -------------------- RTPS inflation (parameters) --------------------

def rtp_inflate_params(Xa, Xf, num_params, alpha=0.5):
    """Relax-To-Prior-Spread (RTPS) on the first num_params columns only.
    Xa, Xf: (N_ens, n_state). alpha in [0,1].
    """
    am = Xa.mean(axis=0, keepdims=True)
    fm = Xf.mean(axis=0, keepdims=True)
    aA = Xa - am
    fA = Xf - fm
    sa = np.std(aA[:, :num_params], axis=0, ddof=1) + 1e-12
    sf = np.std(fA[:, :num_params], axis=0, ddof=1)
    factor = 1.0 + alpha * (sf / sa - 1.0)
    aA[:, :num_params] *= factor
    return am + aA

def simulate_member(args):
    """Simple simulation function (based on working version)"""
    i, nx, ny, current_time, hSL_val, hSS_val, param_values, z0_i, p0_i = args
    
    try:
        # Create simple parameter object (hardcoded approach)
        from model import ForwardParams
        
        # Use 2 sensitive parameters (diffusion_coeff, thickness_scale)
        params_i = ForwardParams(
            nx=nx, ny=ny, times=current_time,
            diffusion_coeff=param_values[0],
            thickness_scale=param_values[1],
            # Use default values for other parameters
            erosion_rate=0.1,  # Default value
            subsidence_scalar=0.1,  # Default value
            diffusion_iters=6,
            bowl_amp=0.5,
            bowl_center_east=0.20,
            bowl_center_north=0.20,
            bowl_width_east=0.22,
            bowl_width_north=0.50
        )

        # Run forward model
        z_layers, p_layers, deposits, compositions = run_forward(
            z0_i, p0_i, current_time, [hSL_val], current_time, [hSS_val], params_i
        )

        return z_layers[-1], p_layers[-1], deposits[-1], compositions[-1]
        
    except Exception as e:
        print(f"Error in ensemble member {i}: {e}")
        # Return fallback values
        return z0_i, p0_i, np.zeros_like(z0_i), np.zeros((z0_i.shape[0], 
                                                           z0_i.shape[1], 4))


# Load the initial settings, parameters, and results from a reference run.
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

    # SENSITIVITY-BASED APPROACH: Use parameters recommended by sensitivity analysis
    print("Using sensitivity-based approach with recommended parameters...")
    
    # Use 2 most sensitive parameters based on global sensitivity analysis
    N_ens = 200
    num_params = 2
    param_names = ["diffusion_coeff", "thickness_scale"]
    true_params = np.array([params.diffusion_coeff, params.thickness_scale])
    param_stds = np.array([0.1, 10.0])  # Appropriate uncertainties for each parameter
    
    print(f"Selected parameters: {param_names}")
    print(f"True values: {true_params}")
    print(f"Standard deviations: {param_stds}")

    # Simple ensemble initialization (like working version)
    param_ensemble = np.empty((N_ens, num_params))
    for i in range(num_params):
        param_ensemble[:, i] = np.random.normal(true_params[i], param_stds[i], size=N_ens)
    
    # Check initial ensemble diversity
    print("Initial parameter ensemble statistics:")
    for i, name in enumerate(param_names):
        mean_val = param_ensemble[:, i].mean()
        std_val = param_ensemble[:, i].std()
        print(f"  {name}: mean={mean_val:.4f}, std={std_val:.4f}")
        if std_val < 1e-6:
            print(f"    WARNING: Very small std for {name}")
    print(f"Parameter ensemble shape: {param_ensemble.shape}")

    # Simple state initialization
    z0_std = 10.0
    z0_ensemble = np.empty((N_ens, ny, nx))
    for i in range(N_ens):
        z0_ensemble[i] = np.random.normal(z0, z0_std)

    p0_std = np.array([0.05, 0.05, 0.05, 0.05])
    p0_ensemble = np.empty((N_ens, ny, nx, 4))
    for i in range(N_ens):
        rnd = np.random.normal(p0, p0_std, size=(ny, nx, 4))
        rnd = np.clip(rnd, 0, 1)
        rnd /= rnd.sum(axis=-1, keepdims=True) + 1e-12
        p0_ensemble[i] = rnd

    # Initialize arrays
    param_history = np.empty((nt, N_ens, num_params))
    z0_history = np.empty((nt, N_ens, ny, nx))
    p0_history = np.empty((nt, N_ens, ny, nx, 4))

    # Observation setup (simplified)
    obs_idx = build_obs_index(ny, nx,
                              stride_z=16, stride_p=16, stride_d=16, stride_c=16,
                              use_z=True, use_p=True, use_d=True, use_c=True)
    
    obs_manager = ObservationManager(ny, nx, obs_idx, sigma_z, sigma_p, sigma_d, sigma_c)

    # Precompute observation metadata
    oy, ox, otype, ocomp, cols = build_obs_meta(
        ny, nx, obs_idx,
        stride_z=16, stride_p=16, stride_d=16, stride_c=16,
        use_z=True, use_p=True, use_d=True, use_c=True
    )

    pos_map = {int(a): i for i, a in enumerate(obs_idx)}
    if cols.size and (cols.max() >= len(obs_idx) or cols.min() < 0):
        cols = np.array([pos_map[int(a)] for a in cols], dtype=int)

    # Initialize hybrid EnKF
    enkf = HybridEnKF(N_ens, num_params, nx, ny)
    
    # Simple settings
    inflation = 1.02  # Low inflation for parameter stability
    param_rw_std = np.array([0.001, 0.1])  # Very small random walk for parameters

    print("Starting hybrid EnKF iterations...")
    
    for t in range(nt):
        if t % 10 == 0:
            print(f"Processing time step {t}/{nt}")
            
        current_time = times[t:t+1]

        # Parameter random walk to maintain informative spread
        param_ensemble += np.random.normal(0.0, param_rw_std, size=param_ensemble.shape)

        # Prepare arguments for parallel execution (simplified)
        args_list = [
            (i, nx, ny, current_time, hSL_vals[t], hSS_vals[t],
             param_ensemble[i].copy(), z0_ensemble[i].copy(), p0_ensemble[i].copy())
            for i in range(N_ens)
        ]
        
        # Run parallel simulations
        with ProcessPoolExecutor(max_workers=min(4, N_ens)) as ex:
            results = list(ex.map(simulate_member, args_list))

        # Unpack results (simple approach)
        model_outputs_z = np.array([r[0] for r in results])
        model_outputs_p = np.array([r[1] for r in results])
        model_outputs_d = np.array([r[2] for r in results])
        model_outputs_c = np.array([r[3] for r in results])

        # Build ensemble vectors (like working version)
        state_size = num_params + nx*ny + nx*ny*4 + nx*ny + nx*ny*4
        X = np.empty((N_ens, state_size), dtype=float)
        for i in range(N_ens):
            offset = 0
            X[i, offset:offset+num_params] = param_ensemble[i]
            offset += num_params
            X[i, offset:offset+nx*ny] = model_outputs_z[i].ravel()
            offset += nx*ny
            X[i, offset:offset+nx*ny*4] = model_outputs_p[i].ravel()
            offset += nx*ny*4
            X[i, offset:offset+nx*ny] = model_outputs_d[i].ravel()
            offset += nx*ny
            X[i, offset:offset+nx*ny*4] = model_outputs_c[i].ravel()

        # Check ensemble health
        if not enkf.check_ensemble_health(X):
            print(f"Ensemble health issues at time {t}, skipping update")
            continue

        # ---------------- ETKF update (ensemble-space, full-state) ----------------
        # Observations for this time step
        obs = obs_manager.get_observations(z_layers_meas[t], p_layers_meas[t],
                                           deposits_meas[t], compositions_meas[t])

        # Randomized obs subset to avoid systematic loss of information
        keep = max(1, int(0.5 * len(obs_idx)))
        obs_subset_indices = np.random.choice(len(obs_idx), size=keep, replace=False)

        # Build mapping from state to obs: we observe columns in X[:, num_params:]
        obs_subset_vals = obs[obs_subset_indices]
        R_var_subset = obs_manager.R_var[obs_subset_indices]

        # Map subset indices to actual columns of the flattened state-part
        # of X (z | p | d | c), which matches how obs_idx was built.
        obs_col_idx = obs_idx[obs_subset_indices]

        if t % 50 == 0:
            # Innovation variance vs R
            Yf_mean = X[:, num_params:][:, obs_col_idx].mean(axis=0)
            innov = obs_subset_vals - Yf_mean
            ratio = np.var(innov) / (np.mean(R_var_subset) + 1e-12)
            print(f"t={t}: innov_var/mean_R ~ {ratio:.3f}")

            # Max abs correlation between each parameter and any obs
            Apar = X[:, :num_params] - X[:, :num_params].mean(axis=0)
            Ay   = X[:, num_params:][:, obs_col_idx] - Yf_mean
            cov  = (Apar.T @ Ay) / (X.shape[0]-1)
            sp   = Apar.std(axis=0, ddof=1) + 1e-12
            so   = Ay.std(axis=0, ddof=1) + 1e-12
            corr = cov / (sp[:, None] * so[None, :])
            for k, name in enumerate(param_names):
                print(f"t={t}: max|corr({name}, obs)| = {np.max(np.abs(corr[k])):.3f}")

        # Save forecast ensemble vector before analysis (for RTPS)
        Xf_save = X.copy()

        # Run deterministic ETKF to update the entire ensemble state (params + fields)
        X = etkf_update_full_state(
            X,
            obs_subset_vals,
            obs_col_idx,
            num_params,
            R_var_subset,
            inflation=inflation,
        )

        # Apply RTPS inflation on parameters to prevent collapse
        X = rtp_inflate_params(X, Xf_save, num_params, alpha=0.5)

        # ---------------- Constraints & write-back ----------------
        # Enforce parameter bounds
        X[:, 0] = np.clip(X[:, 0], 0.05, 1.0)     # diffusion_coeff
        X[:, 1] = np.clip(X[:, 1], 10.0, 80.0)    # thickness_scale

        # Write back to structured arrays
        param_ensemble = X[:, :num_params]
        idx0 = num_params
        for i in range(N_ens):
            z0_ensemble[i] = X[i, idx0:idx0+nx*ny].reshape(ny, nx)
            p_end = idx0 + nx*ny + nx*ny*4
            p0_ensemble[i] = X[i, idx0+nx*ny:p_end].reshape(ny, nx, 4)
            p0_ensemble[i] = np.clip(p0_ensemble[i], 0, 1)
            p_sum = p0_ensemble[i].sum(axis=-1, keepdims=True) + 1e-12
            p0_ensemble[i] /= p_sum

        # Store results
        param_history[t] = param_ensemble
        z0_history[t] = z0_ensemble
        p0_history[t] = p0_ensemble

        # Diagnostics (optional)
        if t % 20 == 0:
            param_mean = param_ensemble.mean(axis=0)
            param_std = param_ensemble.std(axis=0)
            print(f"Time {t}: Params = {param_mean}, Stds = {param_std}")

    print("Hybrid EnKF iterations completed!")

    # Create missing variables for plotting
    param_mean_history = np.empty((nt, num_params))
    param_std_history = np.empty((nt, num_params))
    
    # Fill the history arrays
    for t in range(nt):
        param_mean_history[t] = param_history[t].mean(axis=0)
        param_std_history[t] = param_history[t].std(axis=0)

    # Update the plotting section
    plt.figure(figsize=(15, 5))
    
    for i, name in enumerate(param_names):
        plt.subplot(1, len(param_names), i+1)  # Dynamic subplot count
        
        # Plot ensemble members (reduced for performance)
        step = max(1, N_ens // 20)  # Show max 20 lines
        for j in range(0, N_ens, step):
            plt.plot(times, param_history[:, j, i], 'k-', alpha=0.1)
        
        # Plot ensemble mean
        mean_param = param_mean_history[:, i]
        plt.plot(times, mean_param, 'r-', linewidth=2, label='Ensemble mean')
        
        # Plot true value
        plt.axhline(y=true_params[i], color='b', linestyle='--', label='True value')
        
        # Plot uncertainty bounds
        std_param = param_std_history[:, i]
        plt.fill_between(times, mean_param - 2*std_param, mean_param + 2*std_param, 
                        color='r', alpha=0.2, label='±2σ')
        
        plt.xlabel('Time step')
        plt.ylabel(name)
        plt.title(f'Parameter: {name}')
        plt.legend()

    plt.tight_layout()
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/parameter_convergence.png', dpi=300)
    plt.show()

    # Update RMSE calculation
    param_mean = param_history.mean(axis=1)  # shape: (nt, num_params)
    param_rmse = np.sqrt(np.mean((param_mean - true_params)**2, axis=1))

    # RMSE evolution over time
    plt.figure(figsize=(12, 6))

    # Parameter RMSE
    plt.subplot(1, 2, 1)
    plt.plot(times, param_rmse, 'b-', linewidth=2)
    plt.xlabel('Time step')
    plt.ylabel('Parameter RMSE')
    plt.title('Parameter estimation error')
    plt.grid(True, alpha=0.3)

    # State RMSE (elevation field)
    z_mean = z0_history.mean(axis=1)  # shape: (nt, ny, nx)
    z_rmse = np.sqrt(np.mean((z_mean - z_layers)**2, axis=(1, 2)))
    plt.subplot(1, 2, 2)
    plt.plot(times, z_rmse, 'g-', linewidth=2)
    plt.xlabel('Time step')
    plt.ylabel('Elevation RMSE (m)')
    plt.title('State estimation error')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./plots/rmse_evolution.png', dpi=300)
    plt.show()

    # Parameter correlation at final time step - make it dynamic
    if num_params > 1:
        n_plots = num_params * (num_params - 1) // 2  # Number of pairwise plots
        n_cols = min(3, n_plots)  # Maximum 3 columns
        n_rows = (n_plots + n_cols - 1) // n_cols  # Calculate rows needed
        
        plt.figure(figsize=(4 * n_cols, 4 * n_rows))
        
        # Get final ensemble
        final_params = param_history[-1]  # shape: (N_ens, num_params)
        
        plot_idx = 1
        for i in range(num_params):
            for j in range(i+1, num_params):
                plt.subplot(n_rows, n_cols, plot_idx)
                plot_idx += 1
                
                plt.scatter(final_params[:, i], final_params[:, j], alpha=0.7)
                plt.xlabel(param_names[i])
                plt.ylabel(param_names[j])
                plt.grid(True, alpha=0.3)
                
                # Plot true values
                plt.axvline(x=true_params[i], color='r', linestyle='--')
                plt.axhline(y=true_params[j], color='r', linestyle='--')
                
                # Compute correlation
                corr = np.corrcoef(final_params[:, i], final_params[:, j])[0, 1]
                plt.title(f'Correlation: {corr:.2f}')
        
        plt.tight_layout()
        plt.savefig('./plots/parameter_correlation.png', dpi=300)
        plt.show()
    else:
        print("Only one parameter selected, skipping correlation plot")

    # Compare true vs. estimated states at final time step
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # True final field
    im0 = axes[0].imshow(z_layers[-1], origin='lower')
    axes[0].set_title('True final elevation')
    plt.colorbar(im0, ax=axes[0])

    # Estimated mean final field
    z_final_mean = z0_history[-1].mean(axis=0)
    im1 = axes[1].imshow(z_final_mean, origin='lower')
    axes[1].set_title('Estimated mean elevation')
    plt.colorbar(im1, ax=axes[1])

    # Difference
    diff = z_final_mean - z_layers[-1]
    im2 = axes[2].imshow(diff, origin='lower', cmap='RdBu_r')
    axes[2].set_title('Difference (estimate - true)')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig('./plots/state_comparison.png', dpi=300)
    plt.show()

    # Plot reduction in parameter uncertainty
    plt.figure(figsize=(10, 6))

    for i, name in enumerate(param_names):
        # Initial spread
        initial_std = param_history[0, :, i].std()
        
        # Evolution of spread
        spread = param_history[:, :, i].std(axis=1)
        
        # Normalize by initial spread
        rel_spread = spread / initial_std
        
        plt.plot(times, rel_spread, '-o', label=name)

    plt.xlabel('Time step')
    plt.ylabel('Relative ensemble spread')
    plt.title('Reduction in parameter uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./plots/uncertainty_reduction.png', dpi=300)
    plt.show()
