# %%
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import numpy.linalg as LA
from enkf_helper import *
import matplotlib.pyplot as plt

# %% load the initial settings, parameters, and results from a reference run.
# Load the saved .npz file
if __name__ == '__main__':
    data = np.load("./data/reference_fsm_results.npz", allow_pickle=True)

    # Access variables by their keys
    nx = data["nx"]
    ny = data["ny"]
    nt = data["nt"]
    times = data["times"]
    z0 = data["z0"]
    p0 = data["p0"]
    hSL_vals = data["hSL_vals"] # sea level values over time are assumed known
    hSS_vals = data["hSS_vals"] # subsidence values over time are assumed known
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
    params = data["params"].item()  # .item() is needed for objects
    # Now we can use these variables in DA script


    # %% create initial ensembles for parameters and state
    # Here we assume we want to estimate 3 parameters: diffusion_coeff, erosion_rate, subsidence_scalar
    N_ens = 50  # ensemble size
    num_params = 3
    true_params = np.array([params.diffusion_coeff, params.erosion_rate, params.subsidence_scalar])
    print("True parameters:", true_params)
    param_stds = np.array([0.1, 0.04, 0.04])  # assumed std for each parameter
    # sample initial ensemble from normal distribution around true parameter values
    param_ensemble = np.empty((N_ens, num_params))
    for i in range(num_params):
        param_ensemble[:, i] = np.random.normal(true_params[i], param_stds[i], size=N_ens)

    # initial state ensemble from perturbing the initial bathymetry
    z0_std = 10.0  # std of bathymetry, can be tuned as needed.
    z0_ensemble = np.empty((N_ens, ny, nx))
    for i in range(N_ens):
        z0_ensemble[i] = np.random.normal(z0, z0_std)

    # initial surface proportion ensemble from perturbing the initial proportions
    # we need to ensure proportions remain valid (non-negative, sum to 1)
    p0_std = np.array([0.05, 0.05, 0.05, 0.05])
    p0_ensemble =  np.empty((N_ens, ny, nx, 4)) 
    for i in range(N_ens):
        # 明确给出 size，并沿最后一维归一化
        rnd = np.random.normal(p0, p0_std, size=(ny, nx, 4))
        rnd = np.clip(rnd, 0, 1)
        rnd /= rnd.sum(axis=-1, keepdims=True) + 1e-12
        p0_ensemble[i] = rnd

    # Initialize deposits ensemble
    deposits_ensemble = np.empty((N_ens, ny, nx))

    # Initialize compositions ensemble with proper shape
    compositions_ensemble =  np.empty((N_ens, ny, nx, 4)) 

    # Arrays to store history of parameter and state estimates
    param_history = np.empty((nt, N_ens, num_params))
    z0_history = np.empty((nt, N_ens, ny, nx))
    p0_history = np.empty((nt, N_ens, ny, nx, 4))
    deposits_history = np.empty((nt, N_ens, ny, nx))
    compositions_history = np.empty((nt, N_ens, ny, nx, 4))


    # Choose observation usage (can be adjusted)
    obs_idx = build_obs_index(ny, nx,
                              stride_z=16,   # one in every 16 grid points
                              stride_p=16,   # one in every 16 grid points
                              stride_d=16,   # deposits every 16 grid points
                              stride_c=16,  # compositions every 16 grid points
                              use_z=True, use_p=True, use_d=True, use_c=True)

    # ------- Localized update for grid states -------
    # Precompute obs meta (positions for kept obs)
    oy, ox, otype, ocomp, cols = build_obs_meta(
        ny, nx, obs_idx,
        stride_z=16, stride_p=16, stride_d=16, stride_c=16,
        use_z=True, use_p=True, use_d=True, use_c=True
    )

    pos_map = {int(a): i for i, a in enumerate(obs_idx)}
    if cols.size and (cols.max() >= len(obs_idx) or cols.min() < 0):
        cols = np.array([pos_map[int(a)] for a in cols], dtype=int)

    # Sanity
    assert cols.max() < len(obs_idx)

    inflation = 1.05   # optional: multiplicative inflation factor, 1.0 means off


    # %% start the EnKF iterations (optimized: parallel forward runs + ensemble-space EnKF)
    for t in range(nt):
        current_time = times[t:t+1]
        
        # Prepare arguments for parallel execution
        args_list = [
            (i, nx, ny, current_time, hSL_vals[t], hSS_vals[t],
             param_ensemble[i].copy(), z0_ensemble[i].copy(), p0_ensemble[i].copy())
            for i in range(N_ens)
        ]
        
        # Run parallel simulations
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
            results = list(ex.map(simulate_member, args_list))

        # Unpack results into arrays
        model_outputs_z_layers = np.empty((N_ens, ny, nx))
        model_outputs_p_layers = np.empty((N_ens, ny, nx, 4))
        model_outputs_deposits = np.empty((N_ens, ny, nx))
        model_outputs_compositions = np.empty((N_ens, ny, nx, 4))
        for i, (zl, pl, dep, comp) in enumerate(results):
            model_outputs_z_layers[i] = zl
            model_outputs_p_layers[i] = pl
            model_outputs_deposits[i] = dep
            model_outputs_compositions[i] = comp

        # Build state vectors (params + model outputs at time t)
        ensemble_vectors = np.empty((N_ens, num_params + nx*ny + nx*ny*4 + nx*ny + nx*ny*4), dtype=float)
        for i in range(N_ens):
            offset = 0
            # parameters + z_layers + p_layers + deposits + compositions
            # parameters are not in the observations, but we include them for joint update
            ensemble_vectors[i, offset:offset+num_params] = param_ensemble[i]; offset += num_params
            ensemble_vectors[i, offset:offset+nx*ny] = model_outputs_z_layers[i].ravel(); offset += nx*ny
            ensemble_vectors[i, offset:offset+nx*ny*4] = model_outputs_p_layers[i].ravel(); offset += nx*ny*4
            ensemble_vectors[i, offset:offset+nx*ny] = model_outputs_deposits[i].ravel(); offset += nx*ny
            ensemble_vectors[i, offset:offset+nx*ny*4] = model_outputs_compositions[i].ravel()

        # 2) Observations at this time step
        obs_z_layers = z_layers_meas[t]         # (ny, nx)
        obs_p_layers = p_layers_meas[t]         # (ny, nx, 4)
        obs_deposits = deposits_meas[t]         # (ny, nx)
        obs_compositions = compositions_meas[t] # (ny, nx, 4)

        obs_full = np.concatenate([
            obs_z_layers.ravel(),
            obs_p_layers.ravel(),
            obs_deposits.ravel(),
            obs_compositions.ravel(),
        ])

        #* Subselect observations
        obs = obs_full[obs_idx]

        off_z = 0
        off_p = off_z + ny*nx
        off_d = off_p + ny*nx*4
        off_c = off_d + ny*nx

        R_full = np.empty_like(obs_full, dtype=float)
        R_full[off_z:off_z+ny*nx] = sigma_z**2
        R_full[off_p:off_p+ny*nx*4] = sigma_p**2
        R_full[off_d:off_d+ny*nx] = sigma_d**2
        R_full[off_c:off_c+ny*nx*4] = sigma_c**2

        # 3) EnKF update in ensemble space (no gigantic obs covariance inversion)
        state_dim = ensemble_vectors.shape[1]
        # obs_start = num_params
        # obs_end = state_dim

        # Split state into X (full) and Y (observed part)
        X = ensemble_vectors                              # (N, m)
        # Now I make a simple assumption that all states except parameters are observed
        # This can be adjusted by the observation selection function. For example, only some locations are observed.
        
        # We'll use the per-time Y_full for fast slicing
        Y_full = ensemble_vectors[:, num_params:]  # (N, ny*nx*(1+4+1+4))
        Y_all = Y_full[:, obs_idx]                 # (N, p) for parameter global update

        # Perturbations (anomalies)
        X_mean = X.mean(axis=0)
        X_pert = X - X_mean
        Y_mean = Y_all.mean(axis=0)  # Use Y_all instead of Y
        Y_pert = Y_all - Y_mean      # Use Y_all instead of Y
        Nm1 = float(N_ens - 1)

        S = Y_pert / np.sqrt(Nm1)                  # (N, p)

        # Observation error variances (vector), avoid forming a huge diagonal matrix
        R_var = R_full[obs_idx] + 1e-8
        R_inv = 1.0 / R_var

        # Build M = I + S * R^{-1} * S^T, which is N x N (small)
        SR = S * R_inv  # column-wise scaling, broadcasting over p
        M = np.eye(N_ens) + SR @ S.T                       # (N, N)

        # Precompute cross-covariance factor Cxy = X_pert^T @ S / （N-1）
        Cxy_param = (X_pert[:, :num_params].T @ S) / np.sqrt(Nm1)     # (m, N), since S already has 1/sqrt(N-1)

        # Cholesky solve for numerical stability
        try:
            L = LA.cholesky(M)
            def solve_M(vec):
                # Solve M x = vec via Cholesky
                y = LA.solve(L, vec)
                return LA.solve(L.T, y)
        except LA.LinAlgError:
            # Fallback to generic solver
            def solve_M(vec):
                return LA.solve(M, vec)

        # Update each ensemble member without forming huge matrices: w = (P_yy + R)^{-1} d via Woodbury
        param_rw_std = np.array([0.003, 0.001, 0.001])  # tune small
        for i in range(N_ens):
            d = obs - Y_all[i]
            v = R_inv * d
            Sv = S @ v
            alpha = solve_M(Sv)
            w = v - R_inv * (S.T @ alpha)
            delta_param = Cxy_param @ w
            X[i, :num_params] += delta_param
            X[i, :num_params] += np.random.normal(0.0, param_rw_std)    



        # Local settings
        LOC_RADIUS = 8.0     # in grid cells
        UPDATE_STRIDE = 1    # update every cell; set 2/4 for faster run

        # Prepare convenience arrays
        X_mean = X.mean(axis=0)
        X_pert = X - X_mean

        for iy in range(0, ny, UPDATE_STRIDE):
                for ix in range(0, nx, UPDATE_STRIDE):
                    dy = oy - iy
                    dx = ox - ix
                    dist = np.sqrt(dy*dy + dx*dx)

                    rho = gaspari_cohn(dist, LOC_RADIUS)  # shape == len(cols)
                    mask = (rho > 1e-12)
                    if not np.any(mask):
                        continue

                    # 用 cols 将“保留观测集合”的布尔 mask 映射回 obs/Y_all 的列索引
                    idx_loc = cols[mask]                 # columns into obs = obs_full[obs_idx]
                    obs_loc = obs[idx_loc]               # (p_loc,)
                    Y_loc = Y_all[:, idx_loc]            # (N, p_loc)

                    # 局地异常并做taper
                    Y_loc_mean = Y_loc.mean(axis=0)
                    S_loc = (Y_loc - Y_loc_mean) / np.sqrt(Nm1)
                    S_loc *= np.sqrt(rho[mask][None, :])

                    # 复用全局 R_var 的子集，并用 ρ 做taper（等价于使用 R_eff^{-1} = ρ / R）
                    R_var_loc = R_var[idx_loc] + 1e-12
                    R_inv_loc = rho[mask] / R_var_loc

                    SR = S_loc * R_inv_loc
                    M = np.eye(N_ens) + SR @ S_loc.T
                    try:
                        L = LA.cholesky(M)
                        def solve_M(vec):
                            y_ = LA.solve(L, vec); return LA.solve(L.T, y_)
                    except LA.LinAlgError:
                        def solve_M(vec):
                            return LA.solve(M, vec)

                    state_idx = state_indices_for_cell(iy, ix, ny, nx, num_params)
                    Cxy_loc = (X_pert[:, state_idx].T @ S_loc) / np.sqrt(Nm1)

                    for i in range(N_ens):
                        d = obs_loc - Y_loc[i]
                        v = R_inv_loc * d
                        Sv = S_loc @ v
                        alpha = solve_M(Sv)
                        w = v - R_inv_loc * (S_loc.T @ alpha)
                        delta = Cxy_loc @ w
                        X[i, state_idx] += delta

        # Inflation step
        if inflation != 1.0:
            X_mean = X.mean(axis=0)
            X_pert = X - X_mean
            X = X_mean + inflation * X_pert

        # Write back to arrays (same as your code)
        ensemble_vectors = X
        for i in range(N_ens):
            idx0 = 0
            param_ensemble[i] = X[i, idx0:idx0+num_params]; idx0 += num_params
            z0_ensemble[i] = X[i, idx0:idx0+nx*ny].reshape(ny, nx); idx0 += nx*ny
            p0_ensemble[i] = X[i, idx0:idx0+nx*ny*4].reshape(ny, nx, 4); idx0 += nx*ny*4
            p0_ensemble[i] = np.clip(p0_ensemble[i], 0, 1)
            p0_ensemble[i] /= p0_ensemble[i].sum(axis=-1, keepdims=True) + 1e-12
            deposits_ensemble[i] = X[i, idx0:idx0+nx*ny].reshape(ny, nx); idx0 += nx*ny
            compositions_ensemble[i] = X[i, idx0:idx0+nx*ny*4].reshape(ny, nx, 4)
            compositions_ensemble[i] = np.clip(compositions_ensemble[i], 0, 1)
            compositions_ensemble[i] /= compositions_ensemble[i].sum(axis=-1, keepdims=True) + 1e-12

        # Store the updated values for analysis
        param_history[t] = param_ensemble
        z0_history[t] = z0_ensemble
        p0_history[t] = p0_ensemble

        # Diagnostics
        param_mean = param_ensemble.mean(axis=0)
        param_std = param_ensemble.std(axis=0)
        print(f"Time step {t}")
        print(f"Parameter means: {param_mean}")
        print(f"Parameter stds: {param_std}")

        # Parameter convergence over time
    plt.figure(figsize=(15, 5))
    param_names = ["diffusion_coeff", "erosion_rate", "subsidence_scalar"]
    true_params = np.array([params.diffusion_coeff, params.erosion_rate, params.subsidence_scalar])

    for i, name in enumerate(param_names):
        plt.subplot(1, 3, i+1)
        
        # Plot ensemble members
        for j in range(N_ens):
            plt.plot(times, param_history[:, j, i], 'k-', alpha=0.1)
        
        # Plot ensemble mean
        mean_param = param_history.mean(axis=1)[:, i]
        plt.plot(times, mean_param, 'r-', linewidth=2, label='Ensemble mean')
        
        # Plot true value
        plt.axhline(y=true_params[i], color='b', linestyle='--', label='True value')
        
        # Plot uncertainty bounds (mean ± 2*std)
        std_param = param_history.std(axis=1)[:, i]
        plt.fill_between(times, mean_param - 2*std_param, mean_param + 2*std_param, 
                        color='r', alpha=0.2, label='±2σ')
        
        plt.xlabel('Time step')
        plt.ylabel(name)
        plt.title(f'Parameter: {name}')
        plt.legend()

    plt.tight_layout()
    plt.savefig('./plots/parameter_convergence.png', dpi=300)
    plt.show()

    # RMSE evolution over time
    plt.figure(figsize=(12, 6))

    # Parameter RMSE
    param_mean = param_history.mean(axis=1)  # shape: (nt, 3)
    param_rmse = np.sqrt(np.mean((param_mean - true_params)**2, axis=1))
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

    # Parameter correlation at final time step
    plt.figure(figsize=(12, 4))

    # Get final ensemble
    final_params = param_history[-1]  # shape: (N_ens, 3)

    # Plot pairwise scatter plots - use simple indexing
    subplot_idx = 1
    for i in range(2):
        for j in range(i+1, 3):
            plt.subplot(1, 3, subplot_idx)  # Use sequential subplot index
            subplot_idx += 1
            
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