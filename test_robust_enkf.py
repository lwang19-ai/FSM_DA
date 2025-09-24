#!/usr/bin/env python3
"""
Robust EnKF with better parameter constraints
"""
import numpy as np
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
import copy
from model import run_forward
from sensitivity_analysis import SensitivityAnalyzer

def apply_parameter_constraints(param_value, param_name):
    """Apply physical constraints to a single parameter"""
    # Check for non-finite values first
    if not np.isfinite(param_value):
        print(f"Warning: Non-finite value for {param_name}: {param_value}")
        return 1e-6  # Safe fallback
    
    if param_name.startswith('supply_comp_'):
        # Supply composition: must be positive and ≤ 1
        return np.clip(param_value, 1e-8, 1.0)
    elif 'diffusion_coeff' in param_name:
        # Diffusion coefficient: positive, reasonable range
        return np.clip(param_value, 1e-8, 1e2)
    elif 'erosion_rate' in param_name:
        # Erosion rate: positive, reasonable range
        return np.clip(param_value, 1e-8, 10.0)
    elif 'subsidence' in param_name:
        # Subsidence: can be negative, but reasonable range
        return np.clip(param_value, -1e2, 1e2)
    elif 'rate' in param_name:
        # General rates: positive
        return np.clip(param_value, 1e-8, 1e2)
    elif 'coeff' in param_name:
        # General coefficients: positive
        return np.clip(param_value, 1e-8, 1e3)
    elif 'scalar' in param_name:
        # Scalars: positive
        return np.clip(param_value, 1e-8, 1e2)
    elif 'thickness' in param_name:
        # Thickness: positive
        return np.clip(param_value, 1e-8, 1e4)
    else:
        # Unknown parameters: apply basic positivity
        return np.clip(param_value, 1e-8, None)

def constrain_parameter_ensemble(param_ensemble, param_names):
    """Apply constraints to entire parameter ensemble"""
    constrained_ensemble = param_ensemble.copy()
    
    for j, param_name in enumerate(param_names):
        original_values = param_ensemble[:, j]
        constrained_values = np.array([
            apply_parameter_constraints(val, param_name) 
            for val in original_values
        ])
        
        # Check how many values were significantly changed
        significant_changes = np.abs(original_values - constrained_values) > 0.1 * np.abs(original_values)
        if np.any(significant_changes):
            n_changed = np.sum(significant_changes)
            print(f"Parameter {param_name}: {n_changed}/{len(original_values)} values constrained")
            print(f"  Original range: [{original_values.min():.6f}, {original_values.max():.6f}]")
            print(f"  Constrained range: [{constrained_values.min():.6f}, {constrained_values.max():.6f}]")
        
        constrained_ensemble[:, j] = constrained_values
    
    return constrained_ensemble

def simulate_member_robust(args):
    """Robust simulation with detailed error reporting"""
    i, nx, ny, current_time, hSL_val, hSS_val, selected_params, param_names, z0_i, p0_i, base_params = args
    
    try:
        # Create a copy of the base params
        params_i = copy.deepcopy(base_params)
        
        # Apply constraints and update parameters
        constrained_params = []
        for j, param_name in enumerate(param_names):
            original_value = selected_params[j]
            constrained_value = apply_parameter_constraints(original_value, param_name)
            constrained_params.append(constrained_value)
            
            # Report significant constraint applications
            if abs(original_value - constrained_value) > 0.1 * abs(original_value):
                print(f"Member {i}: {param_name} constrained {original_value:.6f} → {constrained_value:.6f}")
        
        # Update parameters in the model
        for j, param_name in enumerate(param_names):
            constrained_value = constrained_params[j]
            
            if param_name.startswith('supply_comp_'):
                comp_idx = int(param_name.split('_')[-1])
                params_i.supply_composition[comp_idx] = constrained_value
                # Renormalize after all supply components are set
            else:
                setattr(params_i, param_name, constrained_value)
        
        # Renormalize supply composition if any supply components were updated
        supply_comp_updated = any(name.startswith('supply_comp_') for name in param_names)
        if supply_comp_updated:
            total = params_i.supply_composition.sum()
            if total > 0:
                params_i.supply_composition /= total
            else:
                # Fallback to equal distribution
                params_i.supply_composition[:] = 1.0 / len(params_i.supply_composition)
                print(f"Member {i}: Supply composition normalized to equal distribution")

        # Check parameter values before running model
        param_check_failed = False
        for j, param_name in enumerate(param_names):
            if param_name.startswith('supply_comp_'):
                comp_idx = int(param_name.split('_')[-1])
                val = params_i.supply_composition[comp_idx]
            else:
                val = getattr(params_i, param_name)
            
            if not np.isfinite(val) or val <= 0:
                print(f"Member {i}: Invalid {param_name} = {val}")
                param_check_failed = True
        
        if param_check_failed:
            raise ValueError("Parameter validation failed")

        # Run forward model
        z_layers, p_layers, deposits, compositions = run_forward(
            z0_i, p0_i, current_time, [hSL_val], current_time, [hSS_val], params_i
        )

        z_last = z_layers[-1]
        p_last = p_layers[-1]
        d_last = deposits[-1]
        c_last = compositions[-1]

        # Check outputs
        outputs_finite = (
            np.all(np.isfinite(z_last)) and
            np.all(np.isfinite(p_last)) and
            np.all(np.isfinite(d_last)) and
            np.all(np.isfinite(c_last))
        )
        
        if not outputs_finite:
            print(f"Member {i}: Non-finite outputs detected")
            print(f"  z_last finite: {np.all(np.isfinite(z_last))}")
            print(f"  p_last finite: {np.all(np.isfinite(p_last))}")
            print(f"  d_last finite: {np.all(np.isfinite(d_last))}")
            print(f"  c_last finite: {np.all(np.isfinite(c_last))}")
            raise ValueError("Forward model produced non-finite outputs")

        return z_last, p_last, d_last, c_last
        
    except Exception as e:
        print(f"Error in ensemble member {i}: {e}")
        print(f"  Parameters: {dict(zip(param_names, selected_params))}")
        # Return fallback values
        return z0_i, p0_i, np.zeros_like(z0_i), np.zeros((z0_i.shape[0], z0_i.shape[1], 4))

if __name__ == '__main__':
    print("Loading data...")
    data = np.load("./data/reference_fsm_results.npz", allow_pickle=True)
    
    # Extract variables
    nx = data["nx"]
    ny = data["ny"]
    nt = data["nt"]
    times = data["times"]
    z0 = data["z0"]
    p0 = data["p0"]
    hSL_vals = data["hSL_vals"]
    hSS_vals = data["hSS_vals"]
    params = data["params"].item()
    
    print("Running sensitivity analysis...")
    analyzer = SensitivityAnalyzer(params, nx, ny, times, hSL_vals, hSS_vals)
    
    # Run sensitivity analysis
    samples, param_names_sa, outputs, output_names = analyzer.run_sensitivity_analysis(
        z0, p0, n_samples=50  # Reduced for testing
    )
    
    sensitivities = analyzer.compute_sobol_indices(samples, outputs)
    sensitivity_matrix, overall_sensitivity = analyzer.plot_sensitivity_results(
        param_names_sa, output_names, sensitivities
    )
    
    recommended_params = analyzer.recommend_parameters(
        param_names_sa, overall_sensitivity, threshold=0.1
    )
    
    print(f"Selected parameters: {recommended_params}")
    
    # Create parameter ensemble with proper constraints
    N_ens = 10  # Start small for testing
    param_values = []
    param_stds = []
    
    for param_name in recommended_params:
        if param_name.startswith('supply_comp_'):
            comp_idx = int(param_name.split('_')[-1])
            if hasattr(params, 'supply_composition') and comp_idx < len(params.supply_composition):
                val = params.supply_composition[comp_idx]
                std = 0.02 * val  # Reduce uncertainty for testing
                param_values.append(val)
                param_stds.append(std)
        elif hasattr(params, param_name):
            val = getattr(params, param_name)
            if 'coeff' in param_name or 'rate' in param_name:
                std = 0.05 * val  # Reduce uncertainty
            else:
                std = 0.02 * val
            param_values.append(val)
            param_stds.append(std)
    
    param_values = np.array(param_values)
    param_stds = np.array(param_stds)
    
    print(f"True parameter values: {param_values}")
    print(f"Parameter standard deviations: {param_stds}")
    
    # Generate ensemble
    param_ensemble = np.random.normal(
        param_values[None, :], param_stds[None, :], 
        size=(N_ens, len(param_values))
    )
    
    print("\\nApplying parameter constraints...")
    param_ensemble = constrain_parameter_ensemble(param_ensemble, recommended_params[:len(param_values)])
    
    # Initialize state ensemble
    z0_std = 5.0  # Reduced uncertainty
    z0_ensemble = np.random.normal(z0[None, :, :], z0_std, size=(N_ens, ny, nx))
    
    p0_std = np.array([0.02, 0.02, 0.02, 0.02])  # Reduced uncertainty
    p0_ensemble = np.random.normal(p0[None, None, None, :], p0_std[None, None, None, :], 
                                   size=(N_ens, ny, nx, 4))
    p0_ensemble = np.clip(p0_ensemble, 0, 1)
    p0_ensemble /= p0_ensemble.sum(axis=-1, keepdims=True) + 1e-12
    
    # Test first time step
    print("\\nTesting ensemble simulation for first time step...")
    t = 0
    current_time = times[t:t+1]
    
    # Prepare arguments
    args_list = [
        (i, nx, ny, current_time, hSL_vals[t], hSS_vals[t],
         param_ensemble[i].copy(), recommended_params[:len(param_values)], 
         z0_ensemble[i].copy(), p0_ensemble[i].copy(), params)
        for i in range(N_ens)
    ]
    
    # Run simulations
    print("Running ensemble simulations...")
    results = []
    for i, args in enumerate(args_list):
        print(f"  Running member {i+1}/{N_ens}")
        result = simulate_member_robust(args)
        results.append(result)
    
    # Check results
    success_count = sum(1 for r in results if not np.array_equal(r[0], args_list[0][8]))
    print(f"\\nSimulation completed: {success_count}/{N_ens} members successful")
    
    if success_count > 0:
        print("✓ Some ensemble members succeeded - parameter constraints are working!")
    else:
        print("✗ All ensemble members failed - need further investigation")