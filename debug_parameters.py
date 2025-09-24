#!/usr/bin/env python3
"""
Debug script to check parameter values and constraints
"""
import numpy as np
import sys
sys.path.append('.')

# Load data and parameters
data = np.load("./data/reference_fsm_results.npz", allow_pickle=True)
params = data["params"].item()

print("=== Parameter Debugging ===")
print(f"Parameters object type: {type(params)}")
print(f"Parameters attributes: {dir(params)}")

# Check all parameter values
print("\n=== Current Parameter Values ===")
for attr in dir(params):
    if not attr.startswith('_'):
        try:
            value = getattr(params, attr)
            if isinstance(value, (int, float, np.ndarray)):
                print(f"{attr}: {value}")
        except:
            print(f"{attr}: <could not access>")

# Test parameter generation
print("\n=== Testing Parameter Generation ===")
from sensitivity_analysis import SensitivityAnalyzer

try:
    analyzer = SensitivityAnalyzer(params, data["nx"], data["ny"], 
                                  data["times"], data["hSL_vals"], data["hSS_vals"])
    
    # Run a small sensitivity analysis
    samples, param_names_sa, outputs, output_names = analyzer.run_sensitivity_analysis(
        data["z0"], data["p0"], n_samples=10  # Small sample for debugging
    )
    
    sensitivities = analyzer.compute_sobol_indices(samples, outputs)
    sensitivity_matrix, overall_sensitivity = analyzer.plot_sensitivity_results(
        param_names_sa, output_names, sensitivities
    )
    
    recommended_params = analyzer.recommend_parameters(
        param_names_sa, overall_sensitivity, threshold=0.1
    )
    
    print(f"Recommended parameters: {recommended_params}")
    
    # Test parameter extraction and constraints
    print("\n=== Testing Parameter Extraction ===")
    param_values = []
    param_stds = []
    true_params = []
    
    for param_name in recommended_params:
        print(f"\nProcessing parameter: {param_name}")
        
        if param_name.startswith('supply_comp_'):
            comp_idx = int(param_name.split('_')[-1])
            if hasattr(params, 'supply_composition') and comp_idx < len(params.supply_composition):
                val = params.supply_composition[comp_idx]
                std = 0.05 * val
                print(f"  Supply component {comp_idx}: {val} ± {std}")
                param_values.append(val)
                param_stds.append(std)
                true_params.append(val)
            else:
                print(f"  ERROR: supply_composition[{comp_idx}] not accessible")
        elif hasattr(params, param_name):
            val = getattr(params, param_name)
            if 'coeff' in param_name or 'rate' in param_name:
                std = 0.1 * val
            else:
                std = 0.05 * val
            print(f"  {param_name}: {val} ± {std}")
            param_values.append(val)
            param_stds.append(std)
            true_params.append(val)
        else:
            print(f"  ERROR: {param_name} not found in params")
    
    # Test ensemble generation
    if param_values:
        print(f"\n=== Testing Ensemble Generation (N=10) ===")
        param_values = np.array(param_values)
        param_stds = np.array(param_stds)
        true_params = np.array(true_params)
        
        print(f"True values: {true_params}")
        print(f"Standard devs: {param_stds}")
        
        # Generate small ensemble
        N_test = 10
        param_ensemble = np.random.normal(true_params[None, :], param_stds[None, :], 
                                        size=(N_test, len(true_params)))
        
        print(f"\nEnsemble before constraints:")
        print(f"  Min values: {param_ensemble.min(axis=0)}")
        print(f"  Max values: {param_ensemble.max(axis=0)}")
        print(f"  Any negative: {(param_ensemble < 0).any(axis=0)}")
        print(f"  Any non-finite: {(~np.isfinite(param_ensemble)).any(axis=0)}")
        
        # Apply constraints
        for i, param_name in enumerate(recommended_params[:len(true_params)]):
            if param_name.startswith('supply_comp_'):
                param_ensemble[:, i] = np.clip(param_ensemble[:, i], 1e-8, 1.0)
            elif 'coeff' in param_name:
                param_ensemble[:, i] = np.clip(param_ensemble[:, i], 1e-8, 1e3)
            elif 'rate' in param_name:
                param_ensemble[:, i] = np.clip(param_ensemble[:, i], 1e-8, 1e2)
            elif 'scalar' in param_name:
                param_ensemble[:, i] = np.clip(param_ensemble[:, i], 1e-8, 1e2)
            elif 'thickness' in param_name:
                param_ensemble[:, i] = np.clip(param_ensemble[:, i], 1e-8, 1e4)
            else:
                param_ensemble[:, i] = np.clip(param_ensemble[:, i], 1e-8, None)
        
        print(f"\nEnsemble after constraints:")
        print(f"  Min values: {param_ensemble.min(axis=0)}")
        print(f"  Max values: {param_ensemble.max(axis=0)}")
        print(f"  Any negative: {(param_ensemble < 0).any(axis=0)}")
        print(f"  Any non-finite: {(~np.isfinite(param_ensemble)).any(axis=0)}")
    
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Debugging Complete ===")