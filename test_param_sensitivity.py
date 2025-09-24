#!/usr/bin/env python3
"""
Simplified EnKF test for parameter updating
"""

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from model import ForwardParams, run_forward
from enkf_helper import *

def test_parameter_sensitivity():
    """Test if parameters actually affect model output"""
    
    # Load reference data
    data = np.load("./data/reference_fsm_results.npz", allow_pickle=True)
    nx, ny, nt = data["nx"], data["ny"], data["nt"]
    times = data["times"]
    z0 = data["z0"]
    p0 = data["p0"]
    hSL_vals = data["hSL_vals"]
    hSS_vals = data["hSS_vals"]
    params_ref = data["params"].item()
    
    print("Testing parameter sensitivity...")
    print(f"Reference diffusion_coeff: {params_ref.diffusion_coeff}")
    print(f"Reference thickness_scale: {params_ref.thickness_scale}")
    
    # Test different parameter values
    test_values = [
        (0.1, 20.0),   # Low values
        (0.25, 40.0),  # Reference values
        (0.5, 60.0),   # High values
    ]
    
    results = []
    
    for diff_coeff, thick_scale in test_values:
        params_test = ForwardParams(
            nx=nx, ny=ny, times=times[:10],  # Only first 10 time steps
            diffusion_coeff=diff_coeff,
            thickness_scale=thick_scale,
            erosion_rate=0.1,
            subsidence_scalar=0.1,
            diffusion_iters=6,
            bowl_amp=0.5,
            bowl_center_east=0.20,
            bowl_center_north=0.20,
            bowl_width_east=0.22,
            bowl_width_north=0.50
        )
        
        try:
            z_layers, p_layers, deposits, compositions = run_forward(
                z0, p0, times[:10], hSL_vals[:10], times[:10], hSS_vals[:10], params_test
            )
            
            final_elevation_mean = np.mean(z_layers[-1])
            results.append((diff_coeff, thick_scale, final_elevation_mean))
            
            print(f"Params ({diff_coeff:.2f}, {thick_scale:.1f}) -> Final elev mean: {final_elevation_mean:.3f}")
            
        except Exception as e:
            print(f"Error with params ({diff_coeff}, {thick_scale}): {e}")
    
    # Check if parameters have effect
    elevations = [r[2] for r in results]
    if len(set([f"{e:.2f}" for e in elevations])) > 1:
        print("✓ Parameters DO affect model output")
        return True
    else:
        print("✗ Parameters do NOT seem to affect model output")
        return False

if __name__ == '__main__':
    test_parameter_sensitivity()