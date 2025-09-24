# %%
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import fields
import os
from concurrent.futures import ProcessPoolExecutor
from model import ForwardParams, run_forward
from enkf_helper import build_obs_index
import seaborn as sns

class SensitivityAnalyzer:
    """
    Perform global sensitivity analysis for forward stratigraphic model parameters
    """
    
    def __init__(self, base_params, nx, ny, times, hSL_vals, hSS_vals):
        self.base_params = base_params
        self.nx, self.ny = nx, ny
        self.times = times
        self.hSL_vals = hSL_vals
        self.hSS_vals = hSS_vals
        
        # Define parameter ranges and types
        self.param_info = self._define_parameter_ranges()
        
    def _define_parameter_ranges(self):
        """Define parameter ranges for sensitivity analysis"""
        return {
            # Transport parameters
            'diffusion_coeff': {'range': [0.01, 1.0], 'log': True, 'base': 0.2},
            'erosion_rate': {'range': [0.0, 0.5], 'log': False, 'base': 0.1},
            
            # Subsidence parameters  
            'subsidence_scalar': {'range': [0.0, 0.5], 'log': False, 'base': 0.1},
            'bowl_amp': {'range': [0.0, 2.0], 'log': False, 'base': 0.5},
            'bowl_center_east': {'range': [0.1, 0.9], 'log': False, 'base': 0.2},
            'bowl_center_north': {'range': [0.1, 0.9], 'log': False, 'base': 0.2},
            'bowl_width_east': {'range': [0.1, 0.8], 'log': False, 'base': 0.22},
            'bowl_width_north': {'range': [0.1, 0.8], 'log': False, 'base': 0.5},
            
            # Sediment parameters
            'thickness_scale': {'range': [0.1, 100.0], 'log': True, 'base': 1.0},
            
            # Supply composition (individual components)
            'supply_comp_0': {'range': [0.1, 0.8], 'log': False, 'base': 0.5},
            'supply_comp_1': {'range': [0.1, 0.8], 'log': False, 'base': 0.3},
            'supply_comp_2': {'range': [0.1, 0.8], 'log': False, 'base': 0.15},
            'supply_comp_3': {'range': [0.02, 0.3], 'log': False, 'base': 0.05}
        }
    
    def generate_parameter_samples(self, n_samples=100, method='sobol'):
        """
        Generate parameter samples using Sobol sequences or random sampling
        """
        param_names = list(self.param_info.keys())
        n_params = len(param_names)
        
        if method == 'sobol':
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=n_params, scramble=True)
                unit_samples = sampler.random(n_samples)
            except ImportError:
                print("Sobol sampling requires scipy>=1.7, using random sampling")
                unit_samples = np.random.random((n_samples, n_params))
        else:
            unit_samples = np.random.random((n_samples, n_params))
        
        # Transform unit samples to parameter space
        samples = np.zeros((n_samples, n_params))
        for i, param_name in enumerate(param_names):
            info = self.param_info[param_name]
            low, high = info['range']
            
            if info['log']:
                # Log-uniform distribution
                samples[:, i] = np.exp(np.log(low) + unit_samples[:, i] * (np.log(high) - np.log(low)))
            else:
                # Uniform distribution
                samples[:, i] = low + unit_samples[:, i] * (high - low)
        
        return samples, param_names
    
    def run_single_sample(self, sample_params, param_names, z0, p0):
        """Run forward model for a single parameter sample"""
        # Create params object
        params = ForwardParams(
            nx=self.nx, ny=self.ny, times=self.times,
            # Set base values first
            diffusion_coeff=self.base_params.diffusion_coeff,
            erosion_rate=self.base_params.erosion_rate,
            subsidence_scalar=self.base_params.subsidence_scalar,
            bowl_amp=self.base_params.bowl_amp,
            bowl_center_east=self.base_params.bowl_center_east,
            bowl_center_north=self.base_params.bowl_center_north,
            bowl_width_east=self.base_params.bowl_width_east,
            bowl_width_north=self.base_params.bowl_width_north,
            thickness_scale=self.base_params.thickness_scale,
            supply_composition=self.base_params.supply_composition.copy()
        )
        
        # Update with sample values
        for i, param_name in enumerate(param_names):
            if param_name.startswith('supply_comp_'):
                # Handle supply composition components
                comp_idx = int(param_name.split('_')[-1])
                params.supply_composition[comp_idx] = sample_params[i]
                # Renormalize
                params.supply_composition /= params.supply_composition.sum()
            else:
                # Regular parameters
                setattr(params, param_name, sample_params[i])
        
        try:
            z_layers, p_layers, deposits, compositions = run_forward(
                z0, p0, self.times, self.hSL_vals, self.times, self.hSS_vals, params
            )
            
            # Compute summary statistics
            final_elevation_mean = np.mean(z_layers[-1])
            final_elevation_std = np.std(z_layers[-1])
            total_deposit = np.sum(deposits[-1])
            elevation_change = np.mean(z_layers[-1] - z_layers[0])
            
            # Composition statistics
            final_comp_mean = np.mean(compositions[-1], axis=(0,1))
            
            return {
                'final_elevation_mean': final_elevation_mean,
                'final_elevation_std': final_elevation_std,
                'total_deposit': total_deposit,
                'elevation_change': elevation_change,
                'final_comp_0': final_comp_mean[0],
                'final_comp_1': final_comp_mean[1],
                'final_comp_2': final_comp_mean[2],
                'final_comp_3': final_comp_mean[3]
            }
        except Exception as e:
            print(f"Error in simulation: {e}")
            return None
    
    def run_sensitivity_analysis(self, z0, p0, n_samples=100, n_workers=None):
        """
        Run global sensitivity analysis
        """
        print(f"Running sensitivity analysis with {n_samples} samples...")
        
        # Generate parameter samples
        samples, param_names = self.generate_parameter_samples(n_samples)
        
        # Run simulations in parallel
        if n_workers is None:
            n_workers = min(os.cpu_count(), n_samples)
        
        args_list = [(samples[i], param_names, z0, p0) for i in range(n_samples)]
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(self._run_sample_wrapper, args_list))
        
        # Filter out failed runs
        valid_results = [r for r in results if r is not None]
        valid_samples = samples[:len(valid_results)]
        
        print(f"Successfully completed {len(valid_results)}/{n_samples} runs")
        
        # Convert results to arrays
        output_names = list(valid_results[0].keys())
        outputs = np.array([[r[name] for name in output_names] for r in valid_results]).T
        
        return valid_samples, param_names, outputs, output_names
    
    def _run_sample_wrapper(self, args):
        """Wrapper for parallel execution"""
        return self.run_single_sample(*args)
    
    def compute_sobol_indices(self, samples, outputs):
        """
        Compute Sobol sensitivity indices using SALib
        """
        try:
            from SALib.analyze import sobol
            from SALib.sample import saltelli
            
            # This is a simplified version - for proper Sobol indices,
            # you would need to use SALib's sampling method
            # Here we compute correlation-based sensitivity measures
            
            n_params = samples.shape[1]
            n_outputs = outputs.shape[0]
            
            sensitivities = {}
            
            for i, output_name in enumerate(['output_' + str(j) for j in range(n_outputs)]):
                # Compute standardized regression coefficients as sensitivity measure
                X = samples
                y = outputs[i]
                
                # Standardize inputs and outputs
                X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
                y_std = (y - np.mean(y)) / np.std(y)
                
                # Linear regression coefficients
                try:
                    beta = np.linalg.lstsq(X_std, y_std, rcond=None)[0]
                    sensitivities[output_name] = np.abs(beta)
                except:
                    sensitivities[output_name] = np.zeros(n_params)
            
            return sensitivities
            
        except ImportError:
            print("SALib not available, using correlation-based analysis")
            return self._correlation_analysis(samples, outputs)
    
    def _correlation_analysis(self, samples, outputs):
        """Simple correlation-based sensitivity analysis"""
        n_params = samples.shape[1]
        n_outputs = outputs.shape[0]
        
        sensitivities = {}
        
        for i in range(n_outputs):
            correlations = np.zeros(n_params)
            for j in range(n_params):
                correlations[j] = np.abs(np.corrcoef(samples[:, j], outputs[i])[0, 1])
            sensitivities[f'output_{i}'] = correlations
        
        return sensitivities
    
    def plot_sensitivity_results(self, param_names, output_names, sensitivities, save_dir='./plots'):
        """
        Create comprehensive sensitivity analysis plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Sensitivity heatmap
        n_outputs = len(output_names)
        n_params = len(param_names)
        
        sensitivity_matrix = np.zeros((n_outputs, n_params))
        for i in range(n_outputs):
            sensitivity_matrix[i, :] = sensitivities[f'output_{i}']
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(sensitivity_matrix, 
                   xticklabels=param_names, 
                   yticklabels=output_names,
                   annot=True, fmt='.3f', cmap='viridis')
        plt.title('Parameter Sensitivity Heatmap')
        plt.xlabel('Parameters')
        plt.ylabel('Outputs')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sensitivity_heatmap.png'), dpi=300)
        plt.show()
        
        # 2. Parameter ranking for each output
        for i, output_name in enumerate(output_names):
            plt.figure(figsize=(10, 6))
            
            sens_values = sensitivities[f'output_{i}']
            sorted_indices = np.argsort(sens_values)[::-1]
            
            plt.bar(range(len(param_names)), sens_values[sorted_indices])
            plt.xticks(range(len(param_names)), 
                      [param_names[j] for j in sorted_indices], 
                      rotation=45, ha='right')
            plt.ylabel('Sensitivity Index')
            plt.title(f'Parameter Sensitivity for {output_name}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'sensitivity_{output_name}.png'), dpi=300)
            plt.show()
        
        # 3. Overall parameter importance
        plt.figure(figsize=(10, 6))
        
        overall_sensitivity = np.mean(sensitivity_matrix, axis=0)
        sorted_indices = np.argsort(overall_sensitivity)[::-1]
        
        plt.bar(range(len(param_names)), overall_sensitivity[sorted_indices])
        plt.xticks(range(len(param_names)), 
                  [param_names[j] for j in sorted_indices], 
                  rotation=45, ha='right')
        plt.ylabel('Mean Sensitivity Index')
        plt.title('Overall Parameter Importance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'overall_sensitivity.png'), dpi=300)
        plt.show()
        
        return sensitivity_matrix, overall_sensitivity
    
    def recommend_parameters(self, param_names, overall_sensitivity, threshold=0.1):
        """
        Recommend parameters for EnKF based on sensitivity analysis
        """
        important_params = []
        sensitivity_dict = dict(zip(param_names, overall_sensitivity))
        
        print("Parameter Sensitivity Analysis Results:")
        print("=" * 50)
        
        sorted_params = sorted(sensitivity_dict.items(), key=lambda x: x[1], reverse=True)
        
        for param, sensitivity in sorted_params:
            status = "RECOMMENDED" if sensitivity > threshold else "Not recommended"
            print(f"{param:20s}: {sensitivity:.4f} - {status}")
            
            if sensitivity > threshold:
                important_params.append(param)
        
        print(f"\nRecommended parameters for EnKF (sensitivity > {threshold}):")
        for param in important_params:
            print(f"  - {param}")
        
        return important_params

# Usage example
def run_sensitivity_analysis_example():
    """Example of how to use the sensitivity analyzer"""
    
    print("=" * 60)
    print("GLOBAL SENSITIVITY ANALYSIS FOR FSM PARAMETERS")
    print("=" * 60)
    
    try:
        # Load reference data (similar to your enkf_fsm.py)
        print("Loading reference data...")
        data = np.load("./data/reference_fsm_results.npz", allow_pickle=True)
        nx, ny, nt = data["nx"], data["ny"], data["nt"]
        times = data["times"]
        z0 = data["z0"]
        p0 = data["p0"]
        hSL_vals = data["hSL_vals"]
        hSS_vals = data["hSS_vals"]
        params = data["params"].item()
        print(f"✓ Data loaded successfully (nx={nx}, ny={ny}, nt={nt})")
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None, None
    
    try:
        # Create sensitivity analyzer
        print("Initializing sensitivity analyzer...")
        analyzer = SensitivityAnalyzer(params, nx, ny, times, hSL_vals, hSS_vals)
        print(f"✓ Analyzer initialized with {len(analyzer.param_info)} parameters")
        
        # Run analysis (use smaller sample size for testing)
        print("\nRunning sensitivity analysis...")
        samples, param_names, outputs, output_names = analyzer.run_sensitivity_analysis(
            z0, p0, n_samples=200  # Small sample for quick testing
        )
        
        # Compute sensitivity indices
        print("Computing sensitivity indices...")
        sensitivities = analyzer.compute_sobol_indices(samples, outputs)
        
        # Plot results
        print("Generating plots...")
        sensitivity_matrix, overall_sensitivity = analyzer.plot_sensitivity_results(
            param_names, output_names, sensitivities
        )
        
        # Get parameter recommendations
        print("\nGenerating parameter recommendations...")
        recommended_params = analyzer.recommend_parameters(
            param_names, overall_sensitivity, threshold=0.1
        )
        
        print("\n" + "=" * 60)
        print("SENSITIVITY ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Total parameters analyzed: {len(param_names)}")
        print(f"Output variables: {len(output_names)}")
        print(f"Recommended parameters: {len(recommended_params)}")
        print("Check the ./plots/ directory for visualization results.")
        
        return recommended_params, sensitivity_matrix
        
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == '__main__':
    print("Starting Global Sensitivity Analysis...")
    recommended_params, sensitivity_matrix = run_sensitivity_analysis_example()
    
    if recommended_params is not None:
        print("\n🎯 FINAL RECOMMENDATIONS FOR EnKF:")
        print("-" * 40)
        for i, param in enumerate(recommended_params, 1):
            print(f"{i:2d}. {param}")
    else:
        print("❌ Analysis failed. Please check error messages above.")