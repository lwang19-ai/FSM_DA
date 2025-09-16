# %%
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import os

"""
Forward Stratigraphic Model with intuitive coordinate system:
- x: east/horizontal direction (left to right, 0 to nx-1)
- y: north/vertical direction (bottom to top, 0 to ny-1)
- Sediment source: western (left) boundary
- Arrays shaped as (time, y, x) for consistency
"""

def p_to_s(p):
    """Convert 4-part proportions to 3-D logistic coordinates."""
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    p = p / p.sum(axis=-1, keepdims=True)
    s = np.log(p[..., :3] / p[..., 3:4])
    return s

def s_to_p(s):
    """Convert 3-D logistic coordinates back to 4-part proportions."""
    es = np.exp(np.clip(s, -50, 50))
    denom = 1.0 + es.sum(axis=-1, keepdims=True)
    p_first3 = es / denom
    p4 = 1.0 / denom
    return np.concatenate([p_first3, p4], axis=-1)

def piecewise_linear(times, values, t_query):
    """Linear interpolation for sea level and sediment supply curves."""
    times = np.asarray(times)
    values = np.asarray(values)
    t = np.asarray(t_query)
    return np.interp(t, times, values, left=values[0], right=values[-1])

def diffuse(field, coeff, n_iters=1):
    """5-point stencil diffusion for sediment transport."""
    if coeff <= 0 or n_iters <= 0:
        return field
    f = field.copy()
    for _ in range(n_iters):
        fpad = np.pad(f, ((1,1),(1,1)), mode='edge')
        lap = (fpad[1:-1,2:] + fpad[1:-1,:-2] + 
               fpad[2:,1:-1] + fpad[:-2,1:-1] - 4*f)
        f = f + coeff * lap
    return f

# %%
@dataclass
class ForwardParams:
    # Grid dimensions
    nx: int                    # points in east direction
    ny: int                    # points in north direction
    times: np.ndarray          # timesteps (nt,)
    
    # Transport parameters
    diffusion_coeff: float = 0.2
    diffusion_iters: int = 1
    erosion_rate: float = 0.0
    
    # Subsidence parameters
    subsidence_rate: np.ndarray | None = None  # (ny, nx)
    subsidence_scalar: float = 0.0
    
    # Basin geometry (normalized coordinates)
    bowl_amp: float = 0.0
    bowl_center_east: float = 0.5    # center in x (0..1)
    bowl_center_north: float = 0.5   # center in y (0..1)
    bowl_width_east: float = 0.25    # spread in x
    bowl_width_north: float = 0.25   # spread in y
    
    # Sediment properties
    supply_composition: np.ndarray = field(
        default_factory=lambda: np.array([0.50, 0.30, 0.15, 0.05])
    )
    source_mask: np.ndarray | None = None  # (ny, nx)
    thickness_scale: float = 1.0

def run_forward(z0, p0, hSL_times, hSL_vals, hSS_times, hSS_vals, params):
    """
    Forward stratigraphic model with sediment transport and deposition.
    
    Args:
        z0: (ny, nx) initial bathymetry/elevation
        p0: (4,) initial proportions
        hSL_times, hSL_vals: sea level curve
        hSS_times, hSS_vals: sediment supply curve
        params: model parameters
    
    Returns:
        z_layers: (nt, ny, nx) elevation at each timestep
        p_layers: (nt, ny, nx, 4) proportions at each timestep
        deposits: (nt, ny, nx) deposit thickness
        compositions: (nt, ny, nx, 4) deposit composition
    """
    nx, ny = params.nx, params.ny
    times = np.asarray(params.times)
    nt = times.size
    
    # Initialize arrays
    deposits = np.zeros((nt, ny, nx))
    compositions = np.zeros((nt, ny, nx, 4))
    z_layers = np.zeros((nt, ny, nx))
    p_layers = np.zeros((nt, ny, nx, 4))
    
    # Initial conditions
    z = z0.astype(float).copy()
    p_surface = np.broadcast_to(
        np.asarray(p0, dtype=float) / np.sum(p0), 
        (ny, nx, 4)
    ).copy()
    s_surface = p_to_s(p_surface)
    
    # Setup subsidence field
    if params.subsidence_rate is None:
        # Build normalized coordinate grids
        Y = (np.arange(ny) / max(ny-1, 1)).reshape(-1, 1)
        X = (np.arange(nx) / max(nx-1, 1)).reshape(1, -1)
        
        # Compute Gaussian bowl
        dy2 = (Y - params.bowl_center_north)
        dx2 = (X - params.bowl_center_east)
        sigy = max(params.bowl_width_north, 1e-9)
        sigx = max(params.bowl_width_east, 1e-9)
        gaussian = np.exp(-0.5 * ((dy2/sigy)**2 + (dx2/sigx)**2))
        subs = params.subsidence_scalar + params.bowl_amp * gaussian
    else:
        subs = np.asarray(params.subsidence_rate)
    
    # Setup source mask
    if params.source_mask is None:
        src = np.zeros((ny, nx))
        src[:, 0] = 1.0  # western boundary
    else:
        src = np.asarray(params.source_mask)
    
    # Normalize supply composition
    supply_comp = params.supply_composition.astype(float)
    supply_comp = supply_comp / supply_comp.sum()
    
    # Time stepping
    for k, tk in enumerate(times):
        # Get sea level and supply rate
        sea = piecewise_linear(hSL_times, hSL_vals, tk)
        supply_rate = piecewise_linear(hSS_times, hSS_vals, tk)
        
        # Compute accommodation
        accommodation = sea - z
        
        # Handle erosion
        if params.erosion_rate > 0:
            exposed = (accommodation < 0.0)
            dz_erosion = np.zeros_like(z)
            dz_erosion[exposed] = -params.erosion_rate
            z = z + dz_erosion
        
        # Apply subsidence
        z = z - subs
        
        # Compute deposit thickness
        base_thick = (params.thickness_scale * supply_rate * 
                     (src / (src.max() + 1e-12)))
        thick = diffuse(base_thick, params.diffusion_coeff, 
                       params.diffusion_iters)
        
        # Apply accommodation limits
        pos_acc = np.clip(accommodation, 0.0, None)
        acc_w = pos_acc / (pos_acc.max() + 1e-12)
        deposit = thick * acc_w
        
        # Update elevation
        z_new = z + deposit
        
        # Update composition
        h_skin = 1.0  # characteristic mixing depth
        alpha = deposit / (deposit + h_skin)
        alpha = np.clip(alpha, 0.0, 1.0)
        
        p_sup = np.broadcast_to(supply_comp, (ny, nx, 4))
        p_old = s_to_p(s_surface)
        p_new = ((1.0 - alpha)[..., None] * p_old + 
                 alpha[..., None] * p_sup)
        p_new = p_new / p_new.sum(axis=-1, keepdims=True)
        s_surface = p_to_s(p_new)
        
        # Apply diffusion to elevation
        z = diffuse(z_new, params.diffusion_coeff, 
                   params.diffusion_iters)
        
        # Save results
        z_layers[k] = z
        p_layers[k] = s_to_p(s_surface)
        deposits[k] = deposit
        compositions[k] = p_new
        
        # Handle erosion effects on previous layers
        if params.erosion_rate > 0:
            exposed = (accommodation < 0.0)
            erode_amt = np.full((ny, nx), params.erosion_rate)
            for kk in range(k, -1, -1):
                mask = (deposits[kk] > 0) & exposed
                amt = np.minimum(deposits[kk][mask], 
                               erode_amt[mask])
                deposits[kk][mask] -= amt
                erode_amt[mask] -= amt
                zero_mask = (deposits[kk] == 0) & exposed
                compositions[kk][zero_mask] = 0
                if np.all(erode_amt <= 0):
                    break
    
    return z_layers, p_layers, deposits, compositions

# ---------------- Example usage (commented) ----------------
if __name__ == "__main__":
    nx, ny, nt = 64, 64, 100
    times = np.linspace(0, 1000, nt)   # kyr
    z0 = -200.0 * np.ones((ny, nx))    # initial bathymetry, shape: (ny, nx)
    p0 = np.array([0.15, 0.35, 0.35, 0.15])  # initial surface proportions
    hSL_vals = -50 + 30*np.sin(2*np.pi*times/times[-1])    # oscillatory sea level
    hSS_vals = 1.0 + 0.5*np.cos(2*np.pi*times/times[-1])   # oscillatory supply
    params = ForwardParams(
        nx=nx, ny=ny, times=times,
        diffusion_coeff=0.25,
        diffusion_iters=6,
        erosion_rate=0.1,
        thickness_scale=40.0,
        subsidence_scalar=0.1,
        bowl_amp=0.5,
        bowl_center_east=0.20,    # renamed from bowl_center_x
        bowl_center_north=0.20,   # renamed from bowl_center_y
        bowl_width_east=0.22,     # renamed from bowl_sigma_x
        bowl_width_north=0.50     # renamed from bowl_sigma_y
    )

    z_layers, p_layers, deposits, compositions = run_forward(
        z0, p0,
        times, hSL_vals,      # hSL_times, hSL_vals
        times, hSS_vals,      # hSS_times, hSS_vals
        params
    )
    print(z_layers.shape, p_layers.shape, deposits.shape, compositions.shape)

    np.random.seed(42)

    # Recommended absolute observation errors (tune as needed)
    sigma_z = 5.0    # m
    sigma_p = 0.05   # fraction (0..1)
    sigma_d = 5.0    # m (if used)
    sigma_c = 0.05   # fraction (0..1)

    # Create noisy observations with fixed sigmas
    z_layers_meas = z_layers + sigma_z * np.random.randn(*z_layers.shape)
    p_layers_meas = p_layers + sigma_p * np.random.randn(*p_layers.shape)
    deposits_meas = deposits + sigma_d * np.random.randn(*deposits.shape)
    compositions_meas = compositions + sigma_c * np.random.randn(*compositions.shape)

    # clip/renormalize for proportions
    p_layers_meas = np.clip(p_layers_meas, 0.0, 1.0)
    compositions_meas = np.clip(compositions_meas, 0.0, 1.0)
    compositions_meas /= (compositions_meas.sum(axis=-1, keepdims=True) + 1e-12)

    # ---- save truth + observations ----
    os.makedirs("./data", exist_ok=True)
    np.savez(
        "./data/reference_fsm_results.npz",
        nx=nx, ny=ny, nt=nt,
        times=times,
        z0=z0, p0=p0,
        hSL_vals=hSL_vals,
        hSS_vals=hSS_vals,
        # forward-model truth
        z_layers=z_layers,
        p_layers=p_layers,
        deposits=deposits,
        compositions=compositions,
        # noisy observations
        sigma_z=sigma_z,
        sigma_p=sigma_p,
        sigma_d=sigma_d,
        sigma_c=sigma_c,
        z_layers_meas=z_layers_meas,
        p_layers_meas=p_layers_meas,
        deposits_meas=deposits_meas,
        compositions_meas=compositions_meas,
        # params for loader
        params=np.array(params, dtype=object)
    )


    # Plotting functions
    def plot_final_elevation(z_layers, times):
        z_final = z_layers[-1]
        plt.figure()
        im = plt.imshow(z_final, origin="lower")
        plt.colorbar(im, label="Elevation (m)")
        plt.title(f"Final elevation at t={times[-1]:.1f}")
        plt.xlabel("Distance East [index]")
        plt.ylabel("Distance North [index]")
        plt.show()

    def plot_final_proportions(p_layers, times):
        """Plot final surface sediment proportion maps for each lithology.
        
        Args:
            p_layers: array of shape (nt, ny, nx, 4) with proportions
            times: array of timesteps
        """
        p_final = p_layers[-1]  # shape (ny, nx, 4)
        for i in range(4):
            plt.figure()
            im = plt.imshow(p_final[..., i], origin="lower", vmin=0, vmax=1)
            plt.colorbar(im, label="Proportion")
            plt.title(f"Final surface proportion: lithology {i+1} at t={times[-1]:.1f}")
            plt.xlabel("Distance East [index]")
            plt.ylabel("Distance North [index]")
            plt.show()

    def plot_midline_section(z_layers, times):
        nt, ny, nx = z_layers.shape
        mid = ny // 2  # middle in north-south direction
        # stack elevation along the midline over time
        section = z_layers[:, mid, :]  # (time, x)
        plt.figure()
        im = plt.imshow(section, aspect="auto", origin="lower",
                       extent=[0, nx-1, times[0], times[-1]])
        plt.colorbar(im, label="Elevation (m)")
        plt.title("Elevation cross-section (middle latitude) through time")
        plt.xlabel("Distance East [index]")
        plt.ylabel("Time [kyr]")
        plt.show()

    def plot_domain_average_proportions(p_layers, times):
        # p_layers: (nt, ny, nx, 4)
        mean_p = p_layers.mean(axis=(1, 2))  # (nt, 4)
        for i in range(4):
            y = mean_p[:, i]
            ymin, ymax = float(y.min()), float(y.max())
            pad = max(1e-3, 0.05 * (ymax - ymin))
            plt.figure()
            plt.plot(times, y)
            plt.title(f"Domain-average proportion of lithology {i+1}")
            plt.xlabel("Time [kyr]")
            plt.ylabel("Proportion")
            plt.ylim(ymin - pad, ymax + pad)
            plt.grid(True, alpha=0.3)
            plt.show()

    def plot_mean_thickness(z_layers, times):
        # Using change from initial as proxy for cumulative thickness
        dz = z_layers - z_layers[0:1]  # (nt, ny, nx)
        mean_thick = dz.mean(axis=(1,2))  # (nt,)
        plt.figure()
        plt.plot(times, mean_thick)
        plt.title("Domain-average thickness change over time")
        plt.xlabel("Time [kyr]")
        plt.ylabel("Mean Δelevation [m]")
        plt.show()
    # ---- Run the plots ----
    plot_final_elevation(z_layers, times)
    plot_final_proportions(p_layers, times)
    plot_midline_section(z_layers, times)
    plot_domain_average_proportions(p_layers, times)
    plot_mean_thickness(z_layers, times)

    # %% --- Ternary-colored stratigraphic cross-section (sand–silt–clay) ---

    def _ternary_rgb(p4):
        """Convert composition to RGB color."""
        p3 = np.asarray(p4[:3], dtype=float)
        s = p3.sum()
        if s <= 0:
            w = np.array([1.0, 0.0, 0.0])
        else:
            w = p3 / s
        sand = np.array([1.0, 0.0, 0.0])  # red
        silt = np.array([1.0, 1.0, 0.0])  # yellow
        clay = np.array([0.0, 1.0, 0.0])  # green
        return w[0]*sand + w[1]*silt + w[2]*clay

    def plot_layered_cross_section(
        z_layers, p_layers, times,
        direction="east-west",
        position=None,
        nz=800, dx=1.0,
        title_prefix="Stratigraphic cross-section"
    ):
        """
        Plot stratigraphic cross-section with lithology colors.
        
        Args:
            direction: "east-west" or "north-south"
            position: index for section location
        """
        nt, ny, nx = z_layers.shape
        
        if direction == "east-west":
            idx = ny // 2 if position is None else position
            z_ts = z_layers[:, idx, :]
            p_ts = p_layers[:, idx, :, :]
            width = nx
            xlabel = "Distance East [index]"
        else:
            idx = nx // 2 if position is None else position
            z_ts = z_layers[:, :, idx]
            p_ts = p_layers[:, :, idx, :]
            width = ny
            xlabel = "Distance North [index]"
        
        zmin, zmax = float(np.min(z_ts)), float(np.max(z_ts))
        img = np.ones((nz, width, 3), dtype=float)
        
        # Fill layers
        for k in range(nt - 1):
            top = z_ts[k+1]
            bot = z_ts[k]
            compositions = p_ts[k]
            
            for x in range(width):
                t, b = float(top[x]), float(bot[x])
                if t == b:
                    continue
                r_top = int((max(t, b) - zmin) / (zmax - zmin) * (nz - 1))
                r_bot = int((min(t, b) - zmin) / (zmax - zmin) * (nz - 1))
                if r_top == r_bot:
                    continue
                color = _ternary_rgb(compositions[x])
                img[r_bot:r_top, x, :] = color
        
        # Plot
        dist = np.arange(width) * dx
        plt.figure(figsize=(11, 6))
        plt.imshow(img[::-1, :, :],
                origin="lower",
                aspect="auto",
                extent=[dist[0], dist[-1], zmin, zmax])
        plt.plot(dist, z_ts[-1], 'k', linewidth=1.2)
        plt.xlabel(xlabel)
        plt.ylabel("Elevation [m]")
        plt.title(f"{title_prefix}\n{direction}, position={idx}")
        
        # Legend
        ax = plt.gca()
        legend_elements = [
            plt.Rectangle((0,0), 1, 1, fc='red', label='Sand'),
            plt.Rectangle((0,0), 1, 1, fc='yellow', label='Silt'),
            plt.Rectangle((0,0), 1, 1, fc='green', label='Clay')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()

    plot_layered_cross_section(z_layers, p_layers, times, 
                            direction="east-west",    # changed from axis="x"
                            position=None, 
                            nz=800, 
                            dx=1.0,
                            title_prefix="Layered cross-section")

    plot_layered_cross_section(z_layers, p_layers, times, 
                            direction="north-south",  # changed from axis="y"
                            position=None, 
                            nz=800, 
                            dx=1.0,
                            title_prefix="Layered cross-section")
    # %%
