# %%
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

# Utilities: logistic transforms for proportions
def p_to_s(p):
    """
    Convert 4-part proportions (last axis=4) to 3-D logistic coordinates s (last axis=3).
    s_j = log( p_j / p_4 ), j=1..3
    """
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps) # avoid unstable p.
    p = p / p.sum(axis=-1, keepdims=True)
    s = np.log(p[..., :3] / p[..., 3:4])
    return s

def s_to_p(s):
    """
    Inverse transform: given s (...,3), return p (...,4)
    p_j = exp(s_j) / (1 + sum_j exp(s_j)), j=1..3
    p_4 = 1 / (1 + sum_j exp(s_j))
    """
    es = np.exp(np.clip(s, -50, 50))
    denom = 1.0 + es.sum(axis=-1, keepdims=True)
    p_first3 = es / denom
    p4 = 1.0 / denom
    return np.concatenate([p_first3, p4], axis=-1)

# ---------- Piecewise-linear curve evaluator for h_SL and h_SS ----------
# Interpolates sea level and sediment supply curves at arbitrary times,
# ensuring smooth transitions between input values.
def piecewise_linear(times, values, t_query):
    """
    times: (nt,) strictly increasing geological times (e.g., kyr)
    values: (nt,) curve values aligned to `times`
    t_query: scalar or array of query times
    Returns linear interpolation with edge clamping.
    """
    times = np.asarray(times)
    values = np.asarray(values)
    t = np.asarray(t_query)
    return np.interp(t, times, values, left=values[0], right=values[-1])

# ---------- Simple diffusion operator ----------
# Applies a simple smoothing (diffusion) to a 2D field,
# mimicking lateral sediment transport or avulsion.
def diffuse(field, coeff, n_iters=1):
    """
    5-point stencil explicit diffusion (very mild smoothing).
    coeff: small positive scalar (stability if coeff <= 0.25)
    """
    if coeff <= 0 or n_iters <= 0:
        return field
    f = field.copy()
    for _ in range(n_iters):
        fpad = np.pad(f, ((1,1),(1,1)), mode='edge')
        lap = (fpad[1:-1,2:] + fpad[1:-1,:-2] + fpad[2:,1:-1] + fpad[:-2,1:-1] - 4*f)
        f = f + coeff * lap
    return f

@dataclass
class ForwardParams:
    # Grid & run controls
    nx: int
    ny: int
    times: np.ndarray          # shape (nt,)
    diffusion_coeff: float = 0.2
    diffusion_iters: int = 1
    erosion_rate: float = 0.0   # thickness per step when exposed (z > sea level)
    subsidence_rate: np.ndarray | None = None  # (nx, ny), positive = down (adds accommodation)
    subsidence_scalar: float = 0.0             # uniform per-step subsidence (used if subsidence_rate is None)
    # Optional Gaussian subsidence bowl (auto-generated if subsidence_rate is None)
    bowl_amp: float = 0.0          # amplitude per step (positive = down)
    bowl_center_x: float = 0.5     # center in normalized x (0..1)
    bowl_center_y: float = 0.5     # center in normalized y (0..1)
    bowl_sigma_x: float = 0.25     # std-dev in normalized x
    bowl_sigma_y: float = 0.25     # std-dev in normalized y
    # Sediment supply composition (fixed 4-part vector)
    supply_composition: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.3, 0.15, 0.05]))
    # Source mask where sediment enters (nx, ny) in [0,1]; will diffuse/spread
    source_mask: np.ndarray | None = None
    # Thickness scaling
    thickness_scale: float = 1.0  # converts supply rate units into depositional thickness per step

def run_forward(
    z0,                 # (nx, ny) initial bathymetry/elevation
    p0,                 # (4,) initial surface sediment proportions
    hSL_times, hSL_vals,# (nt,), (nt,) sea level
    hSS_times, hSS_vals,# (nt,), (nt,) sediment supply
    params: ForwardParams,
):
    """
    Deterministic toy forward model inspired by the GPM setting in Skauvold & Eidsvik (2018).
    State: elevation z and surface proportions p at each time step.
    Drivers: piecewise-linear sea level h_SL(t) and sediment-supply rate h_SS(t).

    Returns:
      z_layers: (nt, nx, ny)  deposited surface elevation at each step
      p_layers: (nt, 4, nx, ny)  surface sediment proportions at each step
    """
    nx, ny = params.nx, params.ny
    times = np.asarray(params.times)
    nt = times.size

    # Validate & prepare fields
    z = z0.astype(float).copy()
    p_surface = np.broadcast_to(np.asarray(p0, dtype=float) / np.sum(p0), (nx, ny, 4)).copy()
    s_surface = p_to_s(p_surface)  # EnKF-friendly internals

    # automatically generate a subsidence rate field or manually define it
    if params.subsidence_rate is None:
        # Build uniform + Gaussian bowl (domain-normalized coordinates)
        X = (np.arange(nx) / max(nx - 1, 1)).reshape(-1, 1)
        Y = (np.arange(ny) / max(ny - 1, 1)).reshape(1, -1)
        dx2 = (X - params.bowl_center_x)
        dy2 = (Y - params.bowl_center_y)
        # Avoid divide-by-zero if sigma set to 0
        sigx = max(params.bowl_sigma_x, 1e-9)
        sigy = max(params.bowl_sigma_y, 1e-9)
        gaussian = np.exp(-0.5 * ((dx2 / sigx) ** 2 + (dy2 / sigy) ** 2))
        subs = params.subsidence_scalar + params.bowl_amp * gaussian
        subs = subs.astype(float)
    else:
        subs = np.asarray(params.subsidence_rate, dtype=float)

    # Create source mask if not provided. This is a very simplified setting,
    # sediment enters from the left boundary. We can manually adjust it to
    # fit more complex scenarios if needed.
    if params.source_mask is None:
        # Default: sediment enters from the left boundary (simple river entry)
        src = np.zeros((nx, ny), dtype=float)
        src[:, 0] = 1.0
    else:
        src = np.asarray(params.source_mask, dtype=float)

    # Normalize supply composition
    supply_comp = params.supply_composition.astype(float)
    supply_comp = supply_comp / supply_comp.sum()

    # Storage
    z_layers = np.zeros((nt, nx, ny), dtype=float)
    p_layers = np.zeros((nt, 4, nx, ny), dtype=float)

    # Time stepping
    for k, tk in enumerate(times):
        sea = piecewise_linear(hSL_times, hSL_vals, tk)  # scalar sea level at time tk
        supply_rate = piecewise_linear(hSS_times, hSS_vals, tk)  # scalar supply intensity
        # Accommodation (positive -> room to deposit; negative -> exposed)
        accommodation = sea - z  # if sea is higher than surface, there is accommodation

        # --- Erosion where exposed (z > sea) ---
        if params.erosion_rate > 0:
            exposed = (accommodation < 0.0)
            dz_erosion = np.zeros_like(z)
            dz_erosion[exposed] = -params.erosion_rate
            z = z + dz_erosion
            # keep proportions unchanged during erosion (could bias coarser fraction in real model)

        # --- Subsidence/Uplift (positive subs -> down) ---
        z = z - subs  # subsidence increases accommodation (z moves down)

        # --- Compute a provisional deposit thickness field ---
        # Base thickness from supply, scaled and concentrated near source, then smoothed by diffusion
        base_thick = params.thickness_scale * supply_rate * (src / (src.max() + 1e-12))
        thick = diffuse(base_thick, coeff=params.diffusion_coeff, n_iters=params.diffusion_iters)

        # Clip by accommodation: only deposit where there is room (accommodation > 0)
        # More accommodation -> allow more of the diffused field to settle.
        pos_acc = np.clip(accommodation, 0.0, None)
        # Weight deposition by normalized accommodation
        acc_w = pos_acc / (pos_acc.max() + 1e-12)
        deposit = thick * acc_w

        # Update elevation
        z_new = z + deposit

        # --- Update surface sediment composition by thickness-weighted mixing ---
        # Mix previous surface with supplied composition in the newly deposited layer
        # If no deposit at a cell, keep composition
        mix_w = (deposit > 1e-12).astype(float)
        # Fractional contribution of new material to the surface "skin" (simple first-order scheme)
        alpha = np.clip(deposit / (deposit.max() + 1e-12), 0.0, 1.0) * mix_w
        p_sup = np.broadcast_to(supply_comp, (nx, ny, 4))
        p_old = s_to_p(s_surface)
        p_new = (1.0 - alpha)[..., None] * p_old + alpha[..., None] * p_sup
        p_new = p_new / p_new.sum(axis=-1, keepdims=True)
        s_surface = p_to_s(p_new)

        # --- Mild topographic smoothing to mimic lateral transport/avulsion ---
        z = diffuse(z_new, coeff=params.diffusion_coeff, n_iters=params.diffusion_iters)

        # Save layer k (surface at time tk)
        z_layers[k] = z
        p_layers[k] = np.moveaxis(s_to_p(s_surface), -1, 0)  # (4, nx, ny)

    return z_layers, p_layers

# ---------------- Example usage (commented) ----------------
nx, ny, nt = 64, 64, 50
times = np.linspace(0, 1000, nt)   # kyr
z0 = -200.0 * np.ones((nx, ny))    # initial bathymetry (negative = below sea level)
p0 = np.array([0.4, 0.3, 0.2, 0.1])  # initial surface proportions
hSL_vals = -50 + 30*np.sin(2*np.pi*times/times[-1])    # oscillatory sea level
hSS_vals = 1.0 + 0.5*np.cos(2*np.pi*times/times[-1])   # oscillatory supply
params = ForwardParams(
    nx=nx, ny=ny, times=times,
    diffusion_coeff=0.25,
    diffusion_iters=6,
    erosion_rate=0.1,
    thickness_scale=40.0,
    subsidence_scalar=0.2,
    bowl_amp=1.0,
    bowl_center_x=0.20,
    bowl_center_y=0.20,
    bowl_sigma_x=0.22,
    bowl_sigma_y=0.50
)
z_layers, p_layers = run_forward(z0, p0, times, hSL_vals, times, hSS_vals, params)
z_layers.shape, p_layers.shape  # -> (nt, nx, ny), (nt, 4, nx, ny)

# %%
# 1) Final surface elevation map
def plot_final_elevation(z_layers, times):
    z_final = z_layers[-1]
    plt.figure()
    im = plt.imshow(z_final, origin="lower")
    plt.colorbar(im, label="Elevation (m)")
    plt.title(f"Final elevation at t={times[-1]:.1f}")
    plt.xlabel("y-index"); plt.ylabel("x-index")
    plt.show()

# 2) Final surface sediment proportion maps (4 lithologies)
def plot_final_proportions(p_layers, times):
    p_final = p_layers[-1]  # shape (4, nx, ny)
    for i in range(4):
        plt.figure()
        im = plt.imshow(p_final[i], origin="lower", vmin=0, vmax=1)
        plt.colorbar(im, label="Proportion")
        plt.title(f"Final surface proportion: lithology {i+1} at t={times[-1]:.1f}")
        plt.xlabel("y-index"); plt.ylabel("x-index")
        plt.show()

# 3) Midline cross-section of elevation through time
def plot_midline_section(z_layers, times):
    nx, ny = z_layers.shape[1], z_layers.shape[2]
    mid = nx // 2
    # stack elevation along the midline over time (time × y)
    section = z_layers[:, mid, :]
    plt.figure()
    im = plt.imshow(section, aspect="auto", origin="lower",
                    extent=[0, ny-1, times[0], times[-1]])
    plt.colorbar(im, label="Elevation (m)")
    plt.title("Elevation cross-section (midline) through time")
    plt.xlabel("y-index"); plt.ylabel("Time")
    plt.show()

# 4) Domain-average lithology proportions through time
def plot_domain_average_proportions(p_layers, times):
    # p_layers: (nt, 4, nx, ny)
    mean_p = p_layers.mean(axis=(2,3))  # -> (nt, 4)
    for i in range(4):
        plt.figure()
        plt.plot(times, mean_p[:, i])
        plt.title(f"Domain-average proportion of lithology {i+1}")
        plt.xlabel("Time"); plt.ylabel("Proportion")
        plt.ylim(0, 1)
        plt.show()

# 5) Total deposited thickness vs time (domain mean)
def plot_mean_thickness(z_layers, times):
    # Using change from initial as proxy for cumulative thickness
    dz = z_layers - z_layers[0:1]            # (nt, nx, ny)
    mean_thick = dz.mean(axis=(1,2))         # (nt,)
    plt.figure()
    plt.plot(times, mean_thick)
    plt.title("Domain-average thickness change over time")
    plt.xlabel("Time"); plt.ylabel("Mean Δelevation (m)")
    plt.show()

# ---- Run the plots ----
plot_final_elevation(z_layers, times)
plot_final_proportions(p_layers, times)
plot_midline_section(z_layers, times)
plot_domain_average_proportions(p_layers, times)
plot_mean_thickness(z_layers, times)

# %% --- Ternary-colored stratigraphic cross-section (sand–silt–clay) ---

def _ternary_rgb(p4):
    """Map 4-part proportions to ternary RGB using the first three parts
    as Sand (R), Silt (R+G -> yellow), Clay (G). The 4th part is ignored
    for coloring; the first three are renormalized to sum to 1.
    Returns an RGB triple in [0,1].
    """
    p3 = np.asarray(p4[:3], dtype=float)
    s = p3.sum()
    if s <= 0:
        w = np.array([1.0, 0.0, 0.0])
    else:
        w = p3 / s
    sand = np.array([1.0, 0.0, 0.0])      # red
    silt = np.array([1.0, 1.0, 0.0])      # yellow
    clay = np.array([0.0, 1.0, 0.0])      # green
    return w[0]*sand + w[1]*silt + w[2]*clay

# %%
def plot_layered_cross_section(
    z_layers,
    p_layers,
    times,
    axis="x",
    idx=None,
    nz=800,
    dx=1.0,
    title_prefix="Layered cross-section"
):
    """
    Plot a stratigraphic cross-section with each layer in a different color.
    """
    nt, nx, ny = z_layers.shape
    if axis == "x":
        i = nx // 2 if idx is None else int(idx)
        z_ts = z_layers[:, i, :]           # (nt, width)
        width = ny
    elif axis == "y":
        j = ny // 2 if idx is None else int(idx)
        z_ts = z_layers[:, :, j]           # (nt, width)
        width = nx
    else:
        raise ValueError("axis must be 'x' or 'y'")

    zmin = float(np.min(z_ts))
    zmax = float(np.max(z_ts))
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        raise ValueError("Invalid elevation range for cross-section")

    # Raster image (vertical = elevation, horizontal = distance)
    img = np.ones((nz, width, 3), dtype=float)

    # Use a colormap for layers
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, nt-1))

    # Fill between successive surfaces, each layer a different color
    for k in range(nt - 1):
        top = z_ts[k+1]     # (width,)
        bot = z_ts[k]       # (width,)
        color = colors[k][:3]  # RGB for this layer
        for x in range(width):
            t = float(top[x])
            b = float(bot[x])
            if t == b:
                continue
            r_top = int( (max(t, b) - zmin) / (zmax - zmin) * (nz - 1) )
            r_bot = int( (min(t, b) - zmin) / (zmax - zmin) * (nz - 1) )
            if r_top == r_bot:
                continue
            img[r_bot:r_top, x, :] = color

    # Plot
    dist = np.arange(width) * dx
    plt.figure(figsize=(11, 6))
    plt.imshow(img[::-1, :, :],
               origin="lower",
               aspect="auto",
               extent=[dist[0], dist[-1] if width>1 else 0.0, zmin, zmax])
    plt.plot(dist, z_ts[-1], 'k', linewidth=1.2)
    plt.xlabel(f"Distance along section [{ 'km' if dx!=1.0 else 'index'}]")
    plt.ylabel("Elevation [m]")
    plt.title(f"{title_prefix} (axis={axis}, idx={i if axis=='x' else j})")
    plt.tight_layout()
    plt.show()

# Example usage:
plot_layered_cross_section(z_layers, p_layers, times, axis="x", idx=None, nz=800, dx=1.0,
                          title_prefix="Layered cross-section")
plot_layered_cross_section(z_layers, p_layers, times, axis="y", idx=None, nz=800, dx=1.0)
# ...existing code...
# %%
