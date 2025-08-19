# %%
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np

# -----------------------------
# Helpers: composition transforms
# -----------------------------
def p2eta(p: np.ndarray) -> np.ndarray:
    """
    Additive log-ratio (alr) transform.
    p: shape (K,) or (K, M) with p >= 0 and columns summing to 1.
    Uses the last component as the divisor.
    Returns eta of shape (K-1,) or (K-1, M).
    """
    p = np.asarray(p)
    if p.ndim == 1:
        p = p[:, None]
    K, M = p.shape
    eps = 1e-12
    denom = np.clip(p[-1, :], eps, 1.0)
    eta = np.log(np.clip(p[:-1, :], eps, 1.0) / denom)
    return eta if M > 1 else eta[:, 0]


def eta2p(eta: np.ndarray) -> np.ndarray:
    """
    Inverse alr transform.
    eta: shape (K-1,) or (K-1, M).
    Returns p of shape (K, M) with columns summing to 1.
    """
    eta = np.asarray(eta)
    if eta.ndim == 1:
        eta = eta[:, None]
    Kminus1, M = eta.shape
    y = np.exp(eta)
    y_full = np.vstack([y, np.ones((1, M))])
    p = y_full / np.sum(y_full, axis=0, keepdims=True)
    return p if M > 1 else p[:, 0]


# -----------------------------
# Stubs for your project I/O
# -----------------------------
def write_flw_json(values_and_time: np.ndarray, tag: str, template_dir: str) -> None:
    """
    Write sea-level function for the simulator.
    values_and_time: 2 x T array-like with [values; t].
    """
    os.makedirs(template_dir, exist_ok=True)
    path = os.path.join(template_dir, f"{tag}.json")
    data = {"time": values_and_time[1, :].tolist(),
            "value": values_and_time[0, :].tolist()}
    with open(path, "w") as f:
        json.dump(data, f)


def write_asd_json(values_and_time: np.ndarray, area_sed_fun: np.ndarray, tag: str, template_dir: str) -> None:
    """
    Write sediment supply time curve + spatial source intensity.
    """
    os.makedirs(template_dir, exist_ok=True)
    path = os.path.join(template_dir, f"{tag}.json")
    data = {"time": values_and_time[1, :].tolist(),
            "value": values_and_time[0, :].tolist(),
            "area_intensity": area_sed_fun.tolist()}
    with open(path, "w") as f:
        json.dump(data, f)


def writemap(z0: np.ndarray, tag: str, p0vec: np.ndarray, xyminmax: Tuple[float, float, float, float], tstart: float) -> None:
    """
    Write initial map/state file for the simulator.
    Shape conventions should match your GPM.
    """
    # Implement according to your simulator’s input format.
    # Placeholder: write a small JSON with metadata.
    with open(f"{tag}.map", "w") as f:
        json.dump({
            "tstart": float(tstart),
            "xyminmax": list(map(float, xyminmax)),
            "z0_shape": list(z0.shape),
            "p0vec_len": int(p0vec.size)
        }, f)


def write_ctl_json(tspan: Tuple[float, float], tag: str, template_dir: str, idnum: int) -> None:
    """
    Write control file for a single forward step.
    """
    os.makedirs(template_dir, exist_ok=True)
    path = os.path.join(template_dir, f"{tag}.json")
    data = {"t_start": float(tspan[0]),
            "t_end": float(tspan[1]),
            "id": int(idnum),
            "input": f"input{idnum}.map",
            "output": f"output{idnum}.out"}
    with open(path, "w") as f:
        json.dump(data, f)


def run_simulator(idnum: int) -> None:
    """
    Call the external GPM for the current step.
    Replace this with the actual system call / Python wrapper.
    """
    # Example:
    # subprocess.run(["/path/to/gpm_exec", f"--id={idnum}"], check=True)
    raise NotImplementedError("Hook up your GPM executable here.")


def parse_outfile(path: str, step_index: int) -> np.ndarray:
    """
    Parse simulator flat output for step 'step_index' into a 1-D numpy array.
    Must match the layout expected by the slicing logic below.
    """
    # Replace with your parser. For now, raise to signal missing wiring.
    raise NotImplementedError(f"Implement parsing for {path} (step {step_index}).")


# -----------------------------
# Spec structure (optional helper)
# -----------------------------
@dataclass
class GridSpec:
    dims: Tuple[int, int]


@dataclass
class Spec:
    grid: GridSpec
    bathy_mean: np.ndarray
    bathy_cov: np.ndarray
    isp_prior: np.ndarray            # Dirichlet alpha vector (length K)
    tstart: float
    tend: float
    delta_t: float
    sl_prior_mean: np.ndarray        # shape (T,)
    sl_prior_cov: np.ndarray         # shape (T,T)
    ss_prior_mean: np.ndarray        # shape (T,)
    ss_prior_cov: np.ndarray         # shape (T,T)
    asd_intensity: float


# -----------------------------
# Main function: mkreffun
# -----------------------------
def mkreffun(spec: Spec, idnum: int) -> Dict[str, Any]:
    """
    Generate a reference realization by running the GPM forward model.
    Mirrors the structure and slicing of the MATLAB mkreffun.
    Returns a dict with keys: p1,p2,p3,s1,s2,s3,z,seafun,sedfun,time,grid
    """
    nx, ny = spec.grid.dims
    N = nx * ny

    # Coordinates/extents (100 m cell size)
    xmin, xmax = 0.0, 100.0 * (nx - 1)
    ymin, ymax = 0.0, 100.0 * (ny - 1)
    xyminmax = (ymin, ymax, xmin, xmax)

    # Initial Bathymetry
    z0vec = np.random.multivariate_normal(mean=spec.bathy_mean, cov=spec.bathy_cov)

    # Initial sediment proportions (Dirichlet) and transform to eta
    p0vec = np.random.dirichlet(alpha=spec.isp_prior)  # length K
    s0vec = p2eta(p0vec)                                # length K-1 (unused here but retained for parity)

    # Time axis
    t = np.arange(spec.tstart, spec.tend + spec.delta_t, spec.delta_t, dtype=float)
    nsteps = len(t) - 1

    # Sea level curve: sample from prior and write
    f_SL = np.random.multivariate_normal(mean=spec.sl_prior_mean, cov=spec.sl_prior_cov)
    write_flw_json(np.vstack([f_SL, t]), f"sealevel{idnum}", "../templates/Case5")

    # Sediment supply curve: sample, clip to >= 0, write with spatial source
    f_SS = np.random.multivariate_normal(mean=spec.ss_prior_mean, cov=spec.ss_prior_cov)
    f_SS = np.maximum(0.0, f_SS)
    area_sed_fun = np.zeros((nx, ny), dtype=float)
    area_sed_fun[:2, :] = spec.asd_intensity  # first two rows supply sediment
    write_asd_json(np.vstack([f_SS, t]), area_sed_fun, f"areased{idnum}", "../templates/Case5")

    # Prepare initial mapfile
    z0 = z0vec.reshape(nx, ny)
    writemap(z0, f"input{idnum}", p0vec, xyminmax, t[0])

    # Forecast state per step (flat arrays from simulator)
    Xf: List[np.ndarray] = [None] * nsteps  # type: ignore

    # Time stepping: write ctl, run simulator, parse, chain output->input
    for k in range(nsteps):
        write_ctl_json((t[k], t[k + 1]), f"mkref{idnum}", "../templates/mkref", idnum)

        # ---- RUN YOUR SIMULATOR HERE ----
        run_simulator(idnum)  # implement this
        # ---------------------------------

        out_path = f"output{idnum}.out"
        Xf[k] = parse_outfile(out_path, k + 1)  # 1-based step index for parity with MATLAB
        # Chain output as next input (your project may need a real converter here)
        os.replace(out_path, f"input{idnum}.map")

    # Build reference dict and unpack per-step fields
    reference: Dict[str, Any] = {
        "p1": [None] * nsteps,
        "p2": [None] * nsteps,
        "p3": [None] * nsteps,
        "s1": [None] * nsteps,
        "s2": [None] * nsteps,
        "s3": [None] * nsteps,
        "z":  [None] * nsteps,
        "seafun": f_SL,
        "sedfun": f_SS,
        "time": {
            "start": float(spec.tstart),
            "end": float(spec.tend),
            "steplen": float(spec.delta_t),
            "nsteps": int(nsteps),
        },
        "grid": {
            "dims": (nx, ny),
            "xlims": (xmin, xmax),
            "ylims": (ymin, ymax),
        },
    }

    # Unpack each step: layout must match your simulator’s Xf format
    for i in range(1, nsteps + 1):
        Xi = Xf[i - 1]
        if Xi is None:
            raise RuntimeError(f"Missing simulator output for step {i}.")

        # Expected layout:
        # [ surfaces ( (i+1)*N ),
        #   eta1 ( i*N ),
        #   eta2 ( i*N ),
        #   eta3 ( i*N ) ]
        base = 0
        # surfaces
        z_block = Xi[base : base + (i + 1) * N]
        base += (i + 1) * N
        z_i = z_block.reshape((i + 1, N)).T  # shape N x (i+1)
        reference["z"][i - 1] = z_i

        # s1
        s1_block = Xi[base : base + i * N]
        base += i * N
        s1_i = s1_block.reshape((i, N)).T  # N x i
        reference["s1"][i - 1] = s1_i

        # s2
        s2_block = Xi[base : base + i * N]
        base += i * N
        s2_i = s2_block.reshape((i, N)).T  # N x i
        reference["s2"][i - 1] = s2_i

        # s3
        s3_block = Xi[base : base + i * N]
        base += i * N
        s3_i = s3_block.reshape((i, N)).T  # N x i
        reference["s3"][i - 1] = s3_i

        # Convert eta -> proportions per layer (i) and cell (N)
        # Stack as (3, i*N), invert alr to get (4, i*N), then reshape
        eta_stack = np.vstack([s1_i.T.reshape(1, -1),
                               s2_i.T.reshape(1, -1),
                               s3_i.T.reshape(1, -1)])  # 3 x (i*N)
        p_all = eta2p(eta_stack)  # 4 x (i*N)

        # Extract first three components, reshape to (i, N)
        p1_i = p_all[0, :].reshape(i, N)
        p2_i = p_all[1, :].reshape(i, N)
        p3_i = p_all[2, :].reshape(i, N)

        # Store as in MATLAB: p# are (i x N)
        reference["p1"][i - 1] = p1_i
        reference["p2"][i - 1] = p2_i
        reference["p3"][i - 1] = p3_i

    return reference


# -----------------------------
# Example usage (fill spec fields accordingly)
# -----------------------------
if __name__ == "__main__":
    # Dummy shapes just to illustrate construction; replace with real data.
    nx, ny = 72, 16
    T = 21  # number of time grid points (tstart:delta:tend), so nsteps = T-1
    K = 4   # number of sediment classes

    spec = Spec(
        grid=GridSpec(dims=(nx, ny)),
        bathy_mean=np.zeros(nx * ny),
        bathy_cov=np.eye(nx * ny),
        isp_prior=np.ones(K),  # e.g., uniform Dirichlet
        tstart=0.0,
        tend=20.0,
        delta_t=1.0,
        sl_prior_mean=np.zeros(T),
        sl_prior_cov=np.eye(T),
        ss_prior_mean=np.ones(T),
        ss_prior_cov=np.eye(T),
        asd_intensity=1.0
    )

    # This will raise until you implement run_simulator/parse_outfile
    # reference = mkreffun(spec, idnum=1)
    pass