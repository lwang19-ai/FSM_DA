import numpy as np
from model import ForwardParams, run_forward

def simulate_member(args):
    """
    Run forward model for one ensemble member.
    
    Args:
        args: tuple containing:
            i: member index
            nx: number of points in east direction
            ny: number of points in north direction
            current_time: timestep array
            hSL_val: sea level value
            hSS_val: sediment supply value
            param_vec: parameter vector [diffusion_coeff, erosion_rate, subsidence_scalar]
            z0_i: initial elevation field (ny, nx)
            p0_i: initial proportions (4,)
    
    Returns:
        z_out: elevation field (ny, nx)
        p_out: proportions field (ny, nx, 4)
        d_out: deposit thickness (ny, nx)
        c_out: deposit composition (ny, nx, 4)
    """
    i, nx, ny, current_time, hSL_val, hSS_val, param_vec, z0_i, p0_i = args
    
    # Create ForwardParams instance with consistent naming
    params_i = ForwardParams(
        nx=nx,                    # points in east direction
        ny=ny,                    # points in north direction
        times=current_time,
        diffusion_coeff=param_vec[0],
        erosion_rate=param_vec[1],
        subsidence_scalar=param_vec[2],
        # Use updated parameter names for consistency
        bowl_center_east=0.20,    # default center in east direction
        bowl_center_north=0.20,   # default center in north direction
        bowl_width_east=0.22,     # default width in east direction
        bowl_width_north=0.50     # default width in north direction
    )
    
    # Run forward model (note: z0_i should be shape (ny, nx))
    z_layers_i, p_layers_i, deposits_i, compositions_i = run_forward(
        z0_i,
        p0_i, 
        current_time, 
        np.array([hSL_val]), 
        current_time, 
        np.array([hSS_val]), 
        params_i
    )
    
    # Extract results (maintaining ny, nx order)
    z_out = z_layers_i[0]        # shape: (ny, nx)
    p_out = p_layers_i[0]        # shape: (ny, nx, 4)
    d_out = deposits_i[0]        # shape: (ny, nx)
    c_out = compositions_i[0]    # shape: (ny, nx, 4)
    
    return z_out, p_out, d_out, c_out

def gaspari_cohn(r, c):
    """
    Gaspari–Cohn taper ρ(r; c). r: distance, c: localization radius.
    Returns ρ in [0,1]. Vectorized.
    """
    r = np.asarray(r, dtype=float)
    x = np.abs(r) / float(c + 1e-12)
    rho = np.zeros_like(x)
    mask1 = (x <= 1.0)
    mask2 = (x > 1.0) & (x <= 2.0)
    x1 = x[mask1]
    x2 = x[mask2]
    rho[mask1] = 1 + x1**2*( -5/3 + 0.5*x1 + 0.25*x1**2 ) + x1**5*(-0.5/5)
    rho[mask2] = 4 - 5*x2 + (5/3)*x2**2 + 0.5*x2**3 - (1/6)*x2**4 + (1/12)*x2**5
    rho[mask2] *= (1/ x2**5)
    rho[x > 2.0] = 0.0
    return np.clip(rho, 0.0, 1.0)

def build_obs_meta(ny, nx, obs_idx, stride_z=4, stride_p=8, stride_d=8, stride_c=16,
                   use_z=True, use_p=True, use_d=False, use_c=False):
    keep_idx = np.asarray(obs_idx, dtype=np.int64)
    pos_map = {k: i for i, k in enumerate(keep_idx)}  # absolute -> column in obs

    off_z = 0
    off_p = off_z + ny*nx
    off_d = off_p + ny*nx*4
    off_c = off_d + ny*nx

    oy_list, ox_list, ot_list, oc_list, col_list = [], [], [], [], []

    def gen_locs(stride):
        ys = np.arange(0, ny, stride)
        xs = np.arange(0, nx, stride)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        return yy, xx

    def append_entries(yy, xx, base_flat, otype_base, is_4comp=False):
        sel_mask = np.isin(base_flat, keep_idx)
        if not np.any(sel_mask):
            return
        sel_idx = np.nonzero(sel_mask)[0]
        matched_abs = base_flat[sel_idx]
        cols = np.array([pos_map[a] for a in matched_abs], dtype=int)

        if is_4comp:
            by = np.repeat(yy.ravel(), 4)[sel_idx]
            bx = np.repeat(xx.ravel(), 4)[sel_idx]
            bcomp = np.tile(np.arange(4), yy.size)[sel_idx]
            ot = np.full(sel_idx.size, otype_base, int) + bcomp  # 1..4 or 6..9
        else:
            by = yy.ravel()[sel_idx]
            bx = xx.ravel()[sel_idx]
            bcomp = np.full(sel_idx.size, -1, int)
            ot = np.full(sel_idx.size, otype_base, int)

        oy_list.append(by); ox_list.append(bx)
        ot_list.append(ot); oc_list.append(bcomp)
        col_list.append(cols)

    if use_z:
        yy, xx = gen_locs(stride_z)
        base = off_z + (yy * nx + xx)
        append_entries(yy, xx, base.ravel(), 0, is_4comp=False)

    if use_p:
        yy, xx = gen_locs(stride_p)
        base = off_p + ((yy * nx + xx)[:, None] * 4 + np.arange(4)[None, :])
        append_entries(yy, xx, base.ravel(), 1, is_4comp=True)

    if use_d:
        yy, xx = gen_locs(stride_d)
        base = off_d + (yy * nx + xx)
        append_entries(yy, xx, base.ravel(), 5, is_4comp=False)

    if use_c:
        yy, xx = gen_locs(stride_c)
        base = off_c + ((yy * nx + xx)[:, None] * 4 + np.arange(4)[None, :])
        append_entries(yy, xx, base.ravel(), 6, is_4comp=True)

    if len(col_list) == 0:
        return (np.empty((0,), int),)*5

    oy = np.concatenate(oy_list)
    ox = np.concatenate(ox_list)
    otype = np.concatenate(ot_list)
    ocomp = np.concatenate(oc_list)
    cols = np.concatenate(col_list)
    return oy.astype(int), ox.astype(int), otype.astype(int), ocomp.astype(int), cols.astype(int)

def state_indices_for_cell(iy, ix, ny, nx, num_params):
    """
    Return the indices (in the big state vector ensemble_vectors) for this cell:
      z(i), p(i,0..3), deposits(i), compositions(i,0..3)
    Layout of ensemble_vectors row:
      [params | z(:) | p(:) | deposits(:) | compositions(:)]
    """
    off = num_params
    idxs = []

    # z offset
    off_z = off
    idxs.append(off_z + (iy * nx + ix))

    # p offset (4 comps per cell)
    off_p = off_z + ny * nx
    base = (iy * nx + ix) * 4
    idxs.extend([off_p + base + c for c in range(4)])

    # deposits offset
    off_d = off_p + ny * nx * 4
    idxs.append(off_d + (iy * nx + ix))

    # compositions offset (4 comps)
    off_c = off_d + ny * nx
    basec = (iy * nx + ix) * 4
    idxs.extend([off_c + basec + c for c in range(4)])

    return np.array(idxs, dtype=np.int64)

# helper function to build observation index
def build_obs_index(ny, nx, stride_z=4, stride_p=8, stride_d=8, stride_c=16,
                    use_z=True, use_p=True, use_d=False, use_c=False):
    """
    Build index into the concatenated observation vector:
    [z(:), p(:), deposits(:), compositions(:)]  (no parameters)
    Arrays are flattened in C-order from shapes:
      z: (ny, nx)
      p: (ny, nx, 4)
      d: (ny, nx)
      c: (ny, nx, 4)
    Returns an index array idx_obs to subselect observations.
    """
    idx = []

    # helper: grid subsampling indices for (ny, nx) flattened
    def grid_idx(stride):
        ys = np.arange(0, ny, stride)
        xs = np.arange(0, nx, stride)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        return (yy * nx + xx).ravel()

    # offsets inside the concatenated obs vector (excluding parameters)
    off_z = 0
    off_p = off_z + ny*nx
    off_d = off_p + ny*nx*4
    off_c = off_d + ny*nx

    if use_z:
        g = grid_idx(stride_z)
        idx.extend((off_z + g).tolist())

    if use_p:
        g = grid_idx(stride_p)                      # base (y,x) indices
        comp_offsets = np.arange(4)                 # last axis is fastest
        gp = (g[:, None] * 4 + comp_offsets[None, :]).ravel()
        idx.extend((off_p + gp).tolist())

    if use_d:
        g = grid_idx(stride_d)
        idx.extend((off_d + g).tolist())

    if use_c:
        g = grid_idx(stride_c)
        comp_offsets = np.arange(4)
        gc = (g[:, None] * 4 + comp_offsets[None, :]).ravel()
        idx.extend((off_c + gc).tolist())

    return np.asarray(idx, dtype=np.int64)















if __name__ == '__main__':
    pass