import torch

@torch.no_grad()
def compute_metrics(pred, targets, targets_mask, only_full=False, history_mask=None, targets_idx=None, out_slice=2, miss_tol=2, return_unreduced=False):
    """
    Returns scalar ADE, FDE, and miss rate (averaged over valid agents+timesteps).
    
    pred: prediction [B, F, AF, 2]
    targets: groundtruth [B, F, AF, 7]
    targets_mask: mask over groundtruth [B, F, AF]
    only_full: compute metrics only on full history/future trajectories
    history_mask: mask over history [B, H, AH]
    targets_idx: indices in history of agents to predict [B, AF]
    out_slice: terminating index for positions
    miss_tol: tolerance for miss rate, assumed same units as pred and targets
    return_unreduced: whether to also return unreduced along batch/agent dimension (True) or only return scalar (False)
    """
    B, H, Ain = history_mask.shape
    _, F, Aout = targets_mask.shape
    # update masking
    if only_full:
        assert history_mask is not None, "Must specify history mask if computing metrics on full tracks"
        if targets_mask.shape[2] != history_mask.shape[2]: assert targets_idx is not None, "Must specify target indices if predicting fewer output agents than input agents"

        if targets_idx is not None and Aout < Ain:
            history_mask = history_mask.gather(dim=2, index=targets_idx[:, None, :].expand(-1,H,-1)) # [B, H, AF]
        full_history = torch.all(history_mask, dim=1) # [B, AF]
        full_future = torch.all(targets_mask, dim=1) # [B, AF]
        full_track = full_history & full_future # [B, AF]

    tgt = targets[..., :out_slice].permute(0, 2, 1, 3).contiguous()         # [B,AF,F,2]
    if only_full:
        m = full_track[:, :, None].expand(-1, -1, targets_mask.shape[1]).to(pred.dtype).contiguous() # [B, AF, F]
    else:
        m = targets_mask.permute(0, 2, 1).to(pred.dtype).contiguous()           # [B,AF,F]
    
    # compute distances
    pred_xy = pred[..., :out_slice]
    dist = torch.linalg.norm(pred_xy - tgt, dim=-1)                         # [B,AF,F]

    # compute ade
    ade_over_time = (dist * m).sum(dim=2) / (m.sum(dim=2) + 1e-6) # [B, AF]
    
    agent_mask = m.sum(dim=2) > 0 # [B, AF], valid agents are those present for at least one time step
    ade_over_agents = (ade_over_time * agent_mask).sum(dim=1) / (agent_mask.sum(dim=1) + 1e-6) # [B, ]

    batch_mask = agent_mask.sum(dim=1) > 0 # [B, ], valid samples are those with at least one valid agent
    ade = ade_over_agents[batch_mask].mean()

    # compute fde (last horizon step only)
    dist_last = dist[:, :, -1] # [B, AF]
    m_last = m[:, :, -1] # [B, AF]

    fde_over_agents = (dist_last * m_last).sum(dim=1) / (m_last.sum(dim=1) + 1e-6) # [B, ]
    
    fde_batch_mask = m_last.sum(dim=1) > 0 # [B, ], valid samples are those with at least one valid agent at last step
    fde = fde_over_agents[fde_batch_mask].mean()

    # compute miss rate
    max_dist = torch.max(dist, dim=-1).values # [B, AF]
    misses = max_dist >= miss_tol # [B, AF]
    mr_over_agents = (misses * agent_mask).sum(dim=-1) / (agent_mask.sum(dim=1) + 1e-6) # [B, ]
    mr = mr_over_agents[batch_mask].mean()

    if not return_unreduced:
        return ade.item(), fde.item(), mr.item()
    else:
        return ade.item(), fde.item(), mr.item(), \
               ade_over_agents[batch_mask], fde_over_agents[fde_batch_mask], mr_over_agents[batch_mask], \
               ade_over_time[agent_mask], dist_last[m_last.to(bool)], misses[agent_mask]

@torch.no_grad()
def compute_menger_curvature(trajectory, mask):
    '''
    Computes maximum curvature over time and agents Menger curvature. The Menger
    curvature of three points xyz is the reciprocal of the radius of circumcircle 
    of xyz. For each agent trajectory, compute the maximum Menger curvature over
    sliding sets of xyz points. Returns the maximum curvature across all agents.

    trajectory: [B, T, A, 7]
    mask: [B, T, A]

    returns curvatures [B]
    '''
    if trajectory is None or mask is None:
        raise ValueError("trajectory and mask must be provided")

    B, T, A, _ = trajectory.shape
    if T < 3:
        # Not enough points to form a triplet
        return float(trajectory.new_tensor(0.0))

    # Positions in 2D (x,y)
    pos = trajectory[..., :2].to(dtype=torch.float32)  # [B,T,A,2]
    m = mask.to(dtype=torch.bool)                     # [B,T,A]

    # Consecutive triplets: a=pos[t], b=pos[t+1], c=pos[t+2]
    a = pos[:, :-2]     # [B,T-2,A,2]
    b = pos[:, 1:-1]
    c = pos[:, 2:]

    # Triplet validity: all three timesteps valid
    triplet_valid = m[:, :-2] & m[:, 1:-1] & m[:, 2:]  # [B,T-2,A]

    # Edge vectors
    ab = b - a
    bc = c - b
    ca = a - c
    ac = c - a

    # Side lengths
    lab = torch.linalg.norm(ab, dim=-1)  # [B,T-2,A]
    lbc = torch.linalg.norm(bc, dim=-1)
    lca = torch.linalg.norm(ca, dim=-1)

    # 2D cross product magnitude of (b-a) x (c-a)
    cross = ab[..., 0] * ac[..., 1] - ab[..., 1] * ac[..., 0]
    cross_abs = cross.abs()

    # Menger curvature k = 4*Area/(lab*lbc*lca); in 2D Area = 0.5*|cross| => k = 2*|cross|/(lab*lbc*lca)
    denom = (lab * lbc * lca).clamp_min(1e-6)
    curvature = (2.0 * cross_abs) / denom  # [B,T-2,A]

    # Zero-out invalid triplets (also handles masked timesteps)
    curvature = curvature.masked_fill(~triplet_valid, 0.0)

    # Max over time triplets and agents
    max_curv = curvature.amax(dim=(1,2))  # scalar tensor
    return max_curv