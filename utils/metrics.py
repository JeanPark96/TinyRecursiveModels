import torch

@torch.no_grad()
def compute_metrics(pred, targets, targets_mask, only_full=False, history_mask=None, targets_idx=None, out_slice=2, miss_tol=2):
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

    return ade.item(), fde.item(), mr.item()