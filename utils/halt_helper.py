import torch
from typing import Dict, List, Literal, Optional, Tuple, Any
import os

CheckMode = Literal["all_points", "mean", "final_point"]
Metric = Literal["l_inf", "l1", "l2"]

def load_yaml(path: str) -> Dict[str, Any]:
    # robust yaml import
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required. Install with `pip install pyyaml`.") from e

    if not os.path.exists(path):
        raise FileNotFoundError(f"halt_config not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"halt_config must parse to a dict, got: {type(cfg)}")
    return cfg


def build_act_head_kwargs_from_yaml(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts YAML dict -> kwargs for ACTNuscenesLossHead.
    Handles n_correct_upper_bound null, std/mean list -> torch tensors, and basic validation.
    """
    allowed_keys = {
        "halt_threshold", "halt_mode", "halt_metric", "n_correct_upper_bound",
        "pred_key", "targets_key", "targets_mask_key",
        "denorm", "traj_reduction"
    }

    # catch unknown top-level keys (optional but helpful)
    known_top_level = allowed_keys | {"name", "loss"}
    unknown = set(cfg.keys()) - known_top_level
    if unknown:
        raise ValueError(f"Unknown top-level keys in halt_config: {sorted(list(unknown))}")

    out: Dict[str, Any] = dict(cfg)  # shallow copy

    # normalize n_correct_upper_bound
    if out.get("n_correct_upper_bound", None) is None:
        out["n_correct_upper_bound"] = None
    else:
        out["n_correct_upper_bound"] = int(out["n_correct_upper_bound"])

    # denorm conversion
    denorm = bool(out.get("denorm", False))
    out["denorm"] = denorm

    # better pass as argument
    # if "std_xy" in out and out["std_xy"] is not None:
    #     out["std_xy"] = torch.tensor(out["std_xy"], dtype=torch.float32)
    # if "mean_xy" in out and out["mean_xy"] is not None:
    #     out["mean_xy"] = torch.tensor(out["mean_xy"], dtype=torch.float32)

    # if denorm:
    #     if out.get("std_xy", None) is None or out.get("mean_xy", None) is None:
    #         raise ValueError("halt_config has denorm:true but std_xy/mean_xy missing.")
    #     if out["std_xy"].numel() != 2 or out["mean_xy"].numel() != 2:
    #         raise ValueError("std_xy and mean_xy must have length 2 (for x,y).")

    # defaults if omitted
    out.setdefault("halt_threshold", 2.0)
    out.setdefault("halt_mode", "mean")
    out.setdefault("halt_metric", "l2")
    out.setdefault("pred_key", "pred")
    out.setdefault("targets_key", "targets")
    out.setdefault("targets_mask_key", "targets_mask")
    out.setdefault("traj_reduction", "sum")
    # out.setdefault("verbose", False)

    return out

def agent_traj_correct(
    pred: torch.Tensor,          # [B, A, H, 2]
    targets: torch.Tensor,       # [B, H, A, 7]  (we use only :2)
    targets_mask: torch.Tensor,  # [B, H, A]     (1 valid, 0 pad)
    threshold: float,
    mode: CheckMode = "all_points",
    metric: Metric = "l2",
    return_bool: bool = False,
    denorm: bool = False, # set True if inputs are normalized and need to be denormalized
    std_xy: Optional[torch.Tensor] = None,    # ideally [2]
    mean_xy: Optional[torch.Tensor] = None,   # ideally [2]
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      correct:    [B, A] (int {0,1} by default, or bool if return_bool=True)
      agent_valid:[B, A] (bool) True iff the agent has at least 1 valid future step
    """

    # --- align shapes ---
    if pred.ndim != 4 or pred.size(-1) != 2:
        raise ValueError(f"pred must be [B,A,H,2], got {tuple(pred.shape)}")
    if targets.ndim != 4 or targets.size(-1) < 2:
        raise ValueError(f"targets must be [B,H,A,7] (>=2), got {tuple(targets.shape)}")
    if targets_mask.ndim != 3:
        raise ValueError(f"targets_mask must be [B,H,A], got {tuple(targets_mask.shape)}")

    B, A, H, _ = pred.shape

    if denorm:
        pred_xy_denorm = pred[..., :2] * std_xy + mean_xy               # [B, A, H, 2]
        targets_xy_denorm = (targets[..., :2] * std_xy + mean_xy)       # [B, H, A, 2]
        pred = pred_xy_denorm
        targets[..., :2] = targets_xy_denorm[..., :2]

    # targets_xy: [B,H,A,2] -> [B,A,H,2]
    targets_xy = targets[..., :2].permute(0, 2, 1, 3).contiguous()
    # mask: [B,H,A] -> [B,A,H]
    m = targets_mask.permute(0, 2, 1).contiguous()
    m_bool = m.to(dtype=torch.bool)

    # agent is "valid" if it has any valid timestep
    agent_valid = m_bool.any(dim=-1)  # [B,A]

    # --- per-timestep distance ---
    diff = pred - targets_xy  # [B,A,H,2]

    if metric == "l_inf":
        per_point = diff.abs().amax(dim=-1)                # max(|dx|,|dy|)
    elif metric == "l1":
        per_point = diff.abs().sum(dim=-1)                 # |dx|+|dy|
    elif metric == "l2":
        per_point = torch.sqrt((diff * diff).sum(dim=-1))  # sqrt(dx^2+dy^2)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    thr = torch.as_tensor(threshold, device=pred.device, dtype=per_point.dtype)

    # --- modes ---
    if mode == "all_points":
        # Option 1: all valid points must be <= threshold
        ok_t = (per_point <= thr) | (~m_bool)   # invalid timesteps don't count against you
        correct = ok_t.all(dim=-1) & agent_valid
        if verbose:
            print("All points correct:", correct)
    elif mode == "mean":
        # Option 2: mean over valid points must be <= threshold
        denom = m.to(per_point.dtype).sum(dim=-1).clamp_min(1.0)     # [B,A]
        mean_err = (per_point * m.to(per_point.dtype)).sum(dim=-1) / denom
        correct = (mean_err <= thr) & agent_valid
        if verbose:
            print("Denom:", denom, " Mean_err:", mean_err)
    elif mode == "final_point":
        # Option 3: last valid timestep must be <= threshold
        B, A, H = m_bool.shape
        t_idx = torch.arange(H, device=pred.device).view(1, 1, H)
        last_idx = (t_idx * m_bool.long()).amax(dim=-1)  # [B,A]
        last_err = per_point.gather(dim=-1, index=last_idx.unsqueeze(-1)).squeeze(-1)
        correct = (last_err <= thr) & agent_valid
        if verbose:
            print("Last idx:", last_idx, " Last_err:", last_err)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if return_bool:
        return correct, agent_valid
    return correct.long(), agent_valid

def batch_correct_by_agent_count(
    pred: torch.Tensor,          # [B, A, H, 2]
    targets: torch.Tensor,       # [B, H, A, 7]
    targets_mask: torch.Tensor,  # [B, H, A]
    threshold: float,
    mode: CheckMode = "all_points",
    metric: Metric = "l2",
    # If provided: output 1 iff at least N agents are correct.
    # BUT if T valid agents < N, then require all T to be correct (still output 1 if all valid are correct).
    n_correct_upper_bound: Optional[int] = None,
    return_counts: bool = True,
    denorm: bool = False, # set True if inputs are normalized and need to be denormalized
    std_xy: Optional[torch.Tensor] = None,    # ideally [2]
    mean_xy: Optional[torch.Tensor] = None,   # ideally [2]
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
      sample_ok:     [B] int {0,1}  -- whether the sample passes the criterion
      num_correct:   [B] long       -- number of valid agents that are correct
      num_valid:     [B] long       -- number of valid agents
      correct_mask:  [B, A] long {0,1} or None  -- per-agent correctness (only if return_counts=True)

    Behavior:
      - Always counts correctness only over VALID agents (agent has >=1 valid timestep).
      - If n_correct_upper_bound is None:
          sample_ok = 1 iff ALL valid agents are correct (vacuously 1 if no valid agents).
      - If n_correct_upper_bound = N:
          Let T = #valid agents in the sample.
          Required correct agents = min(N, T).
          sample_ok = 1 iff num_correct >= min(N, T).
          (So when T < N, you effectively require "all valid agents are correct".)
    """
    correct_agent, agent_valid = agent_traj_correct(
        pred=pred,
        targets=targets,
        targets_mask=targets_mask,
        threshold=threshold,
        mode=mode,
        metric=metric,
        return_bool=False,
        denorm=denorm,
        std_xy=std_xy,
        mean_xy=mean_xy,
        verbose=verbose,
    )  # correct_agent: [B,A] {0,1}, agent_valid: [B,A] bool

    correct_agent = correct_agent.to(torch.long)
    agent_valid_l = agent_valid.to(torch.long)

    # Count only valid agents
    correct_valid = correct_agent * agent_valid_l  # [B,A]
    num_correct = correct_valid.sum(dim=1)         # [B]
    num_valid = agent_valid_l.sum(dim=1)           # [B]

    if n_correct_upper_bound is None:
        # all valid agents must be correct
        sample_ok = (num_correct == num_valid).to(torch.long)
    else:
        if n_correct_upper_bound < 0:
            raise ValueError("n_correct_upper_bound must be >= 0")
        required = torch.minimum(
            num_valid,
            torch.tensor(n_correct_upper_bound, device=num_valid.device, dtype=num_valid.dtype),
        )
        sample_ok = (num_correct >= required).to(torch.long)

    if return_counts:
        return sample_ok, num_correct, num_valid, correct_agent
    else:
        return sample_ok, num_correct, num_valid, None