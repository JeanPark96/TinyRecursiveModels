from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn


class VideoTRMACTDenseLossHead(nn.Module):
    def __init__(self, model: nn.Module, halt_iou_thr=0.4, halt_eps=1e-5):
        super().__init__()
        self.model = model
        self.loss_fn = compute_dense_tr_loss
        self.halt_iou_thr = halt_iou_thr 
        self.halt_eps = halt_eps        
        self.halt_eps_iou = 1e-3 
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        
        # 1. Forward Pass
        new_carry, outputs = self.model(**model_kwargs)
        
        # Unpack targets
        target_start_sec = new_carry.current_data["i0"]
        target_end_sec = new_carry.current_data["i1"]
        target_mask = new_carry.current_data["video_mask"]
        
        # --- HANDLE PYRAMID TUPLES ---
        logits_tuple = outputs["logits"]
        offsets_tuple = outputs["extra_logits"]
        
        # Ensure tuples
        if isinstance(logits_tuple, torch.Tensor):
            logits_tuple = (logits_tuple,)
            offsets_tuple = (offsets_tuple,)
        
        # 2. Logic (No Gradients)
        with torch.no_grad():
            # A. Calculate Current Loss (Sum over Levels)
            current_loss_total = 0.0
            base_stride = 1.0 
            
            for i, (l_logits, l_offsets) in enumerate(zip(logits_tuple, offsets_tuple)):
                curr_stride = base_stride * (2 ** i)
                if i == 0:
                    curr_mask = target_mask
                else:
                    curr_mask = F.interpolate(
                        target_mask.unsqueeze(1).float(), 
                        size=l_logits.shape[1], 
                        mode='nearest'
                    ).squeeze(1).bool()

                l_loss, _ = batch_compute_dense_tr_loss(
                    l_logits, l_offsets, target_start_sec, target_end_sec, curr_mask,
                    sec_per_step=curr_stride
                )
                current_loss_total += l_loss

            # B. Decode Predictions for IoU (Use Level 0 Only)
            l0_logits = logits_tuple[0]
            l0_offsets = offsets_tuple[0]
            
            best_idx = torch.argmax(l0_logits, dim=1)
            b_ids = torch.arange(best_idx.shape[0], device=best_idx.device)
            offsets = l0_offsets[b_ids, best_idx]
            
            pred_s = best_idx - offsets[:, 0]
            pred_e = best_idx + offsets[:, 1]
            ious = temporal_iou_1d(pred_s, pred_e, target_start_sec, target_end_sec)
            
            # C. Determine Halting Signal
            is_success = (ious >= self.halt_iou_thr)
            prev_loss = new_carry.prev_loss
            delta = prev_loss - current_loss_total
            is_stuck = (delta.abs() < self.halt_eps)

            prev_iou = new_carry.prev_iou
            delta_iou = prev_iou - ious

            
            # should_halt = (is_success | is_stuck).float()
            should_halt = (delta / prev_loss.clamp_min(1e-6)) <= self.halt_eps
            should_halt_iou = (delta_iou / prev_iou.clamp_min(1e-6)) > self.halt_eps_iou
            should_halt = should_halt
            # should_continue = 1.0 - should_halt
            should_continue = (delta / prev_loss.clamp_min(1e-6)) > self.halt_eps
            # Update history
            new_carry.prev_loss = current_loss_total.detach()
            new_carry.prev_iou = ious.detach()

            # Metrics
            valid = new_carry.halted 
            metrics = {
                "count": new_carry.halted.sum(),
                "halted_loss": torch.where(valid, current_loss_total, 0).sum(),
                "halt_rate": torch.where(valid, should_halt, 0).sum(),
                "steps": torch.where(valid, new_carry.steps, 0).sum(),
                "mean_iou": ious[valid].mean() if valid.any() else torch.tensor(0.0, device=ious.device),
            }

        # 3. Calculate Losses (With Gradients)
        total_dense_loss = 0.0
        for i, (l_logits, l_offsets) in enumerate(zip(logits_tuple, offsets_tuple)):
            curr_stride = base_stride * (2 ** i)
            if i == 0:
                curr_mask = target_mask
            else:
                curr_mask = F.interpolate(
                    target_mask.unsqueeze(1).float(), 
                    size=l_logits.shape[1], 
                    mode='nearest'
                ).squeeze(1).bool()

            lm, _ = compute_dense_tr_loss(
                l_logits, l_offsets, target_start_sec, target_end_sec, curr_mask,
                sec_per_step=curr_stride
            )
            total_dense_loss += lm.sum()

        # Q-Head Loss
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            should_halt.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )

        metrics.update({
            "loss": total_dense_loss.detach(),
            "q_halt_loss": q_halt_loss.detach()
        })

        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                should_continue.to(outputs["q_continue_logits"].dtype),
                reduction="sum",
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # --- FIX: Detach Logic for Tuples ---
        detached_outputs = {}
        for k in return_keys:
            if k in outputs:
                val = outputs[k]
                if isinstance(val, (tuple, list)):
                    # Option A: Return Tuple of detached tensors (Correct behavior)
                    detached_outputs[k] = tuple(v.detach() for v in val)
                    
                    # Option B (ALTERNATIVE): Return just Level 0 to prevent downstream crashes 
                    # If your metric function crashes on tuples, uncomment the line below instead:
                    # detached_outputs[k] = val[0].detach() 
                elif isinstance(val, torch.Tensor):
                    detached_outputs[k] = val.detach()
                else:
                    detached_outputs[k] = val

        if "debug_stats" in outputs:
            detached_outputs["debug_stats"] = outputs["debug_stats"]

        # Debug Printout
        if self.training and new_carry.steps[0] > 0 and (new_carry.steps[0] % 4 == 0):
             n_print = min(3, current_loss_total.shape[0])
             print(f"\n[Step {new_carry.steps[0].item()}] Debug:")
             for i in range(n_print):
                 print(f"S{i}: IoU={ious[i]:.2f} (Thr={self.halt_iou_thr}) | Delta={delta[i]:.4f}\n"
                       f"-> Target={should_halt[i].item()} | PredProb={torch.sigmoid(outputs['q_halt_logits'][i]).item():.4f}")
                 #print(should_halt_iou)
                 print(pred_s[i], pred_e[i], "     ", target_start_sec[i], [target_end_sec[i]])

        return (
            new_carry,
            total_dense_loss + 0.5 * (q_halt_loss + q_continue_loss),
            metrics,
            detached_outputs,
            new_carry.halted.all(),
        )
    
def temporal_iou_1d(ps, pe, gs, ge, eps=1e-6):
    # all are (B,) continuous seconds
    # ensure ordering
    p_start = torch.minimum(ps, pe)
    p_end   = torch.maximum(ps, pe)
    g_start = torch.minimum(gs, ge)
    g_end   = torch.maximum(gs, ge)

    inter = torch.clamp(torch.minimum(p_end, g_end) - torch.maximum(p_start, g_start), min=0.0)
    union = torch.clamp(torch.maximum(p_end, g_end) - torch.minimum(p_start, g_start), min=eps)
    return inter / union  # (B,)

def gaussian_soft_labels(gt_idx: torch.Tensor, T: int, sigma: float, mask: torch.Tensor):
    # gt_idx: (B,) long in [0,T-1]; mask: (B,T)
    B = gt_idx.shape[0]
    t = torch.arange(T, device=gt_idx.device)[None, :].float()  # (1,T)
    mu = gt_idx[:, None].float()  # (B,1)
    y = torch.exp(-0.5 * ((t - mu) / sigma) ** 2)  # (B,T)
    y = y * mask.float()
    y = y / (y.sum(dim=-1, keepdim=True) + 1e-8)
    return y

def get_center_sampling_mask(grid, gt_s_idx, gt_e_idx, radius=1.5):
    """
    Generates a mask selecting only the center regions of actions.
    grid: (B, T)
    gt_s_idx: (B,) start indices
    gt_e_idx: (B,) end indices
    radius: float, radius (in stride units) to sample around center. 
            SnAG/ActionFormer typically use 1.5.
    """
    # 1. Compute Center and Width
    # unsqueeze to broadcast against grid (B, 1)
    centers = (gt_s_idx + gt_e_idx) / 2.0
    centers = centers.unsqueeze(1) # (B, 1)
    
    # 2. Define Window
    # We use a fixed radius (1.5 tokens) regardless of action length, 
    # similar to ActionFormer/FCOS.
    # Alternatively, you can use min(radius, width/2) to handle tiny actions.
    left_bound = centers - radius
    right_bound = centers + radius
    
    # 3. Create Mask
    # Select tokens strictly inside the center window
    mask = (grid >= left_bound) & (grid <= right_bound)
    
    # 4. SAFETY: Ensure at least one token is selected per action
    # If action is smaller than the grid (e.g. sub-token length), the window might be empty.
    # We force the closest token to be positive.
    distances = torch.abs(grid - centers) # (B, T)
    closest_indices = distances.argmin(dim=1) # (B,)
    
    # Create a mask for the single closest token
    closest_mask = torch.zeros_like(mask, dtype=torch.bool)
    closest_mask.scatter_(1, closest_indices.unsqueeze(1), True)
    
    # Combine: Use window, but fallback/add closest token
    mask = mask | closest_mask
    
    # 5. Sanity Check: Don't sample outside the original Ground Truth box
    # (Just in case radius > actual length)
    gt_mask = (grid >= gt_s_idx.unsqueeze(1)) & (grid <= gt_e_idx.unsqueeze(1))
    mask = mask & gt_mask
    
    return mask

def batch_compute_dense_tr_loss(
    cls_logits,         # (B, T)
    reg_offsets,        # (B, T, 2)
    gt_start_sec,       # (B,)
    gt_end_sec,         # (B,)
    video_mask,         # (B, T)
    sec_per_step=1.0,
    lambda_cls=1.0,
    lambda_reg=1.0,
    return_per_sample=True,
    center_radius=1.5   # New param for sampling width
):
    B, T = cls_logits.shape
    device = cls_logits.device
    
    # 1. Generate Targets (Indices)
    gt_s_idx = (gt_start_sec / sec_per_step).round().float() # keep float for center calc
    gt_e_idx = (gt_end_sec / sec_per_step).round().float()
    
    grid = torch.arange(T, device=device).unsqueeze(0).expand(B, T).float()
    
    # --- Generate Center Mask (The "Positive" Samples) ---
    # This determines WHERE we train.
    pos_mask = get_center_sampling_mask(grid, gt_s_idx, gt_e_idx, radius=center_radius)
    pos_mask = pos_mask & video_mask.bool() # Apply video padding mask
    
    # --- Classification Targets ---
    # Strategy: "Hard" Assignment
    # 1.0 (Action) if inside Center Mask
    # 0.0 (Background) if outside Center Mask
    # This teaches the model to only "fire" when it's confidently in the middle.
    target_cls = pos_mask.float() 
    
    # --- Regression Targets ---
    # target_off_l = t - start
    # target_off_r = end - t
    target_off_l = grid - gt_s_idx.unsqueeze(1)
    target_off_r = gt_e_idx.unsqueeze(1) - grid
    target_reg = torch.stack([target_off_l, target_off_r], dim=-1).float() # (B, T, 2)
    
    # --- Loss Calculation ---
    
    # 1. Classification (Focal Loss)
    # We apply classification loss to ALL valid video frames (FG + BG)
    # Background frames are pushed to 0, Center frames are pushed to 1.
    # Note: Edge frames (inside action but outside center) are treated as BG here, 
    # which suppresses ambiguity.
    L_cls_map = sigmoid_focal_loss(cls_logits, target_cls, alpha=0.25, gamma=2.0, reduction='none')
    L_cls_map = L_cls_map * video_mask.float()
    
    # 2. Regression (GIoU Loss)
    # Initialize with ZEROS.
    L_reg_map = torch.zeros_like(cls_logits) # (B, T)
    
    if pos_mask.sum() > 0:
        # Extract ONLY positive center samples
        pred_reg_pos = reg_offsets[pos_mask]   
        target_reg_pos = target_reg[pos_mask]  
        
        # Compute loss on subset
        loss_pos = ctr_giou_loss(pred_reg_pos, target_reg_pos, reduction='none') 
        
        # Assign back
        L_reg_map.view(-1)[pos_mask.view(-1)] = loss_pos
    
    if return_per_sample:
        # Sum over time to get (B,)
        L_cls_per_sample = L_cls_map.sum(dim=1)
        L_reg_per_sample = L_reg_map.sum(dim=1)
        
        # Normalization (Optional but recommended): Divide by number of centers per sample
        # num_pos_per_sample = pos_mask.sum(dim=1).clamp(min=1.0)
        # per_sample_loss = (lambda_cls * L_cls_per_sample + lambda_reg * L_reg_per_sample) / num_pos_per_sample
        
        per_sample_loss = (lambda_cls * L_cls_per_sample) + (lambda_reg * L_reg_per_sample)
        return per_sample_loss, {}
    else:
        num_pos = pos_mask.sum().clamp(min=1.0)
        total_loss = ((lambda_cls * L_cls_map.sum()) + (lambda_reg * L_reg_map.sum())) / num_pos
        return total_loss, {"loss": total_loss}

def compute_dense_tr_loss(
    cls_logits,         # (B, T)
    reg_offsets,        # (B, T, 2)
    gt_start_sec,       # (B,)
    gt_end_sec,         # (B,)
    video_mask,         # (B, T)
    sec_per_step=1.0,
    lambda_cls=1.0,
    lambda_reg=1.0,
    center_radius=1.5   # New param
):
    B, T = cls_logits.shape
    device = cls_logits.device
    
    # 1. Generate Targets
    gt_s_idx = (gt_start_sec / sec_per_step).round().float()
    gt_e_idx = (gt_end_sec / sec_per_step).round().float()
    
    grid = torch.arange(T, device=device).unsqueeze(0).expand(B, T).float()
    
    # --- POS MASK (Center Sampling) ---
    pos_mask = get_center_sampling_mask(grid, gt_s_idx, gt_e_idx, radius=center_radius)
    pos_mask = pos_mask & video_mask.bool()

    # --- Targets ---
    target_cls = pos_mask.float()
    
    target_off_l = grid - gt_s_idx.unsqueeze(1)
    target_off_r = gt_e_idx.unsqueeze(1) - grid
    target_reg = torch.stack([target_off_l, target_off_r], dim=-1).float()
    
    # --- A. Focal Loss ---
    valid_mask = video_mask.view(-1)
    cls_flat = cls_logits.view(-1)[valid_mask]
    target_cls_flat = target_cls.view(-1)[valid_mask]
    
    L_cls = sigmoid_focal_loss(
        cls_flat, 
        target_cls_flat, 
        alpha=0.25, 
        gamma=2.0, 
        reduction='sum'
    )
    
    # Normalize by number of POSITIVES (Centers), not total tokens
    num_pos = pos_mask.sum().clamp(min=1.0)
    L_cls = L_cls / num_pos
    
    # --- B. GIoU Loss ---
    if pos_mask.sum() > 0:
        pred_reg_pos = reg_offsets[pos_mask]
        target_reg_pos = target_reg[pos_mask]
        
        L_reg = ctr_giou_loss(
            pred_reg_pos, 
            target_reg_pos, 
            reduction='sum'
        )
        L_reg = L_reg / num_pos
    else:
        L_reg = torch.tensor(0.0, device=device)

    total_loss = (lambda_cls * L_cls) + (lambda_reg * L_reg)
    
    return total_loss, {
        "loss": total_loss, 
        "L_cls": L_cls, 
        "L_reg": L_reg,
        "num_pos": num_pos
    }



from typing import Sequence, Tuple, Dict, Optional, Any


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2.0,
    smoothing: bool = True,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Taken from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
    Args:
        inputs: A float tensor of arbitrary shape.
                The prediction logits for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    mask = (targets >= 0.5).float()     # positive mask

    p = torch.sigmoid(inputs)
    if smoothing:
        p_t = p * targets + (1 - p) * (1 - targets)
    else:
        p_t = p * mask + (1 - p) * (1 - mask)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * (1 - p_t) ** gamma

    if alpha >= 0:
        alpha_t = alpha * mask + (1 - alpha) * (1 - mask)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def ctr_giou_loss(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Generalized Intersection over Union Loss (Hamid Rezatofighi et al.)
    https://arxiv.org/abs/1902.09630
    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0
    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py
    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # giou is reduced to iou in our setting, skip unnecessary steps
    loss = 1.0 - iouk

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

@torch.jit.script
def ctr_diou_loss(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et al.)
    https://arxiv.org/abs/1911.08287
    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0
    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py
    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # offset between centers
    rho = 0.5 * (rp - lp - rg + lg)

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
