from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math
#from utils.helper import *
IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim=-1):
    # logits: (B, T), mask: (B, T) bool
    very_neg = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~mask, very_neg)
    return F.softmax(logits, dim=dim)

def soft_expectation(p: torch.Tensor, time_grid: torch.Tensor):
    # p: (B, T), time_grid: (T,)
    return (p * time_grid[None, :]).sum(dim=-1)  # (B,)



def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim=-1):
    # logits: (B, T), mask: (B, T) bool
    very_neg = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~mask, very_neg)
    return F.softmax(logits, dim=dim)

def soft_expectation(p: torch.Tensor, time_grid: torch.Tensor):
    # p: (B, T), time_grid: (T,)
    return (p * time_grid[None, :]).sum(dim=-1)  # (B,)

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

# def batch_compute_dense_tr_loss(
#     cls_logits,         # (B, T)
#     reg_offsets,        # (B, T, 2)
#     gt_start_sec,       # (B,)
#     gt_end_sec,         # (B,)
#     video_mask,         # (B, T)
#     sec_per_step=1.0,
#     lambda_cls=1.0,
#     lambda_reg=1.0,
#     return_per_sample=True # Add this flag
# ):
#     B, T = cls_logits.shape
#     device = cls_logits.device
    
#     # 1. Generate Targets (Indices)
#     # Convert GT seconds to indices
#     gt_s_idx = (gt_start_sec / sec_per_step).round().long()
#     gt_e_idx = (gt_end_sec / sec_per_step).round().long()
    
#     # Create grid: [0, 1, 2, ... T-1]
#     # Shape (1, T) -> (B, T)
#     grid = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    
#     # --- Classification Targets ---
#     # 1 if inside action, 0 otherwise
#     # Shape: (B, T)
#     target_cls = (grid >= gt_s_idx.unsqueeze(1)) & (grid <= gt_e_idx.unsqueeze(1))
#     target_cls = target_cls.float()
    
#     # Mask out padding from targets
#     target_cls = target_cls * video_mask.float()
    
#     # --- Regression Targets ---
#     # Left Offset = t - start
#     # Right Offset = end - t
#     target_off_l = grid - gt_s_idx.unsqueeze(1)
#     target_off_r = gt_e_idx.unsqueeze(1) - grid
#     target_reg = torch.stack([target_off_l, target_off_r], dim=-1).float() # (B, T, 2)
    
#     # --- Loss Calculation ---
    
#     # 1. Classification (Focal Loss)
#     # Use reduction='none' to keep (B, T) or (N,)
#     L_cls_map = sigmoid_focal_loss(cls_logits, target_cls, alpha=0.25, gamma=2.0, reduction='none')
    
#     # Mask invalid frames (padding)
#     L_cls_map = L_cls_map * video_mask.float()
    
#     # 2. Regression (GIoU Loss)
#     # Initialize with ZEROS. Background frames get 0 regression loss.
#     L_reg_map = torch.zeros_like(cls_logits) # (B, T)
    
#     # Identify foreground indices
#     #pos_mask = (target_cls > 0.5) & video_mask
    

    
#     if pos_mask.sum() > 0:
#         # Extract ONLY positive samples (where offsets are guaranteed positive)
#         pred_reg_pos = reg_offsets[pos_mask]   # Shape: (N_pos, 2)
#         target_reg_pos = target_reg[pos_mask]  # Shape: (N_pos, 2)
        
#         # Compute loss on subset
#         loss_pos = ctr_giou_loss(pred_reg_pos, target_reg_pos, reduction='none') # Shape: (N_pos,)
        
#         # Scatter/Assign back to the full map
#         # We flatten the map to index it easily with the boolean mask
#         L_reg_map.view(-1)[pos_mask.view(-1)] = loss_pos
    
#     # Now L_reg_map is (B, T) and contains valid losses for FG and 0 for BG
    
#     if return_per_sample:
#         # Sum over time to get (B,)
#         L_cls_per_sample = L_cls_map.sum(dim=1)
#         L_reg_per_sample = L_reg_map.sum(dim=1)
        
#         per_sample_loss = (lambda_cls * L_cls_per_sample) + (lambda_reg * L_reg_per_sample)
#         return per_sample_loss, {}
#     else:
#         # Standard reduction
#         num_pos = target_cls.sum().clamp(min=1.0)
#         total_loss = ((lambda_cls * L_cls_map.sum()) + (lambda_reg * L_reg_map.sum())) / num_pos
#         return total_loss, {"loss": total_loss}

# def compute_dense_tr_loss(
#     cls_logits,         # (B, T)
#     reg_offsets,        # (B, T, 2)
#     gt_start_sec,       # (B,)
#     gt_end_sec,         # (B,)
#     video_mask,         # (B, T)
#     sec_per_step=1.0,
#     lambda_cls=1.0,
#     lambda_reg=1.0
# ):
#     B, T = cls_logits.shape
#     device = cls_logits.device
    
#     # 1. Generate Targets (Indices)
#     # Convert GT seconds to indices
#     gt_s_idx = (gt_start_sec / sec_per_step).round().long()
#     gt_e_idx = (gt_end_sec / sec_per_step).round().long()
    
#     # Create grid: [0, 1, 2, ... T-1]
#     # Shape (1, T) -> (B, T)
#     grid = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    
#     # --- Classification Targets ---
#     # 1 if inside action, 0 otherwise
#     # Shape: (B, T)
#     target_cls = (grid >= gt_s_idx.unsqueeze(1)) & (grid <= gt_e_idx.unsqueeze(1))
#     target_cls = target_cls.float()
    
#     # Mask out padding from targets
#     target_cls = target_cls * video_mask.float()
    
#     # --- Regression Targets ---
#     # Left Offset = t - start
#     # Right Offset = end - t
#     target_off_l = grid - gt_s_idx.unsqueeze(1)
#     target_off_r = gt_e_idx.unsqueeze(1) - grid
#     target_reg = torch.stack([target_off_l, target_off_r], dim=-1).float() # (B, T, 2)
    
#     # --- A. Focal Loss (Classification) ---
#     # Apply to ALL valid frames (Foreground & Background)
#     # Note: Flatten batch and time for loss calculation
#     # Only mask out the padding
#     valid_mask = video_mask.view(-1)
    
#     # Flatten
#     cls_flat = cls_logits.view(-1)[valid_mask]
#     target_cls_flat = target_cls.view(-1)[valid_mask]
    
#     L_cls = sigmoid_focal_loss(
#         cls_flat, 
#         target_cls_flat, 
#         alpha=0.25, 
#         gamma=2.0, 
#         reduction='sum'
#     )
#     # Normalize by number of positive samples (standard practice for Focal Loss)
#     num_pos = target_cls.sum().clamp(min=1.0)
#     L_cls = L_cls / num_pos
    
#     # --- B. GIoU Loss (Regression) ---
#     # Only apply to POSITIVE frames (Foreground)
#     # We cannot regress boundaries from the background.
#     pos_mask = (target_cls > 0.5) & video_mask
    
#     if pos_mask.sum() > 0:
#         pred_reg_pos = reg_offsets[pos_mask]     # (N_pos, 2)
#         target_reg_pos = target_reg[pos_mask]    # (N_pos, 2)
        
#         # Use ctr_giou_loss or ctr_diou_loss (provided in your snippet)
#         L_reg = ctr_giou_loss(
#             pred_reg_pos, 
#             target_reg_pos, 
#             reduction='sum'
#         )
#         L_reg = L_reg / num_pos
#     else:
#         L_reg = torch.tensor(0.0, device=device)

#     total_loss = (lambda_cls * L_cls) + (lambda_reg * L_reg)
    
#     return total_loss, {
#         "loss": total_loss, 
#         "L_cls": L_cls, 
#         "L_reg": L_reg,
#         "num_pos": num_pos
#     }
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

class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()

