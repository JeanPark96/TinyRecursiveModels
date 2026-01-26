from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

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


def compute_dense_tr_loss(
    cls_logits,         # (B, T)
    reg_offsets,        # (B, T, 2)
    gt_start_sec,       # (B,)
    gt_end_sec,         # (B,)
    video_mask,         # (B, T)
    sec_per_step=1.0,
    lambda_cls=1.0,
    lambda_reg=1.0
):
    B, T = cls_logits.shape
    device = cls_logits.device
    
    # 1. Generate Targets (Indices)
    # Convert GT seconds to indices
    gt_s_idx = (gt_start_sec / sec_per_step).round().long()
    gt_e_idx = (gt_end_sec / sec_per_step).round().long()
    
    # Create grid: [0, 1, 2, ... T-1]
    # Shape (1, T) -> (B, T)
    grid = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    
    # --- Classification Targets ---
    # 1 if inside action, 0 otherwise
    # Shape: (B, T)
    target_cls = (grid >= gt_s_idx.unsqueeze(1)) & (grid <= gt_e_idx.unsqueeze(1))
    target_cls = target_cls.float()
    
    # Mask out padding from targets
    target_cls = target_cls * video_mask.float()
    
    # --- Regression Targets ---
    # Left Offset = t - start
    # Right Offset = end - t
    target_off_l = grid - gt_s_idx.unsqueeze(1)
    target_off_r = gt_e_idx.unsqueeze(1) - grid
    target_reg = torch.stack([target_off_l, target_off_r], dim=-1).float() # (B, T, 2)
    
    # --- A. Focal Loss (Classification) ---
    # Apply to ALL valid frames (Foreground & Background)
    # Note: Flatten batch and time for loss calculation
    # Only mask out the padding
    valid_mask = video_mask.view(-1)
    
    # Flatten
    cls_flat = cls_logits.view(-1)[valid_mask]
    target_cls_flat = target_cls.view(-1)[valid_mask]
    
    L_cls = sigmoid_focal_loss(
        cls_flat, 
        target_cls_flat, 
        alpha=0.25, 
        gamma=2.0, 
        reduction='sum'
    )
    # Normalize by number of positive samples (standard practice for Focal Loss)
    num_pos = target_cls.sum().clamp(min=1.0)
    L_cls = L_cls / num_pos
    
    # --- B. GIoU Loss (Regression) ---
    # Only apply to POSITIVE frames (Foreground)
    # We cannot regress boundaries from the background.
    pos_mask = (target_cls > 0.5) & video_mask
    
    if pos_mask.sum() > 0:
        pred_reg_pos = reg_offsets[pos_mask]     # (N_pos, 2)
        target_reg_pos = target_reg[pos_mask]    # (N_pos, 2)
        
        # Use ctr_giou_loss or ctr_diou_loss (provided in your snippet)
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




class VideoTRMACTDenseLossHead(nn.Module):
    def __init__(self, model: nn.Module, halt_eps=0.01):
        super().__init__()
        self.model = model
        self.loss_fn = compute_dense_tr_loss
        self.halt_eps = halt_eps
        
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
        target_start_sec = new_carry.current_data["i0"]
        target_end_sec = new_carry.current_data["i1"]
        target_mask = new_carry.current_data["video_mask"]

        with torch.no_grad():
            # Preds (holdover from previous version of code, where pred is softmax of logits)
            # outputs["preds"] = outputs["pred"]
            # print(outputs.keys())
            # print(outputs)
            # Gain from previous step
            prev_loss = new_carry.prev_loss
            current_loss, current_log = compute_dense_tr_loss(
                outputs["logits"], outputs["extra_logits"], target_start_sec, target_end_sec, target_mask
            )  # shape (B,)

            delta = prev_loss - current_loss
            should_halt = (delta / prev_loss.clamp_min(1e-6)) <= self.halt_eps
            should_continue = (delta / prev_loss.clamp_min(1e-6)) > self.halt_eps
           
            # Update previous loss
            new_carry.prev_loss = current_loss.detach()

            # Metrics (halted)
            valid = new_carry.halted 
            metrics = {
                "count": new_carry.halted.sum(),
                "halted_loss": torch.where(valid, current_loss, 0).sum(),
                "halt_rate": torch.where(valid, should_halt, 0).sum(),
                "steps": torch.where(valid, new_carry.steps, 0).sum()
            }

        # Losses
        lm, curr_log = compute_dense_tr_loss(
           outputs["logits"], outputs["extra_logits"], target_start_sec, target_end_sec, target_mask
        )  # shape (B,)
        total_loss = lm.sum()

        # Q-halt loss
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            should_halt.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        metrics.update({
            "loss": total_loss.detach(),
            "q_halt_loss": q_halt_loss.detach()
        })

        # Q-continue loss
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                should_continue.to(outputs["q_continue_logits"].dtype),
                reduction="sum",
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        detached_outputs = {
            k: outputs[k].detach()
            for k in return_keys if k in outputs
        }

        return (
            new_carry,
            total_loss + 0.5 * (q_halt_loss + q_continue_loss),
            metrics,
            detached_outputs,
            new_carry.halted.all(),
        )
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
