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

def traj_loss_smooth_l1_from_batch(pred, targets, targets_mask, out_slice=2):
    """
    pred:         [B, A, H, out_dim]   (from model)
    targets:      [B, H, A, 7]
    targets_mask: [B, H, A]            (1 valid, 0 pad)
    """
    # align targets -> [B, A, H, out_slice]
    tgt = targets[..., :out_slice].permute(0, 2, 1, 3).contiguous()
    m = targets_mask.permute(0, 2, 1).to(pred.dtype).contiguous()  # [B, A, H]

    pred_xy = pred[..., :out_slice]
    loss = torch.nn.functional.smooth_l1_loss(pred_xy, tgt, reduction="none").sum(-1)  # [B,A,H]
    return (loss * m).sum(dim=(1,2)) / (m.sum(dim=(1,2)) + 1e-6)

class ACTLossHeadNuScenes(nn.Module):
    def __init__(self, model: nn.Module, halt_eps=0.01):
        super().__init__()
        self.model = model
        self.loss_fn = traj_loss_smooth_l1_from_batch
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
        targets = new_carry.current_data["targets"]
        targets_mask = new_carry.current_data["targets_mask"]
        num_agents = targets_mask.sum(dim=(1,2))

        with torch.no_grad():
            # Preds (holdover from previous version of code, where pred is softmax of logits)
            # outputs["preds"] = outputs["pred"]

            # Gain from previous step
            prev_loss = new_carry.prev_loss
            current_loss = traj_loss_smooth_l1_from_batch(
                outputs["pred"], targets, targets_mask
            )  # shape (B,)

            current_loss = torch.where( # there may be no agents (halt immediately, nothing to predict)
                num_agents > 0,
                current_loss,
                torch.zeros_like(current_loss),
            )

            delta = prev_loss - current_loss
            should_halt = (delta / prev_loss.clamp_min(1e-6)) <= self.halt_eps
            should_continue = (delta / prev_loss.clamp_min(1e-6)) > self.halt_eps
            
            should_halt = torch.where( # there may be no agents (halt immediately, nothing to predict)
                num_agents > 0,
                should_halt,
                torch.ones_like(should_halt, dtype=torch.bool),
            )
            should_continue = torch.where( # there may be no agents (halt immediately, nothing to predict)
                num_agents > 0,
                should_continue,
                torch.zeros_like(should_continue, dtype=torch.bool),
            )

            # Update previous loss
            new_carry.prev_loss = current_loss.detach()

            # Metrics (halted)
            valid = new_carry.halted 
            metrics = {
                "count": new_carry.halted.sum(),
                "halted_loss": torch.where(valid, current_loss, 0).sum(),
                "halt_rate": torch.where(valid, should_halt, 0).sum(),
                "steps": torch.where(valid, new_carry.steps, 0).sum(),
            }

        # Losses
        lm = traj_loss_smooth_l1_from_batch(
            outputs["pred"], targets, targets_mask
        )  # shape (B,)
        traj_loss = lm.sum()

        # Q-halt loss
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            should_halt.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        metrics.update({
            "loss": traj_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
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
            traj_loss + 0.5 * (q_halt_loss + q_continue_loss),
            metrics,
            detached_outputs,
            new_carry.halted.all(),
        )

class TrajectoryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # SmoothL1Loss is often better than MSE for trajectories as it's less sensitive to outliers
        self.reg_loss = nn.SmoothL1Loss(reduction='none') 

    def forward(self, pred_traj, target_traj, valid_mask):
        """
        Args:
            pred_traj: [Batch, Agents, Future, 2]
            target_traj: [Batch, Future, Agents, 7] (From NuScenes dataloader)
            valid_mask: [Batch, Future, Agents] (1 if agent exists, 0 if padding)
        """
        # 1. Align Target Shape: [B, T, N, 7] -> [B, N, T, 7]
        target_traj = target_traj.permute(0, 2, 1, 3)
        valid_mask = valid_mask.permute(0, 2, 1) # [B, N, T]

        # 2. Slice Target: We only want (x, y), usually the first 2 indices
        gt_xy = target_traj[..., :2] 

        # 3. Calculate Loss (Smooth L1)
        # loss dimensions: [B, N, T, 2]
        loss = self.reg_loss(pred_traj, gt_xy)
        
        # 4. Average over (x, y) dimensions
        loss = loss.mean(dim=-1) # [B, N, T]

        # 5. Apply Masking (Ignore non-existent agents)
        # valid_mask needs to match shape. 
        # Note: NuScenes mask might be for history. Ensure you have a FUTURE mask.
        # If your 'obs_mask' is only for history, you might assume if it existed in history,
        # it exists in future, OR use the target values themselves (if 0,0,0,0 means empty).
        
        # Assuming valid_mask is provided for the future steps:
        masked_loss = loss * valid_mask
        
        # 6. Normalize by number of valid elements to keep loss magnitude consistent
        num_valid = valid_mask.sum() + 1e-6 # Avoid div by zero
        return masked_loss.sum() / num_valid

def calculate_metrics(pred_traj, target_traj, valid_mask):
    """
    Calculates ADE (Average Displacement Error) and FDE (Final Displacement Error)
    """
    with torch.no_grad():
        # Align Target: [B, T, N, 7] -> [B, N, T, 2]
        gt_xy = target_traj.permute(0, 2, 1, 3)[..., :2]
        valid_mask = valid_mask.permute(0, 2, 1)

        # Displacement: Euclidean distance between pred and gt
        # shape: [B, N, T]
        displacement = torch.norm(pred_traj - gt_xy, dim=-1)

        # Apply mask
        masked_displacement = displacement * valid_mask
        
        # Sum of valid agents per sample (or total)
        num_valid = valid_mask.sum() + 1e-6

        # --- ADE: Average over all Time Steps ---
        ade = masked_displacement.sum() / num_valid

        # --- FDE: Displacement at the Last Time Step ---
        final_displacement = displacement[:, :, -1]      # [B, N]
        final_mask = valid_mask[:, :, -1]                # [B, N]
        
        fde = (final_displacement * final_mask).sum() / (final_mask.sum() + 1e-6)

        return ade.item(), fde.item()