from typing import Any, Tuple, Dict, Sequence, Optional
import torch
import torch.nn.functional as F
from torch import nn

from models.losses import traj_loss_smooth_l1_from_batch
from utils.halt_helper import batch_correct_by_agent_count, CheckMode, Metric


class ACTLossHeadNuScenes(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        name: str,
        loss: str, 
        # --- halting-label config ---
        halt_threshold: float = 1.0,          # threshold in DENORMALIZED units
        halt_mode: CheckMode = "mean",
        halt_metric: Metric = "l2",
        n_correct_upper_bound: Optional[int] = None,
        # keys
        pred_key: str = "pred",
        targets_key: str = "targets",
        targets_mask_key: str = "targets_mask",
        # denorm stats
        denorm: bool = False,
        std_xy: Optional[torch.Tensor] = None,    # ideally [2]
        mean_xy: Optional[torch.Tensor] = None,   # ideally [2]
        # loss reduction
        traj_reduction: str = "sum",  # "sum" or "mean"
        verbose: bool = False,
    ):
        super().__init__()
        self.model = model
        self.name = name
        self.loss = loss
        # Your traj loss function (returns [B])
        self.loss_fn = traj_loss_smooth_l1_from_batch

        self.halt_threshold = halt_threshold
        self.halt_mode = halt_mode
        self.halt_metric = halt_metric
        self.n_correct_upper_bound = n_correct_upper_bound

        self.pred_key = pred_key
        self.targets_key = targets_key
        self.targets_mask_key = targets_mask_key

        self.denorm = denorm
        self.std_xy = std_xy
        self.mean_xy = mean_xy
        if self.denorm and (self.std_xy is None or self.mean_xy is None):
            raise ValueError("If denorm=True, std_xy and mean_xy must be provided to the loss head.")

        if traj_reduction not in ("sum", "mean"):
            raise ValueError("traj_reduction must be 'sum' or 'mean'")
        self.traj_reduction = traj_reduction

        self.verbose = verbose

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def decode(self, *args, **kwargs):
        return self.model.decode(*args, **kwargs)
    
    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:

        new_carry, outputs = self.model(**model_kwargs)

        # Required batch items
        targets = new_carry.current_data[self.targets_key]           # [B,H,A,7]
        targets_mask = new_carry.current_data[self.targets_mask_key] # [B,H,A]

        # Required model outputs
        if self.pred_key not in outputs:
            raise KeyError(f"Missing '{self.pred_key}' in outputs. Available: {list(outputs.keys())}")
        pred = outputs[self.pred_key]  # [B,A,H,2]

        if "q_halt_logits" not in outputs:
            raise KeyError("Missing 'q_halt_logits' in outputs.")
        q_halt_logits = outputs["q_halt_logits"]

        # Normalize q_halt_logits to [B]
        if q_halt_logits.ndim == 1:
            q_halt_logits_eval = q_halt_logits
        elif q_halt_logits.ndim == 2:
            if q_halt_logits.size(1) == 1:
                q_halt_logits_eval = q_halt_logits.squeeze(1)
            else:
                # e.g. [B, steps] -> take last step
                q_halt_logits_eval = q_halt_logits[:, -1]
        elif q_halt_logits.ndim == 3 and q_halt_logits.size(-1) == 1:
            # e.g. [B, steps, 1] -> last step
            q_halt_logits_eval = q_halt_logits[:, -1, 0]
        else:
            raise ValueError(f"Unsupported q_halt_logits shape: {tuple(q_halt_logits.shape)}")

        # -------------------------
        # Task loss (trajectory)
        # -------------------------
        traj_loss_per_sample = self.loss_fn(pred, targets, targets_mask)  # [B]

        if self.traj_reduction == "sum":
            traj_loss = traj_loss_per_sample.sum()
        else:
            traj_loss = traj_loss_per_sample.mean()

        # -------------------------
        # Halt label + metrics
        # -------------------------
        with torch.no_grad():
            sample_ok, num_correct, num_valid, _ = batch_correct_by_agent_count(
                pred=pred,
                targets=targets.clone(),   # avoid in-place mutation when denorm=True
                targets_mask=targets_mask,
                threshold=self.halt_threshold,
                mode=self.halt_mode,
                metric=self.halt_metric,
                n_correct_upper_bound=self.n_correct_upper_bound,
                return_counts=True,
                denorm=self.denorm,
                std_xy=self.std_xy,
                mean_xy=self.mean_xy,
                verbose=False,
            )
            seq_is_correct = sample_ok.to(torch.bool)  # [B]

            valid_metrics = new_carry.halted & (num_valid > 0)

            frac_correct_agents = num_correct.to(torch.float32) / num_valid.clamp_min(1).to(torch.float32)

            metrics: Dict[str, torch.Tensor] = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(valid_metrics, frac_correct_agents, torch.zeros_like(frac_correct_agents)).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((q_halt_logits_eval >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, torch.zeros_like(new_carry.steps)).sum(),
            }

        # -------------------------
        # Halt BCE loss
        # -------------------------
        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_halt_logits_eval,
            seq_is_correct.to(q_halt_logits_eval.dtype),
            reduction="sum",
        )

        # Optional q_continue (unchanged)
        q_continue_loss = torch.tensor(0.0, device=traj_loss.device, dtype=traj_loss.dtype)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum"
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Logging losses
        metrics.update({
            "loss": traj_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
            "loss_per_sample_mean": traj_loss_per_sample.mean().detach(),
        })

        # -------------------------
        # Verbose prints
        # -------------------------
        if self.verbose:
            B = pred.shape[0]
            print("\n[ACTNuscenesLossHead verbose]")
            print(f" pred: {tuple(pred.shape)}  targets: {tuple(targets.shape)}  mask: {tuple(targets_mask.shape)}")
            print(f" q_halt_logits: {tuple(q_halt_logits.shape)} -> eval: {tuple(q_halt_logits_eval.shape)}")
            print(f" denorm={self.denorm} threshold={self.halt_threshold} mode={self.halt_mode} metric={self.halt_metric}")
            if self.denorm:
                print(f" std_xy={self.std_xy}  mean_xy={self.mean_xy}")
            print(f" num_valid agents per sample: {num_valid.tolist()}")
            print(f" num_correct agents per sample: {num_correct.tolist()}")
            print(f" sample_ok (seq_is_correct): {sample_ok.tolist()}")
            print(f" q_halt_logits_eval (first min(8,B)): {q_halt_logits_eval[:min(8,B)].detach().cpu().tolist()}")
            print(f" pred_halt (logit>=0): {(q_halt_logits_eval>=0)[:min(8,B)].detach().cpu().tolist()}")
            print(f" traj_loss scalar ({self.traj_reduction}): {traj_loss.item():.6f}")
            print(f" q_halt_loss scalar (sum): {q_halt_loss.item():.6f}")
            if "target_q_continue" in outputs:
                print(f" q_continue_loss scalar (sum): {q_continue_loss.item():.6f}")

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        total_loss = traj_loss + 0.5 * (q_halt_loss + q_continue_loss)
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
