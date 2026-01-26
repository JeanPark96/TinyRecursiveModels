from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
import argparse
from utils.log import Logger
import random
import numpy as np
import json
import datetime
import sys
import importlib
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

import tqdm
# import wandb
# import coolname
# import hydra
# import pydantic
from omegaconf import DictConfig
# from adam_atan2 import AdamATan2

from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper

# new imports
from dataset.build_charades import *
# from utils.debug import plot_trajectories, select_debug_batch, plot_debug_batch
from models.losses import VideoTRMACTDenseLossHead
import models.recursive_reasoning.trm_phase1_org as trm
from models.recursive_reasoning.trm_phase1_org import (
    Video_TRM_ACT,
    TRMLocalizerConfig
)


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    optimizer_lr_schedule: bool
    optimizer_lr_min_ratio: float
    optimizer_lr_warmup_steps: int
    carry: Any

    step: int
    total_steps: int

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))

def compute_lr(base_lr: float, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(train_state.optimizer_lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=train_state.optimizer_lr_min_ratio
    )

def soft_nms_1d_pytorch(segs, scores, iou_thresh=0.5, sigma=0.5, min_score=0.001, method=2):
    """
    Pure PyTorch implementation of 1D Soft-NMS.
    segs: (N, 2) [start, end]
    scores: (N,)
    method: 1 (linear), 2 (gaussian)
    """
    # 1. Sort by score descending
    indices = scores.argsort(descending=True)
    segs = segs[indices]
    scores = scores[indices]
    
    final_segs = []
    final_scores = []
    
    while len(scores) > 0 and scores[0] > min_score:
        # Pick best
        best_seg = segs[0]
        best_score = scores[0]
        
        final_segs.append(best_seg)
        final_scores.append(best_score)
        
        # If only one left, break
        if len(scores) == 1:
            break
            
        # Remove best from list
        segs = segs[1:]
        scores = scores[1:]
        
        # Compute IoU with best_seg
        # Intersection
        tt1 = torch.maximum(best_seg[0], segs[:, 0])
        tt2 = torch.minimum(best_seg[1], segs[:, 1])
        intersection = (tt2 - tt1).clamp(min=0)
        
        # Union
        len_best = best_seg[1] - best_seg[0]
        len_others = segs[:, 1] - segs[:, 0]
        union = len_best + len_others - intersection
        
        iou = intersection / union.clamp(min=1e-6)
        
        # Soft-NMS Decay
        if method == 1: # Linear
            weight = torch.ones_like(iou)
            weight[iou > iou_thresh] -= iou[iou > iou_thresh]
        else: # Gaussian (Standard for SnAG/DecafNet)
            weight = torch.exp(-(iou * iou) / sigma)
            
        scores = scores * weight
        
        # Re-sort to bring next best to top
        new_idx = scores.argsort(descending=True)
        segs = segs[new_idx]
        scores = scores[new_idx]

    return torch.stack(final_segs), torch.stack(final_scores)
@torch.no_grad()
def decode_dense_candidates(cls_logits, reg_offsets, mask, duration, max_vid_len=256):
    """
    Generates ALL candidate segments.
    Returns:
        proposals: (N, 2) in Seconds
        scores: (N,)
    """
    # 1. Get Scores
    scores = torch.sigmoid(cls_logits) * mask.float() # (B, T)
    
    B, T = scores.shape
    candidates_list = []
    
    for i in range(B):
        # 2. Filter low scores to speed up NMS (Standard trick)
        # SOTA usually keeps top 100 or scores > 0.01
        valid_indices = torch.nonzero(scores[i] > 0.01).squeeze(1)
        
        if len(valid_indices) == 0:
            # Fallback if nothing is confident: pick max
            valid_indices = torch.argmax(scores[i]).unsqueeze(0)

        # 3. Decode specific indices
        idx_t = valid_indices.float()
        
        # Get Offsets: (N_valid, 2)
        cur_offsets = reg_offsets[i][valid_indices] 
        off_l = cur_offsets[:, 0]
        off_r = cur_offsets[:, 1]
        
        # Grid Indices -> Normalized Time
        # Start = (t - off_l) / 256
        # End   = (t + off_r) / 256
        pred_s = (idx_t - off_l) / float(max_vid_len)
        pred_e = (idx_t + off_r) / float(max_vid_len)
        
        # Convert to Seconds
        dur = duration[i]
        pred_s = pred_s * dur
        pred_e = pred_e * dur
        
        # Clamp
        pred_s = pred_s.clamp(min=0, max=dur)
        pred_e = pred_e.clamp(min=0, max=dur)
        
        # Stack [Start, End]
        proposals = torch.stack([pred_s, pred_e], dim=1)
        prop_scores = scores[i][valid_indices]
        
        candidates_list.append((proposals, prop_scores))
        
    return candidates_list

        
@torch.no_grad()
def compute_retrieval_metrics(
    proposals_list, # List of (segs, scores) tuples from decode function
    gt_start,       # (B,) Tensor
    gt_end,          # (B,) Tensor
    iou_thresh=0.1, # Standard for temporal grounding
    sigma=0.9, 
    method=2
):
    """
    Computes R@1 and R@5 at IoU=0.3, 0.5, 0.7 and returns batch statistics.
    """
    # Metrics accumulator
    batch_metrics = {
        "R1@IoU=0.3": [], "R1@IoU=0.5": [], "R1@IoU=0.7": [],
        "R5@IoU=0.3": [], "R5@IoU=0.5": [], "R5@IoU=0.7": [],
        "mIoU": []
    }
    
    # Batch Stats accumulator
    stat_pred_s = []
    stat_pred_e = []
    stat_gt_s = []
    stat_gt_e = []
    stat_ious = []
    
    for i, (segs, scores) in enumerate(proposals_list):
        # 1. Apply Soft-NMS (Standard SOTA processing)
        # Uses the python implementation we discussed
        nms_segs, nms_scores = soft_nms_1d_pytorch(
            segs, scores, 
            iou_thresh=iou_thresh, 
            sigma=sigma, 
            method=method
        )
        
        # 2. Extract Top-K
        # If NMS removed everything (rare), fallback to [0,0]
        if len(nms_segs) == 0:
            print("Warning: NMS removed everything. Fallback to [0,0]")
            top_1_seg = torch.tensor([0.0, 0.0], device=gt_start.device)
            top_5_segs = top_1_seg.unsqueeze(0)
        else:
            top_1_seg = nms_segs[0]       # Best prediction
            top_5_segs = nms_segs[:5]     # Top-5 candidates
        
        # 3. Get Ground Truth for this sample
        g_s = gt_start[i]
        g_e = gt_end[i]
        
        # 4. Compute IoU for Top-5
        # Intersection
        inter_min = torch.maximum(top_5_segs[:, 0], g_s)
        inter_max = torch.minimum(top_5_segs[:, 1], g_e)
        inter_len = (inter_max - inter_min).clamp(min=0)
        
        # Union
        pred_len = top_5_segs[:, 1] - top_5_segs[:, 0]
        gt_len = g_e - g_s
        union_len = pred_len + gt_len - inter_len
        
        ious = inter_len / union_len.clamp(min=1e-6)
        
        # 5. Compute Metrics
        top1_iou = ious[0].item()
        batch_metrics["mIoU"].append(top1_iou) 
        
        for thresh in [0.3, 0.5, 0.7]:
            # R@1
            r1 = 1.0 if top1_iou >= thresh else 0.0
            batch_metrics[f"R1@IoU={thresh}"].append(r1)
            # R@5
            r5 = 1.0 if (ious >= thresh).any() else 0.0
            batch_metrics[f"R5@IoU={thresh}"].append(r5)
            
        # 6. Store Stats (Top-1 prediction only)
        stat_pred_s.append(top_1_seg[0].cpu())
        stat_pred_e.append(top_1_seg[1].cpu())
        stat_gt_s.append(g_s.cpu())
        stat_gt_e.append(g_e.cpu())
        stat_ious.append(ious[0].cpu()) # Store Top-1 IoU

    # Average metrics over batch
    final_metrics = {k: torch.tensor(v).mean().item() for k, v in batch_metrics.items()}
    
    # Pack Stats
    batch_stats = {
        "pred_s": torch.stack(stat_pred_s),
        "pred_e": torch.stack(stat_pred_e),
        "gt_s": torch.stack(stat_gt_s),
        "gt_e": torch.stack(stat_gt_e),
        "ious": torch.stack(stat_ious)
    }
    
    return final_metrics, batch_stats

def temporal_iou_1d(pred_s, pred_e, gt_s, gt_e):
    """
    Computes IoU between two time spans.
    Input shapes: (B,)
    """
    # Intersection
    inter_min = torch.maximum(pred_s, gt_s)
    inter_max = torch.minimum(pred_e, gt_e)
    inter_len = torch.clamp(inter_max - inter_min, min=0)
    
    # Union
    union_min = torch.minimum(pred_s, gt_s)
    union_max = torch.maximum(pred_e, gt_e)
    union_len = torch.clamp(union_max - union_min, min=1e-6)
    
    return inter_len / union_len


def decode_dense_predictions(cls_logits, reg_offsets, mask, sec_per_step=1.0):
    """
    Decodes dense predictions into (start, end) spans.
    """
    # 1. Sigmoid & Threshold
    scores = torch.sigmoid(cls_logits) * mask.float() # (B, T)
    
    # 2. Pick best frame (Simple Argmax)
    # For more advanced usage, you can use NMS, but Argmax works for single-instance
    best_score, best_idx = scores.max(dim=1) # (B,)
    
    # 3. Get offsets at that frame
    # reg_offsets: (B, T, 2)
    # Gather: select the offset corresponding to best_idx
    batch_indices = torch.arange(scores.shape[0], device=scores.device)
    best_offsets = reg_offsets[batch_indices, best_idx] # (B, 2)
    
    off_l = best_offsets[:, 0]
    off_r = best_offsets[:, 1]
    
    # 4. Calculate Start/End
    # Start = t - off_l
    # End   = t + off_r
    pred_s_idx = best_idx.float() - off_l
    pred_e_idx = best_idx.float() + off_r
    
    return pred_s_idx * sec_per_step, pred_e_idx * sec_per_step

@torch.no_grad()
def compute_localization_metric(
    logits,          # (B, T)
    mask,            # (B, T)
    gt_start_sec,    # (B,)
    gt_end_sec,      # (B,)
    sec_per_step=1.0, 
    max_dur=None,
    offsets=None, # optional, valid only if dense loc prediction
    option="dense" 
):
    """
    Computes IoU-based metrics: R@0.3, R@0.5, R@0.7 and mIoU.
    Note: True mAP requires ranking over the entire dataset. 
    This function returns per-batch 'Recall' (Acc) which approximates mAP during training.
    """
    
    if option == "dense_head":
        pred_s, pred_e = decode_dense_predictions(logits, offsets, mask, sec_per_step)
        scores = []
    else:
        raise ValueError(f"Unknown option: {option}")

        
    # 2. Compute IoU for the batch
    ious = temporal_iou_1d(pred_s, pred_e, gt_start_sec, gt_end_sec)
    
    # 3. Compute Metrics
    metrics = {}
    metrics["mIoU"] = ious.mean().item()
    
    # Recall at IoU thresholds (Success Rate)
    for thresh in [0.3, 0.5, 0.7]:
        # (B,) bool -> float -> mean
        r_at_k = (ious >= thresh).float().mean().item()
        metrics[f"R1@IoU={thresh}"] = r_at_k
        
    # 'mAP' Proxy: Average of Recalls [0.5 : 0.05 : 0.95] is a common video eval metric
    # But usually people want Average Precision. Since we can't rank the whole dataset here,
    # we return the Batch Average IoU and R@0.5 as the primary indicators.
    
    # If you need to accumulate for global mAP, return lists:
    batch_stats = {
        "pred_s": pred_s.cpu(),
        "pred_e": pred_e.cpu(),
        #"scores": scores.cpu(), # Confidence score for ranking
        "gt_s": gt_start_sec.cpu(),
        "gt_e": gt_end_sec.cpu(),
        "ious": ious.cpu()
    }
    
    return metrics, batch_stats

def train_batch(train_state: TrainState, batch: Any):
    train_state.step += 1
    # if train_state.step > train_state.total_steps:  # At most train_total_steps
    #     return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}
    batch_size = batch[list(batch.keys())[0]].shape[0]

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, outputs, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=["logits", "extra_logits"])

    ((1 / batch_size) * loss).backward()
    
    torch.nn.utils.clip_grad_norm_(train_state.model.parameters(), max_norm=1.0)

    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        # # lr_this_step = compute_lr(base_lr, config, train_state)
        # lr_this_step = base_lr # not on a schedule right now
        if train_state.optimizer_lr_schedule:
            lr_this_step = compute_lr(base_lr, train_state)
        else:
            lr_this_step = base_lr


        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])

        metric_values = metric_values.cpu().numpy()
        reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
        
        # Postprocess
        count = max(reduced_metrics["count"], 1)  # Avoid NaNs
        reduced_metrics = {f"train/{k}": v / (batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

        reduced_metrics["train/lr"] = lr_this_step
        return reduced_metrics, outputs

def train(args, tr_dataset, val_dataset, test_dataset, tr_dataloader, val_dataloader, test_dataloader):
    RUN_NAME = args.run_name
    LOG_DIR = "logs"
    CKPT_DIR = "checkpoints"
    TBOARD_DIR = "tboard"
    run_ckpt_dir = os.path.join(CKPT_DIR, RUN_NAME)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TBOARD_DIR, exist_ok=True)

    # Device / GPU selection
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(args.gpu_id)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Infer key dims from a real batch (prevents config mismatch) ---
    sample = next(iter(tr_dataloader))

    # --- Check for resume ---
    last_ckpt_path = os.path.join(run_ckpt_dir, "last.pth")
    ckpt = None
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume and os.path.exists(last_ckpt_path):
        print(f"Resuming from checkpoint: {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location="cpu")
        config_dict = ckpt["config"]
        start_epoch = ckpt.get("epoch", 0) + 1  # epoch stored as 0-based
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
    else:
        # --- Fresh config (for TRM_ACT_NuScenes) ---
        config_dict = {
            "batch_size": args.config_batch_size,  # logical batch size; dataloader can differ
            # "global_len": 1,       # keep global latent token
            "hidden_size": args.hidden_size,
            "H_cycles": args.H_cycles,
            "L_cycles": args.L_cycles,
            "L_layers": 2,
            #"pos_encodings": "none",  # can switch to "rope" later
            "num_y_tokens" : 32, 
            "num_z_tokens" : 32,
            "max_frames": 256, 
            "halt_max_steps": args.halt_max_steps,
            "halt_exploration_prob": 0.0,
            "no_ACT_continue": True,
            "forward_dtype" : "float32",
            "loc_head" : args.loss_option,
            "video_in_dim": 512,
            "query_in_dim": 512,
            "max_text_len": 77
        }

        with open(os.path.join("./config", f"{RUN_NAME}.json"), "w") as f:
            json.dump(config_dict, f, indent=4)

        if args.resume:
            print(f"No checkpoint found at {last_ckpt_path}; starting from scratch.")

    # Initialize Logger
    logger = Logger(LOG_DIR, RUN_NAME, config_dict)
    logger.log("Loading Dataset...")
    logger.log(f"Dataset Loaded. Train samples: {len(tr_dataset)}, Val samples: {len(val_dataset)}")

    if ckpt is not None:
        logger.log(
            f"Resuming from checkpoint at epoch {start_epoch}, "
            f"global_step {global_step}, best_val_loss={best_val_loss:.4f}"
        )
    
    # Set up tensorboard
    datetimestr = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    tbd_writer = SummaryWriter(os.path.join(TBOARD_DIR, f'{RUN_NAME}/{datetimestr}'))

    # --- Model & optimizer ---
    logger.log("Initializing Model...")
    model = Video_TRM_ACT(config_dict).to(device)
    print_model_params(model, logger)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    # Train state
    train_state = TrainState(
        step=0,
        total_steps=0, # unused for now

        model=VideoTRMACTDenseLossHead(model=model),
        optimizers=[optimizer],
        optimizer_lrs=[args.lr],
        optimizer_lr_schedule=args.lr_schedule,
        optimizer_lr_min_ratio=1.0,
        optimizer_lr_warmup_steps=2000,
        carry=None
    )


    
    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        logger.log(f"\n=== Starting Epoch {epoch+1}/{args.epochs} ===")

        ############ Train Iter
        train_state.model.train()

        # step-wise stats
        running_loss = running_thr3 = running_thr5 = running_thr7 = r5_running_thr3 = r5_running_thr5 = r5_running_thr7 = 0.0
        running_count = 0

        # epoch-wise stats
        thr7_sum = thr5_sum = thr3_sum = r5_thr7_sum = r5_thr5_sum = r5_thr3_sum =tr_loss_sum = 0.0
        tr_n = 0

        for batch_idx, batch in enumerate(tr_dataloader):
            video_emb = batch["video_emb"].to(device)         # [B, T, D]
            query_tokens = batch["query_tokens"].to(device)   # [B, T_len, D]
            video_mask = batch["video_mask"].to(device)       # [B, T]
            query_mask = batch["query_mask"].to(device)       # [B, T_len]
            
            gt_start_sec, gt_end_sec = batch["start_sec"].to(device), batch["end_sec"].to(device)

            model_input = {
                "video_emb": video_emb,
                "text_emb": query_tokens,
                "video_mask": video_mask,
                "query_mask": query_mask,
                "i0": batch["i0"].to(device),
                "i1": batch["i1"].to(device)
            }

            metrics, outputs = train_batch(train_state, model_input)

            # calc extra metrics
            pred = outputs["logits"]
            if args.loss_option == "dense_head":
                loc_metric, stats = compute_localization_metric(pred, video_mask, gt_start_sec, gt_end_sec, offsets=outputs["extra_logits"] , option=args.loss_option)
            elif args.loss_option == "soft_nms":
                candidates = decode_dense_candidates(pred, outputs["extra_logits"], video_mask, batch["duration"].to(device), max_vid_len=256)
                loc_metric, stats = compute_retrieval_metrics(candidates, gt_start_sec, gt_end_sec)

            # log metrics
            running_thr3 += loc_metric[f"R1@IoU=0.3"]
            running_thr5 += loc_metric[f"R1@IoU=0.5"]
            running_thr7 += loc_metric[f"R1@IoU=0.7"]
            if "R5@IoU=0.3" in loc_metric:
                r5_running_thr3 += loc_metric[f"R5@IoU=0.3"]
                r5_running_thr5 += loc_metric[f"R5@IoU=0.5"]
                r5_running_thr7 += loc_metric[f"R5@IoU=0.7"]
            running_loss += metrics['train/loss']
            running_count += 1
            if train_state.step % 50 == 0:
                logger.log(
                    f"Epoch [{epoch+1}] Step [{train_state.step}] "
                    f"Loss: {running_loss/running_count:.4f} | "
                    f"IoU=0.7: {running_thr7/running_count:.4f} | "
                    f"IoU=0.5: {running_thr5/running_count:.4f} | "
                    f"IoU=0.3: {running_thr3/running_count:.4f} | "
                    f"R5@IoU=0.7: {r5_running_thr7/running_count:.4f} | "
                    f"R5@IoU=0.5: {r5_running_thr5/running_count:.4f} | "
                    f"R5@IoU=0.3: {r5_running_thr3/running_count:.4f} | "
                )
                # Iterate up to 4 samples (or batch size if smaller)
                n_print = min(4, pred.shape[0])
                
                p_s = stats["pred_s"]
                p_e = stats["pred_e"]
                g_s = stats["gt_s"]
                g_e = stats["gt_e"]
                ious = stats["ious"]

                for i in range(n_print):
                    print(f"  Sample {i}: Pred [{p_s[i]:.2f} - {p_e[i]:.2f}] "
                          f"| GT [{g_s[i]:.2f} - {g_e[i]:.2f}] "
                          f"| IoU: {ious[i]:.2f}")
                print("-" * 40)

                running_loss = running_thr7 = running_thr5 = running_thr3 = 0.0
                r5_running_thr7 = r5_running_thr5 = r5_running_thr3 = 0.0
                running_count = 0

            thr7_sum += loc_metric[f"R1@IoU=0.7"]
            thr5_sum += loc_metric[f"R1@IoU=0.5"]
            thr3_sum += loc_metric[f"R1@IoU=0.3"]
            if "R5@IoU=0.3" in loc_metric:
                r5_thr7_sum += loc_metric[f"R5@IoU=0.7"]
                r5_thr5_sum += loc_metric[f"R5@IoU=0.5"]
                r5_thr3_sum += loc_metric[f"R5@IoU=0.3"]
            tr_loss_sum += metrics['train/loss']
            tr_n += 1

        # tensorboard logging
        tbd_writer.add_scalar(f"Loss/train", tr_loss_sum / max(tr_n, 1), epoch+1)
        tbd_writer.add_scalar(f"IoU=0.7/train", thr7_sum / max(tr_n, 1), epoch+1)
        tbd_writer.add_scalar(f"IoU=0.5/train", thr5_sum / max(tr_n, 1), epoch+1)
        tbd_writer.add_scalar(f"IoU=0.3/train", thr3_sum / max(tr_n, 1), epoch+1)
        tbd_writer.add_scalar(f"R5@IoU=0.7/train", r5_thr7_sum / max(tr_n, 1), epoch+1)
        tbd_writer.add_scalar(f"R5@IoU=0.5/train", r5_thr5_sum / max(tr_n, 1), epoch+1)
        tbd_writer.add_scalar(f"R5@IoU=0.3/train", r5_thr3_sum / max(tr_n, 1), epoch+1)


        ############ Evaluation
        if epoch % 1 == 0:
            logger.log("Running Validation...")
            train_state.model.eval()
            val_th7_sum = val_th5_sum = val_th3_sum = val_loss_sum = 0.0
            r5_val_th7_sum = r5_val_th5_sum = r5_val_th3_sum = 0.0
            val_n = 0

            with torch.no_grad():
                for batch in val_dataloader:
                    video_emb = batch["video_emb"].to(device)              # [B, T, D]
                    query_tokens = batch["query_tokens"].to(device)              # [B, T_len, D]
                    video_mask = batch["video_mask"].to(device)       # [B, T]
                    query_mask = batch["query_mask"].to(device)       # [B, T_len]
                    
                    gt_start_sec, gt_end_sec = batch["start_sec"].to(device), batch["end_sec"].to(device)     

                    model_input = {
                        "video_emb": video_emb,
                        "text_emb": query_tokens,
                        "video_mask": video_mask,
                        "query_mask": query_mask,
                        "i0": batch["i0"].to(device),
                        "i1": batch["i1"].to(device)
                    }
                    batch_size = video_emb.shape[0]

                    with torch.device("cuda"):
                        carry = train_state.model.initial_carry(model_input)  # type: ignore

                    # Forward
                    inference_steps = 0
                    while True:
                        carry, loss, metrics, outputs, all_finish = train_state.model(
                            carry=carry, batch=model_input, return_keys=["logits", "extra_logits"]
                        )
                        inference_steps += 1

                        if all_finish:
                            break

                    # Reduce metrics
                    if len(metrics):
                        assert not any(v.requires_grad for v in metrics.values())

                        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
                        # Reduce and reconstruct
                        metric_values = torch.stack([metrics[k] for k in metric_keys])

                        metric_values = metric_values.cpu().numpy()
                        reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
                        
                        # Postprocess
                        count = max(reduced_metrics["count"], 1)  # Avoid NaNs
                        reduced_metrics = {f"val/{k}": v / (batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

                        reduced_metrics["val/lr"] = args.lr # original code has lr on a schedule
                            
                    # calc extra metrics
                    pred = outputs["logits"]
                    if args.loss_option == "dense_head":
                        loc_metric, stats = compute_localization_metric(pred, video_mask, gt_start_sec, gt_end_sec, offsets=outputs["extra_logits"] , option=args.loss_option)
                    elif args.loss_option == "soft_nms":
                        candidates = decode_dense_candidates(pred, outputs["extra_logits"], video_mask, batch["duration"].to(device), max_vid_len=256)
                        loc_metric, stats = compute_retrieval_metrics(candidates, gt_start_sec, gt_end_sec)

                    # log metrics
                    val_th3_sum += loc_metric[f"R1@IoU=0.3"]
                    val_th5_sum += loc_metric[f"R1@IoU=0.5"]
                    val_th7_sum += loc_metric[f"R1@IoU=0.7"]
                    if "R5@IoU=0.3" in loc_metric:
                        r5_val_th3_sum += loc_metric[f"R5@IoU=0.3"]
                        r5_val_th5_sum += loc_metric[f"R5@IoU=0.5"]
                        r5_val_th7_sum += loc_metric[f"R5@IoU=0.7"]
                    val_loss_sum += reduced_metrics['val/loss']
                    val_n += 1

            val_th3 = val_th3_sum / max(val_n, 1)
            val_th5 = val_th5_sum / max(val_n, 1)
            val_th7 = val_th7_sum / max(val_n, 1)
            r5_val_th3 = r5_val_th3_sum / max(val_n, 1)
            r5_val_th5 = r5_val_th5_sum / max(val_n, 1)
            r5_val_th7 = r5_val_th7_sum / max(val_n, 1)
            val_loss = val_loss_sum / max(val_n, 1)

            logger.log(
                f"Validation Results - Epoch {epoch+1}: "
                f"Loss: {val_loss:.4f} | "
                f"IoU=0.3: {val_th3:.4f} | "
                f"IoU=0.5: {val_th5:.4f} | "
                f"IoU=0.7: {val_th7:.4f} |"
                f"R5@IoU=0.3: {r5_val_th3:.4f} | "
                f"R5@IoU=0.5: {r5_val_th5:.4f} | "
                f"R5@IoU=0.7: {r5_val_th7:.4f}"
            )

            # tensorboard logging
            tbd_writer.add_scalar(f"Loss/val",val_loss,epoch+1)
            tbd_writer.add_scalar(f"IoU=0.3/val",val_th3,epoch+1)
            tbd_writer.add_scalar(f"IoU=0.5/val",val_th5,epoch+1)
            tbd_writer.add_scalar(f"IoU=0.7/val",val_th7,epoch+1)
            tbd_writer.add_scalar(f"R5@IoU=0.3/val",r5_val_th3,epoch+1)
            tbd_writer.add_scalar(f"R5@IoU=0.5/val",r5_val_th5,epoch+1)
            tbd_writer.add_scalar(f"R5@IoU=0.7/val",r5_val_th7,epoch+1)

                
            ############ Checkpointing
            epoch_ckpt_path = os.path.join(run_ckpt_dir, f"epoch_{epoch+1}.pth")
            ckpt_payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step, # note: this is meaningless in this ver (can't track global steps bc deep supervision is asynchronous)
                "epoch": epoch,  # 0-based
                "config": config_dict,
                "val_loss": val_loss,
                "val_iou=0.3": val_th3,
                "val_iou=0.5": val_th5,
                "val_iou=0.7": val_th7,
                "val_r5_iou=0.3": r5_val_th3,
                "val_r5_iou=0.5": r5_val_th5,
                "val_r5_iou=0.7": r5_val_th7,
                "best_val_loss": best_val_loss,
            }
            torch.save(ckpt_payload, epoch_ckpt_path)
            logger.log(f"Saved epoch checkpoint to {epoch_ckpt_path}")

            # --- Track & save best model ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = os.path.join(run_ckpt_dir, "best.pth")
                best_payload = ckpt_payload.copy()
                best_payload["best_val_loss"] = best_val_loss
                torch.save(best_payload, best_ckpt_path)
                logger.log(
                    f"New best model (val_loss={best_val_loss:.4f}); saved to {best_ckpt_path}"
                )

        logger.log("-" * 30)

    # finalize
    logger.log("Training Complete.")
    tbd_writer.flush()
    tbd_writer.close()

def print_model_params(model, logger):
    print(f"{'LAYER NAME':<50} {'SHAPE':<25} {'PARAMS'}")
    print("-" * 85)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Get shape (e.g., [2048, 512])
        shape = list(param.shape)
        # Get total count (e.g., 1,048,576)
        param_count = param.numel()
        
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
            
        print(f"{name:<50} {str(shape):<25} {param_count:,}")

    print("-" * 85)
    print(f"Total Trainable Params: {trainable_params:,}")
    print(f"Total Params: {total_params:,}")
    logger.log(f"{'LAYER NAME':<50} {'SHAPE':<25} {'PARAMS'}")
    logger.log("-" * 85)
    logger.log(f"Total Trainable Params: {trainable_params:,}")
    logger.log(f"Total Params: {total_params:,}")
    logger.log("-" * 85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="trm_video_exp1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--halt_max_steps", type=int, default=1)
    parser.add_argument("--config_batch_size", type=int, default=16)
    parser.add_argument("--L_cycles", type=int, default=6)
    parser.add_argument("--H_cycles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4501)
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint if available")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use (0-based)")
    parser.add_argument("--loss_option", type=str, default="dense_head")
    parser.add_argument("--lr_schedule", action="store_true", help="Use learning rate scheduler")

    args = parser.parse_args()

    # Optional CUDA debug envs (you can comment these out if you don't want sync execution)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    # Seeds for reproducibility (also used for picking debug batch)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # load dataset
    # tr_dataset, test_dataset, tr_dataloader, test_dataloader= load_dataset(args.config_batch_size)
    tr_ds, val_ds, test_ds, tr_loader, val_loader, test_loader = load_dataset(args.config_batch_size)
    
    # train
    train(args, tr_ds, val_ds, test_ds, tr_loader, val_loader, test_loader )
