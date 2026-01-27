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
from dataset.build_charades_sta_snag import *
# from utils.debug import plot_trajectories, select_debug_batch, plot_debug_batch
from models.losses import VideoTRMACTDenseLossHead
import models.recursive_reasoning.trm_phase1_fixedlen_trunc_grad as trm
from models.recursive_reasoning.trm_phase1_fixedlen_trunc_grad import (
    Video_TRM_ACT,
    TRMLocalizerConfig
)
from utils.metric import compute_retrieval_metrics, compute_localization_metric, decode_dense_candidates


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


def eval(args, dataset, dataloader):
    RUN_NAME = f'{args.run_name}' 
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    # Device / GPU selection
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(args.gpu_id)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")


    # Load trained model
    ckpt = torch.load(args.model_pth, map_location="cpu", weights_only=False)
    config_dict = ckpt["config"]
    start_epoch = ckpt.get("epoch", 0) + 1  # epoch stored as 0-based
    global_step = ckpt.get("global_step", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))

    # Initialize Logger
    logger = Logger(LOG_DIR, RUN_NAME, config_dict)
    logger.log("Loading Dataset...")
    logger.log(f"Dataset Loaded. Test samples: {len(dataset)}")

    # --- Model & optimizer ---
    logger.log("Initializing Model...")
    model = Video_TRM_ACT(config_dict).to(device)
    model.load_state_dict(ckpt["model"])

    # Train state
    train_state = TrainState(
        step=0,
        total_steps=0, # unused for now

        model=VideoTRMACTDenseLossHead(model=model),
        optimizers=[None],
        optimizer_lrs=[None],
        optimizer_lr_schedule=args.lr_schedule,
        optimizer_lr_min_ratio=1.0,
        optimizer_lr_warmup_steps=2000,
        carry=None
    )

    # Testing Loop
    logger.log("Running Test...")
    train_state.model.eval()
    th3_sum = th5_sum = th7_sum = loss_sum = 0.0
    r5_th3_sum = r5_th5_sum = r5_th7_sum = 0.0
    n = 0

    with torch.no_grad():
        for b, batch in enumerate(dataloader):
            if b % 10 == 0:
                print(f'Batch {b}')
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
                reduced_metrics = {f"test/{k}": v / (batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}
                    
            
            pred = outputs["logits"]
            if args.loss_option == "dense_head":
                if args.feature_type == "i3d":
                    max_dur = batch["duration"].to(device)
                loc_metric, stats = compute_localization_metric(pred, video_mask, gt_start_sec, gt_end_sec, offsets=outputs["extra_logits"] , max_dur=max_dur, option=args.loss_option)
            elif args.loss_option == "soft_nms":
                candidates = decode_dense_candidates(pred, outputs["extra_logits"], video_mask, batch["duration"].to(device), max_vid_len=256)
                loc_metric, stats = compute_retrieval_metrics(candidates, gt_start_sec, gt_end_sec)
            # loc_metric, stats = compute_localization_metric(pred, video_mask, gt_start_sec, gt_end_sec, offsets=outputs["extra_logits"] , option=args.loss_option)
            #print(loc_metric)
            # log metrics
            th3_sum += loc_metric[f"R1@IoU=0.3"]
            th5_sum += loc_metric[f"R1@IoU=0.5"]
            th7_sum += loc_metric[f"R1@IoU=0.7"]
            if "R5@IoU=0.3" in loc_metric:
                r5_th3_sum += loc_metric[f"R5@IoU=0.3"]
                r5_th5_sum += loc_metric[f"R5@IoU=0.5"]
                r5_th7_sum += loc_metric[f"R5@IoU=0.7"]

            loss_sum += reduced_metrics['test/loss']
            n += 1

    avg_th3 = th3_sum / max(n, 1)
    avg_th5 = th5_sum / max(n, 1)
    avg_th7 = th7_sum / max(n, 1)
    r5_avg_th3 = r5_th3_sum / max(n, 1)
    r5_avg_th5 = r5_th5_sum / max(n, 1)
    r5_avg_th7 = r5_th7_sum / max(n, 1)
    avg_val_loss = loss_sum / max(n, 1)

    logger.log(
        f"Test Results: "
        f"Loss: {avg_val_loss:.4f} | "
        f"R1@IoU=0.3: {avg_th3:.4f} | "
        f"R1@IoU=0.5: {avg_th5:.4f} | "
        f"R1@IoU=0.7: {avg_th7:.4f} |"
        f"R5@IoU=0.3: {r5_avg_th3:.4f} | "
        f"R5@IoU=0.5: {r5_avg_th5:.4f} | "
        f"R5@IoU=0.7: {r5_avg_th7:.4f} |"
    )

    
    logger.log("Testing Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pth", type=str)
    parser.add_argument("--run_name", type=str, default="trm_video_exp1")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--halt_max_steps", type=int, default=1)
    parser.add_argument("--config_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4501)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use (0-based)")
    parser.add_argument("--loss_option", type=str, default="dense_head")
    parser.add_argument("--lr_schedule", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--feature_type", type=str, default="i3d", help="choice = [clip, i3d]")
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
    eval(args, test_ds, test_loader )
