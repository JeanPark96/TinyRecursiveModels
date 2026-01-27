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

# from utils.functions import load_model_class, get_model_source_path
# from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
# from models.ema import EMAHelper

# new imports
from dataset.build_charades_sta_snag import *
# from utils.debug import plot_trajectories, select_debug_batch, plot_debug_batch
from models.losses import VideoTRMACTDenseLossHead
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

def train(args, tr_dataset, val_dataset, test_dataset, tr_dataloader, val_dataloader, test_dataloader):
    RUN_NAME = args.run_name
    LOG_DIR = args.log_dir
    CKPT_DIR = args.checkpoint_dir
    TBOARD_DIR = args.tboard_dir
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
            "video_in_dim": 2048,
            "query_in_dim": 300,
            "max_text_len": 16
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
            # if batch_idx <= 3:
            #     print(video_emb[0, :, :])
            #     print(query_tokens[0, :, :])

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
            #batch_target_outputs = train_state.carry.current_data[""]

            # calc extra metrics
            pred = outputs["logits"]
            if args.loss_option == "dense_head":
                if args.feature_type == "i3d":
                    max_dur = batch["duration"].to(device)
                    #print(max_dur)
                loc_metric, stats = compute_localization_metric(pred, video_mask, gt_start_sec, gt_end_sec, offsets=outputs["extra_logits"] , max_dur=max_dur, option=args.loss_option)
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
                        if args.feature_type == "i3d":
                            max_dur = batch["duration"].to(device)
                        loc_metric, stats = compute_localization_metric(pred, video_mask, gt_start_sec, gt_end_sec, offsets=outputs["extra_logits"] , max_dur=max_dur, option=args.loss_option)
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
    parser.add_argument("--log_dir", type=str, default="./video_trm_log")
    parser.add_argument("--checkpoint_dir", type=str, default="./video_trm_checkpoints")
    parser.add_argument("--tboard_dir", type=str, default="./video_trm_tboard")
    parser.add_argument("--data_root", type=str, default="/home/hlpark/common-data/jean")
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
    tr_ds, val_ds, test_ds, tr_loader, val_loader, test_loader = load_dataset(args, args.config_batch_size)
    
    # train
    train(args, tr_ds, val_ds, test_ds, tr_loader, val_loader, test_loader )
