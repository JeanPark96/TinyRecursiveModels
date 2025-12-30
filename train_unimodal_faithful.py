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
from nuscenes_dataset import NuScenesDataset, custom_collate
import argparse
from utils.log import Logger
from utils.debug import plot_trajectories, select_debug_batch, plot_debug_batch
from models.losses import ACTLossHeadNuScenes
import random
import numpy as np
import json
import datetime
import sys
import importlib
import models.recursive_reasoning.trm_unimodal_v2 as trm_unimodal
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

importlib.reload(trm_unimodal)
# --- IMPORTS ---
# Ensure these imports match your file structure
# from my_dataset import NuScenesMiniDataset, custom_collate 
from models.recursive_reasoning.trm_unimodal_v2 import (
    TRM_ACT_NuScenes,
    TRM_ACT_NuScenes_Config
)

SAMPLE_FREQ = 2
max_obstacles = 1#30
n_history = 2*SAMPLE_FREQ # current time inclusive
n_horizon = 2*SAMPLE_FREQ

print(f"n_history: {n_history}, n_horizon: {n_horizon}")

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
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

# def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
#     return cosine_schedule_with_warmup_lr_lambda(
#         current_step=train_state.step,
#         base_lr=base_lr,
#         num_warmup_steps=round(config.lr_warmup_steps),
#         num_training_steps=train_state.total_steps,
#         min_ratio=config.lr_min_ratio
#     )

@torch.no_grad()
def compute_ade_fde(pred, targets, targets_mask, out_slice=2):
    """
    Returns scalar ADE/FDE (averaged over valid agents+timesteps).
    """
    tgt = targets[..., :out_slice].permute(0, 2, 1, 3).contiguous()         # [B,A,H,2]
    m = targets_mask.permute(0, 2, 1).to(pred.dtype).contiguous()           # [B,A,H]

    pred_xy = pred[..., :out_slice]
    dist = torch.linalg.norm(pred_xy - tgt, dim=-1)                         # [B,A,H]

    ade = (dist * m).sum() / (m.sum() + 1e-6)

    # FDE: last horizon step only
    dist_last = dist[:, :, -1]
    m_last = m[:, :, -1]
    fde = (dist_last * m_last).sum() / (m_last.sum() + 1e-6)

    return ade.item(), fde.item()

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
    train_state.carry, loss, metrics, outputs, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=["pred"])

    ((1 / batch_size) * loss).backward()
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        # lr_this_step = compute_lr(base_lr, config, train_state)
        lr_this_step = base_lr # not on a schedule right now

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

def train(args, tr_dataset, val_dataset, test_dataset, ood_dataset, tr_dataloader, val_dataloader, test_dataloader, ood_dataloader, stats, mean_xy, std_xy):
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

    mean_xy = mean_xy.to(device)
    std_xy = std_xy.to(device)

    # --- Infer key dims from a real batch (prevents config mismatch) ---
    sample = next(iter(tr_dataloader))
    # n_history = sample["obs_pose"].shape[1]
    # max_obstacles = sample["obs_pose"].shape[2]
    # n_horizon = sample["targets"].shape[1]

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
        print("Horizon", n_horizon)
        config_dict = {
            "batch_size": args.config_batch_size,  # logical batch size; dataloader can differ
            "n_history": n_history,
            "max_obstacles": max_obstacles,
            "n_horizon": n_horizon,

            "in_dim": 7,
            "out_dim": 2,          # predict x,y
            "out_slice": 2,        # supervise x,y
            "predict_delta": True,

            "global_len": 1,       # keep global latent token
            "seq_len": max_obstacles * n_history,

            "hidden_size": args.hidden_size,
            "expansion": 2.0,
            "num_heads": 4,
            "H_cycles": 3,
            "L_cycles": 6,
            "H_layers": 0,
            "L_layers": 2,
            "pos_encodings": "none",  # can switch to "rope" later

            "halt_max_steps": args.halt_max_steps,
            "halt_exploration_prob": 0.0,
            "no_ACT_continue": True,

            "forward_dtype": "float32",
            "mlp_t": False,
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
    model = TRM_ACT_NuScenes(config_dict).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    # --- Choose a debug batch for qualitative comparison across epochs ---
    debug_batch, debug_idx = select_debug_batch(val_dataloader, seed=args.seed)
    logger.log(f"Selected validation batch index {debug_idx} as debug batch for plotting.")

    # --- Choose a debug batch for ood for qualitative comparison across epochs ---
    if ood_dataloader is not None:
        ood_debug_batch, ood_debug_idx = select_debug_batch(ood_dataloader, seed=args.seed)
        logger.log(f"Selected ood batch index {debug_idx} as debug batch for plotting.")

    # Train state
    train_state = TrainState(
        step=0,
        total_steps=0, # unused for now

        model=ACTLossHeadNuScenes(model=model),
        optimizers=[optimizer],
        optimizer_lrs=[args.lr],
        carry=None
    )
    
    # --- Plot BEFORE training/resume (using current model state) ---
    plot_debug_batch(
        train_state,
        debug_batch,
        device,
        epoch=start_epoch,
        run_name=RUN_NAME,
        out_slice=config_dict["out_slice"],
    )

    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        logger.log(f"\n=== Starting Epoch {epoch+1}/{args.epochs} ===")

        ############ Train Iter
        train_state.model.train()

        # step-wise stats
        running_loss = running_ade = running_fde = running_ade_real = running_fde_real = 0.0
        running_count = 0

        # epoch-wise stats
        tr_ade_sum = tr_fde_sum = tr_ade_real_sum = tr_fde_real_sum = tr_loss_sum = 0.0
        tr_n = 0

        for batch_idx, batch in enumerate(tr_dataloader):
            obs_pose = batch["obs_pose"].to(device)              # [B, Hist, A, 7]
            obs_mask = batch["obs_mask"].to(device)              # [B, Hist, A]
            targets = batch["targets"].to(device)                # [B, Fut, A, 7]
            targets_mask = batch.get("targets_mask", None)       # [B, Fut, A]
            if targets_mask is None:
                targets_mask = (targets[..., :2].abs().sum(dim=-1) > 1e-3).to(obs_pose.dtype)
            else:
                targets_mask = targets_mask.to(device)
            
            model_input = {
                "obs_pose": obs_pose,
                "obs_mask": obs_mask,
                "targets": targets,
                "targets_mask": targets_mask,
            }

            metrics, outputs = train_batch(train_state, model_input)

            # calc extra metrics
            pred = outputs["pred"]
            ade, fde = compute_ade_fde(
                pred, targets, targets_mask, out_slice=config_dict["out_slice"]
            )

            pred_xy_denorm = pred[..., :2] * std_xy + mean_xy               # [B, A, H, 2]
            targets_xy_denorm = (targets[..., :2] * std_xy + mean_xy)       # [B, H, A, 2]

            ade_real, fde_real = compute_ade_fde(
                pred_xy_denorm, targets_xy_denorm, targets_mask, out_slice=config_dict["out_slice"]
            )

            # log metrics
            running_ade += ade
            running_fde += fde
            running_ade_real += ade_real
            running_fde_real += fde_real
            running_loss += metrics['train/loss']
            running_count += 1
            if train_state.step % 50 == 0:
                logger.log(
                    f"Epoch [{epoch+1}] Step [{train_state.step}] "
                    f"Loss: {running_loss/running_count:.4f} | "
                    f"ADE: {running_ade/running_count:.4f} | "
                    f"FDE: {running_fde/running_count:.4f} | "
                    f"ADE_real: {running_ade_real/running_count:.4f} | "
                    f"FDE_real: {running_fde_real/running_count:.4f}" 
                )
                running_loss = running_ade = running_fde = running_ade_real = running_fde_real = 0.0
                running_count = 0

            tr_ade_sum += ade
            tr_fde_sum += fde
            tr_ade_real_sum += ade_real
            tr_fde_real_sum += fde_real
            tr_loss_sum += metrics['train/loss']
            tr_n += 1

        # tensorboard logging
        tbd_writer.add_scalar(f"Loss/train", tr_loss_sum / max(tr_n, 1), epoch+1)
        tbd_writer.add_scalar(f"ADE/train", tr_ade_sum / max(tr_n, 1), epoch+1)
        tbd_writer.add_scalar(f"FDE/train", tr_fde_sum / max(tr_n, 1), epoch+1)
        tbd_writer.add_scalar(f"ADE_real/train", tr_ade_real_sum / max(tr_n, 1), epoch+1)
        tbd_writer.add_scalar(f"FDE_real/train", tr_fde_real_sum / max(tr_n, 1), epoch+1)


        ############ Evaluation
        if epoch % 1 == 0:
            logger.log("Running Validation...")
            train_state.model.eval()
            val_ade_sum = val_fde_sum = val_ade_real_sum = val_fde_real_sum = val_loss_sum = 0.0
            val_n = 0

            with torch.no_grad():
                for batch in val_dataloader:
                    obs_pose = batch["obs_pose"].to(device)
                    obs_mask = batch["obs_mask"].to(device)
                    targets = batch["targets"].to(device)
                    targets_mask = batch.get("targets_mask", None)
                    if targets_mask is None:
                        targets_mask = (targets[..., :2].abs().sum(dim=-1) > 1e-3).to(obs_pose.dtype)
                    else:
                        targets_mask = targets_mask.to(device)
                    batch_size = obs_pose.shape[0]

                    model_input = {
                        "obs_pose": obs_pose,
                        "obs_mask": obs_mask,
                        "targets": targets,
                        "targets_mask": targets_mask,
                    }

                    with torch.device("cuda"):
                        carry = train_state.model.initial_carry(model_input)  # type: ignore

                    # Forward
                    inference_steps = 0
                    while True:
                        carry, loss, metrics, outputs, all_finish = train_state.model(
                            carry=carry, batch=model_input, return_keys=["pred"]
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
                    pred = outputs["pred"]
                    ade, fde = compute_ade_fde(
                        pred, targets, targets_mask, out_slice=config_dict["out_slice"]
                    )

                    pred_xy_denorm = pred[..., :2] * std_xy + mean_xy               # [B, A, H, 2]
                    targets_xy_denorm = (targets[..., :2] * std_xy + mean_xy)       # [B, H, A, 2]

                    ade_real, fde_real = compute_ade_fde(
                        pred_xy_denorm, targets_xy_denorm, targets_mask, out_slice=config_dict["out_slice"]
                    )

                    # log metrics
                    val_ade_sum += ade
                    val_fde_sum += fde
                    val_ade_real_sum += ade_real
                    val_fde_real_sum += fde_real
                    val_loss_sum += reduced_metrics['val/loss']
                    val_n += 1

            val_ade = val_ade_sum / max(val_n, 1)
            val_fde = val_fde_sum / max(val_n, 1)
            val_ade_real = val_ade_real_sum / max(val_n, 1)
            val_fde_real = val_fde_real_sum / max(val_n, 1)
            val_loss = val_loss_sum / max(val_n, 1)

            logger.log(
                f"Validation Results - Epoch {epoch+1}: "
                f"Loss: {val_loss:.4f} | "
                f"ADE: {val_ade:.4f} | "
                f"FDE: {val_fde:.4f} | "
                f"Real ADE: {val_ade_real:.4f} | "
                f"Real FDE: {val_fde_real:.4f}"
            )

            # tensorboard logging
            tbd_writer.add_scalar(f"Loss/val",val_loss,epoch+1)
            tbd_writer.add_scalar(f"ADE/val",val_ade,epoch+1)
            tbd_writer.add_scalar(f"FDE/val",val_fde,epoch+1)
            tbd_writer.add_scalar(f"ADE_real/val",val_ade_real,epoch+1)
            tbd_writer.add_scalar(f"FDE_real/val",val_fde_real,epoch+1)

            # --- Plot debug batch AFTER this epoch ---
            plot_debug_batch(
                train_state,
                debug_batch,
                device,
                epoch=epoch+1,
                run_name=RUN_NAME,
                out_slice=config_dict["out_slice"],
            )
                
            ############ Checkpointing
            epoch_ckpt_path = os.path.join(run_ckpt_dir, f"epoch_{epoch+1}.pth")
            ckpt_payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step, # note: this is meaningless in this ver (can't track global steps bc deep supervision is asynchronous)
                "epoch": epoch,  # 0-based
                "config": config_dict,
                "val_loss": val_loss,
                "val_ade": val_ade,
                "val_fde": val_fde,
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

        #TODO: ADD OOD EVAL

        logger.log("-" * 30)

    # finalize
    logger.log("Training Complete.")
    tbd_writer.flush()
    tbd_writer.close()

def load_dataset(split_type="standard", batch_size=16, n_history=4, n_horizon=12, use_camera=False, use_lidar=False, use_bev=False):
    print("Loading Dataset...")
        
    
    train_data_pth = f'/home/vilin/Rapid_Adapt_SM/src/data/{split_type}/train.npz'
    val_data_pth = f'/home/vilin/Rapid_Adapt_SM/src/data/{split_type}/val.npz'
    test_data_pth = f'/home/vilin/Rapid_Adapt_SM/src/data/{split_type}/test.npz'
    ood_data_pth = f'/home/vilin/Rapid_Adapt_SM/src/data/{split_type}/ood.npz'
    
    raw_data_dir = '/home/vilin/Rapid_Adapt_SM/raw_data/nuscenes'
    
    
        
    print(f'Loading train dataset...')
    tr_dataset = NuScenesDataset(train_data_pth, raw_data_dir, n_history, n_horizon, use_camera=use_camera, use_lidar=use_lidar, use_bev=use_bev)
    print('Loaded!')
    stats = tr_dataset.compute_normalization_stats()
    print(f"Computed normalization stats: {stats}")
    tr_dataset.set_norm_stats(stats)
    print('Updated train dataset with normalization stats!')

    print(f'Loading val dataset...')
    val_dataset = NuScenesDataset(val_data_pth, raw_data_dir, n_history, n_horizon, use_camera=use_camera, use_lidar=use_lidar, use_bev=use_bev, norm_stats=stats)
    print(f'Loaded! {len(val_dataset)}')

    print(f'Loading test dataset...')
    test_dataset = NuScenesDataset(test_data_pth, raw_data_dir, n_history, n_horizon, use_camera=use_camera, use_lidar=use_lidar, use_bev=use_bev, norm_stats=stats)
    print(f'Loaded! {len(test_dataset)}')

    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, drop_last=True) # need to trop last for asynchronous deep supervision
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    pos_mean = stats["pos_mean"]
    pos_std  = stats["pos_std"]
    mean_xy = pos_mean[:2]                         # [2]
    std_xy  = pos_std[:2]                          # [2]

    print("Denormalize params: ", mean_xy, std_xy)

    if 'standard' not in args.split_type:
        print(f'Loading ood dataset...')
        ood_dataset = NuScenesDataset(ood_data_pth, raw_data_dir, use_camera=use_camera, use_lidar=use_lidar, use_bev=use_bev, norm_stats=stats)
        print(f'Loaded ood dataset! {len(ood_dataset)}')
        ood_dataloader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    else:
        ood_dataset = None
        ood_dataloader = None

    return tr_dataset, val_dataset, test_dataset, ood_dataset, tr_dataloader, val_dataloader, test_dataloader, ood_dataloader, stats, mean_xy, std_xy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="trm_av_unimodal_experiment_norm_v1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--halt_max_steps", type=int, default=1)
    parser.add_argument("--config_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4501)
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint if available")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use (0-based)")
    parser.add_argument('--split_type', type=str, default='standard', help='Dataset split methods. OOD splits of type oodType are named with convention oodType-oodSubType, where oodSubType does not appear in the ID train/val/test distribution.',
                                    choices=['standard',
                                             'city-boston', 'city-singapore',
                                             'map-boston-seaport', 'map-singapore-onenorth', 'map-singapore-queensto', 'map-singapore-hollandv',
                                             'object-animal', 'object-child', 'object-construction_worker', 'object-personal_mobility', 'object-police_officer', 'object-stroller', 'object-wheelchair',
                                             'object-debris', 'object-bicycle_rack',
                                             'object-bendy', 'object-ambulance', 'object-police'],)
    
    # modalities (always use pose data, but optionally add extra sensor data)
    parser.add_argument("--camera", action="store_true", help="Use camera data.")
    parser.add_argument("--lidar", action="store_true", help="Use raw LIDAR data.")
    parser.add_argument("--bev", action="store_true", help="Use processed BEV data.")
    
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
    tr_dataset, val_dataset, test_dataset, ood_dataset, tr_dataloader, val_dataloader, test_dataloader, ood_dataloader, stats, mean_xy, std_xy = load_dataset(args.split_type, args.config_batch_size, n_history, n_horizon, args.camera, args.lidar, args.bev)
    
    # train
    train(args, tr_dataset, val_dataset, test_dataset, ood_dataset, tr_dataloader, val_dataloader, test_dataloader, ood_dataloader, stats, mean_xy, std_xy)
