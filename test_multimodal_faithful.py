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
from utils.debug import plot_trajectories, plot_test_batch
from models.losses import ACTLossHeadNuScenes
import random
import numpy as np
import json
import datetime
import sys
import importlib
import models.recursive_reasoning.trm_multimodal as trm_multimodal
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

# --- IMPORTS ---
# Ensure these imports match your file structure
# from my_dataset import NuScenesMiniDataset, custom_collate 
from models.recursive_reasoning.trm_multimodal import (
    TRM_ACT_NuScenes,
    TRM_ACT_NuScenes_Config
)

SAMPLE_FREQ = 2
max_obstacles = 30 #30
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

def eval(args, dataset, dataloader, stats, mean_xy, std_xy, ood=False):
    if ood:
        RUN_NAME = f'{args.run_name}_ood' 
    else:
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

    mean_xy = mean_xy.to(device)
    std_xy = std_xy.to(device)

    # Load trained model
    ckpt = torch.load(args.model_pth, map_location="cpu", weights_only=False)
    config_dict = ckpt["config"]
    start_epoch = ckpt.get("epoch", 0) + 1  # epoch stored as 0-based
    global_step = ckpt.get("global_step", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))

    # Initialize Logger
    logger = Logger(LOG_DIR, RUN_NAME, config_dict)
    logger.log("Loading Dataset...")
    logger.log(f"Dataset Loaded. Test samples: {len(test_dataset)}")

    # --- Model & optimizer ---
    logger.log("Initializing Model...")
    model = TRM_ACT_NuScenes(config_dict).to(device)
    model.load_state_dict(ckpt["model"])

    # Train state
    train_state = TrainState(
        step=0,
        total_steps=0, # unused for now

        model=ACTLossHeadNuScenes(model=model),
        optimizers=[None],
        optimizer_lrs=[None],
        carry=None
    )

    # Testing Loop
    if ood:
        logger.log("Running OOD...")
    else:
        logger.log("Running Test...")
    train_state.model.eval()
    ade_sum = fde_sum = ade_real_sum = fde_real_sum = loss_sum = 0.0
    n = 0

    with torch.no_grad():
        for b, batch in enumerate(dataloader):
            if b % 10 == 0:
                print(f'Batch {b}')

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

            # plot batch
            plot_test_batch(
                dataset,
                batch,
                b,
                outputs,
                device,
                run_name=f'test_{RUN_NAME}',
                out_slice=config_dict["out_slice"],
            )

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
            ade_sum += ade
            fde_sum += fde
            ade_real_sum += ade_real
            fde_real_sum += fde_real
            loss_sum += reduced_metrics['val/loss']
            n += 1

    ave_ade = ade_sum / max(n, 1)
    ave_fde = fde_sum / max(n, 1)
    ave_ade_real = ade_real_sum / max(n, 1)
    ave_fde_real = fde_real_sum / max(n, 1)
    ave_loss = loss_sum / max(n, 1)

    if ood:
        logger.log(
            f"OOD Results: "
            f"Loss: {ave_loss:.4f} | "
            f"ADE: {ave_ade:.4f} | "
            f"FDE: {ave_fde:.4f} | "
            f"Real ADE: {ave_ade_real:.4f} | "
            f"Real FDE: {ave_fde_real:.4f}"
        )
        logger.log("OOD Complete.")
    else:
        logger.log(
            f"Test Results: "
            f"Loss: {ave_loss:.4f} | "
            f"ADE: {ave_ade:.4f} | "
            f"FDE: {ave_fde:.4f} | "
            f"Real ADE: {ave_ade_real:.4f} | "
            f"Real FDE: {ave_fde_real:.4f}"
        )
        logger.log("Testing Complete.")


def load_dataset(args):
    print("Loading Dataset...")
        
    
    train_data_pth = f'/home/vilin/Rapid_Adapt_SM/src/data/{args.split_type}/train.npz'
    val_data_pth = f'/home/vilin/Rapid_Adapt_SM/src/data/{args.split_type}/val.npz'
    test_data_pth = f'/home/vilin/Rapid_Adapt_SM/src/data/{args.split_type}/test.npz'
    ood_data_pth = f'/home/vilin/Rapid_Adapt_SM/src/data/{args.split_type}/ood.npz'
    
    raw_data_dir = '/home/vilin/Rapid_Adapt_SM/raw_data/nuscenes'
    
    train_vid_feat_path = f"/home/vilin/Rapid_Adapt_SM/src/data/{args.split_type}_resnet_feat18/camera_features_train.h5"
    val_vid_feat_path = f"/home/vilin/Rapid_Adapt_SM/src/data/{args.split_type}_resnet_feat18/camera_features_val.h5"
    test_vid_feat_path = f"/home/vilin/Rapid_Adapt_SM/src/data/{args.split_type}_resnet_feat18/camera_features_test.h5"
    ood_vid_feat_path = f"/home/vilin/Rapid_Adapt_SM/src/data/{args.split_type}_resnet_feat18/camera_features_ood.h5"

        
    print(f'Loading train dataset...')
    tr_dataset = NuScenesDataset(train_data_pth, raw_data_dir, args.n_history, args.n_horizon, args.max_obstacles, use_camera=args.camera, use_lidar=args.lidar, use_bev=args.bev, use_preprocessed=args.preprocessed_vid_fea, feature_path=train_vid_feat_path)
    print('Loaded!')
    stats = tr_dataset.compute_normalization_stats()
    print(f"Computed normalization stats: {stats}")

    print(f'Loading test dataset...')
    test_dataset = NuScenesDataset(test_data_pth, raw_data_dir, args.n_history, args.n_horizon, args.max_obstacles, use_camera=args.camera, use_lidar=args.lidar, use_bev=args.bev, use_preprocessed=args.preprocessed_vid_fea, feature_path=test_vid_feat_path, norm_stats=stats)
    print(f'Loaded! {len(test_dataset)}')

    test_dataloader = DataLoader(test_dataset, batch_size=args.config_batch_size, shuffle=False, collate_fn=custom_collate)

    pos_mean = stats["pos_mean"]
    pos_std  = stats["pos_std"]
    mean_xy = pos_mean[:2]                         # [2]
    std_xy  = pos_std[:2]                          # [2]

    print("Denormalize params: ", mean_xy, std_xy)

    if 'standard' not in args.split_type:
        print(f'Loading ood dataset...')
        ood_dataset = NuScenesDataset(ood_data_pth, raw_data_dir, args.n_history, args.n_horizon, args.max_obstacles, use_camera=args.camera, use_lidar=args.lidar, use_bev=args.bev, use_preprocessed=args.preprocessed_vid_fea, feature_path=ood_vid_feat_path, norm_stats=stats)
        print(f'Loaded ood dataset! {len(ood_dataset)}')
        ood_dataloader = DataLoader(ood_dataset, batch_size=args.config_batch_size, shuffle=False, collate_fn=custom_collate)
    else:
        ood_dataset = None
        ood_dataloader = None

    return test_dataset, ood_dataset, test_dataloader, ood_dataloader, stats, mean_xy, std_xy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="trm_av_unimodal_experiment_norm_v1")
    parser.add_argument("--model_pth", type=str)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--halt_max_steps", type=int, default=1)
    parser.add_argument("--config_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4501)
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
    parser.add_argument("--preprocessed_vid_fea", action="store_true", help="Use preprocessed video features.")
    
    # task parameters (non-defaults are used for sanity checking and testing)
    parser.add_argument("--history_sec", type=int, default=2, help='Length of history in seconds')
    parser.add_argument("--horizon_sec", type=int, default=6, help='Length of future in seconds')
    parser.add_argument("--max_obstacles", type=int, default=30, help='Max number of obstacles considered')
    
    args = parser.parse_args()

    # update task parameters
    args.n_history = args.history_sec*SAMPLE_FREQ # current time inclusive
    args.n_horizon = args.horizon_sec*SAMPLE_FREQ

    assert os.path.exists(args.model_pth)

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
    test_dataset, ood_dataset, test_dataloader, ood_dataloader, stats, mean_xy, std_xy = load_dataset(args)
    # test
    eval(args, test_dataset, test_dataloader, stats, mean_xy, std_xy, ood=False)
    eval(args, ood_dataset, ood_dataloader, stats, mean_xy, std_xy, ood=True)
