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

from tqdm import tqdm
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
from nuscenes_dataset import load_dataset
from utils.metrics import compute_metrics
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
import models.recursive_reasoning.trm_unimodal_v4 as trm_unimodal
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

importlib.reload(trm_unimodal)
# --- IMPORTS ---
# Ensure these imports match your file structure
# from my_dataset import NuScenesMiniDataset, custom_collate 
from models.recursive_reasoning.trm_unimodal_v4 import (
    TRM_ACT_NuScenes,
    TRM_ACT_NuScenes_Config
)

SAMPLE_FREQ = 2
# max_obstacles = 1 #30
# n_history = 2*SAMPLE_FREQ # current time inclusive
# n_horizon = 2*SAMPLE_FREQ

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
        optimizer_lr_schedule=False,
        optimizer_lr_min_ratio=0.01,
        optimizer_lr_warmup_steps=2000,
        carry=None
    )

    # Testing Loop
    if ood:
        logger.log("Running OOD...")
    else:
        logger.log("Running Test...")
    train_state.model.eval()
    ade_sum = fde_sum = ade_real_sum = fde_real_sum = mr_sum = loss_sum = 0.0
    n = 0

    with torch.no_grad():
        for b, batch in enumerate(tqdm(dataloader)):
            # if b % 10 == 0:
            #     print(f'Batch {b}')

            obs_pose = batch["obs_pose"].to(device)
            obs_mask = batch["obs_mask"].to(device)
            targets = batch["targets"].to(device)
            targets_mask = batch.get("targets_mask", None)
            targets_idx = batch.get("targets_idx", None).to(device)
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
                "targets_idx": targets_idx,
            }

            with torch.device("cuda"):
                carry = train_state.model.initial_carry(model_input)  # type: ignore

            # Forward
            inference_steps = 0
            supervisions = []
            while True:
                carry, loss, metrics, outputs, all_finish = train_state.model(
                    carry=carry, batch=model_input, return_keys=["pred", "pred_recursions"]
                )
                inference_steps += 1
                supervisions.append((outputs["pred"], outputs["pred_recursions"]))

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
                sub_name=args.tboard_name,
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
            ade, fde, _ = compute_metrics(
                pred, targets, targets_mask, only_full=True, history_mask=obs_mask, targets_idx=targets_idx, out_slice=config_dict["out_slice"]
            )

            pred_xy_denorm = pred[..., :2] * std_xy + mean_xy               # [B, A, H, 2]
            targets_xy_denorm = (targets[..., :2] * std_xy + mean_xy)       # [B, H, A, 2]

            ade_real, fde_real, mr = compute_metrics(
                pred_xy_denorm, targets_xy_denorm, targets_mask, only_full=True, history_mask=obs_mask, targets_idx=targets_idx, out_slice=config_dict["out_slice"]
            )

            # log metrics
            ade_sum += ade
            fde_sum += fde
            ade_real_sum += ade_real
            fde_real_sum += fde_real
            mr_sum += mr
            loss_sum += reduced_metrics['val/loss']
            n += 1

            # plot recursions
            if b == 0:
                pred_recursions = outputs["pred_recursions"] # [H_cycles, B, A*H+1, D]
                # print(pred_recursions[:,0,0,0])
                # raise NotImplementedError
                valid_batch_idxs = valid_agent_idxs = None
                for s, sup in enumerate(supervisions):
                    pred, z_Hs = sup
                    for cycle in range(z_Hs.size(0)):
                        if cycle == z_Hs.size(0)-1: # for last recursion, we already have the output prediction
                            recursion_out = pred
                        else:
                            recursion_out = train_state.model.decode(z_Hs[cycle], model_input)
                        valid_batch_idxs, valid_agent_idxs = plot_test_batch(
                            dataset,
                            batch,
                            b,
                            recursion_out,
                            device,
                            run_name=f'recursions_{RUN_NAME}',
                            out_slice=config_dict["out_slice"],
                            sub_name=args.tboard_name,
                            filename=f'supervision{s}_recursion{cycle}',
                            valid_batch_idxs=valid_batch_idxs,
                            valid_agent_idxs=valid_agent_idxs,
                        )

    ave_ade = ade_sum / max(n, 1)
    ave_fde = fde_sum / max(n, 1)
    ave_ade_real = ade_real_sum / max(n, 1)
    ave_fde_real = fde_real_sum / max(n, 1)
    mr = mr_sum / max(n, 1)
    ave_loss = loss_sum / max(n, 1)

    if ood:
        logger.log(
            f"OOD Results: "
            f"Loss: {ave_loss:.4f} | "
            f"ADE: {ave_ade:.4f} | "
            f"FDE: {ave_fde:.4f} | "
            f"Real ADE: {ave_ade_real:.4f} | "
            f"Real FDE: {ave_fde_real:.4f} | "
            f"Miss rate: {mr:.4f}"
        )
        logger.log("OOD Complete.")
    else:
        logger.log(
            f"Test Results: "
            f"Loss: {ave_loss:.4f} | "
            f"ADE: {ave_ade:.4f} | "
            f"FDE: {ave_fde:.4f} | "
            f"Real ADE: {ave_ade_real:.4f} | "
            f"Real FDE: {ave_fde_real:.4f} | "
            f"Miss rate: {mr:.4f}"
        )
        logger.log("Testing Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="trm_av_unimodal_experiment_norm_v1")
    parser.add_argument("--tboard_name", type=str, help="Name for tensorboard run")
    parser.add_argument("--model_pth", type=str)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--halt_max_steps", type=int, default=16)
    parser.add_argument("--config_batch_size", type=int, default=32)
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
    parser.add_argument("--camera_F", action="store_true", help="Use front camera data.")
    parser.add_argument("--camera_FL", action="store_true", help="Use front left camera data.")
    parser.add_argument("--camera_FR", action="store_true", help="Use front right camera data.")
    parser.add_argument("--camera_B", action="store_true", help="Use back camera data.")
    parser.add_argument("--camera_BL", action="store_true", help="Use back left camera data.")
    parser.add_argument("--camera_BR", action="store_true", help="Use back right camera data.")
    parser.add_argument("--preprocessed_vid_fea", action="store_true", help="Use preprocessed video features.")
    parser.add_argument("--lidar", action="store_true", help="Use raw LIDAR data.")
    parser.add_argument("--bev", action="store_true", help="Use processed BEV data.")
    parser.add_argument('--map', action='store_true', help='Add map context.')
    
    # task parameters (non-defaults are used for sanity checking and testing)
    parser.add_argument("--history_sec", type=int, default=2, help='Length of history in seconds')
    parser.add_argument("--horizon_sec", type=int, default=6, help='Length of future in seconds')
    parser.add_argument("--max_obstacles", type=int, default=30, help='Max number of obstacles considered')
    parser.add_argument("--max_predict", type=int, default=8, help='Max number of obstacles to predict.')
    parser.add_argument("--dynamic_only", action="store_true", help="Only predict dynamic agents.")
    parser.add_argument("--feature_set", type=str, choices=['hpnet'], help="Types of map features to use")

    args = parser.parse_args()

    assert os.path.exists(args.model_pth)

    # update task parameters
    args.n_history = args.history_sec*SAMPLE_FREQ # current time inclusive
    args.n_horizon = args.horizon_sec*SAMPLE_FREQ
    print(f"n_history: {args.n_history}, n_horizon: {args.n_horizon}")

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
    _, _, test_dataset, ood_dataset, _, _, test_dataloader, ood_dataloader, stats, mean_xy, std_xy = load_dataset(args)

    # test
    eval(args, test_dataset, test_dataloader, stats, mean_xy, std_xy, ood=False)
    if 'standard' not in args.split_type:
        eval(args, ood_dataset, ood_dataloader, stats, mean_xy, std_xy, ood=True)
