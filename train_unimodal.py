import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import sys
import datetime
import json
import importlib
import models.recursive_reasoning.trm_unimodal as trm_unimodal
from torch.utils.tensorboard import SummaryWriter

importlib.reload(trm_unimodal)
# --- IMPORTS ---
# Ensure these imports match your file structure
# from my_dataset import NuScenesMiniDataset, custom_collate 
from models.recursive_reasoning.trm_unimodal import (
    TRM_ACT_NuScenes,
    TRM_ACT_NuScenes_Config
)
SAMPLE_FREQ = 2
max_obstacles = 30
n_history = 2*SAMPLE_FREQ # current time inclusive
n_horizon = 2*SAMPLE_FREQ

print(f"n_history: {n_history}, n_horizon: {n_horizon}")

'''
Standalone demo of how to use the preprocessed nuscenes data.

Requires standard packages plus:
    pip install nuscenes-devkit torch torchvision
'''
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision import transforms # <--- ADDED: Required for ResNet transformations
import os
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import random
import argparse

# --- 2. LOGGING HELPER ---

class Logger:
    def __init__(self, log_dir, run_name, config_dict):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{run_name}.log")
        
        # Initialize file and write header/config
        with open(self.log_path, 'w') as f:
            f.write(f"=== TRAINING LOG: {run_name} ===\n")
            f.write(f"Date: {datetime.datetime.now()}\n")
            f.write("=== CONFIGURATION ===\n")
            # Pretty print config dictionary
            f.write(json.dumps(config_dict, indent=4, default=str)) 
            f.write("\n=====================\n\n")
            
    def log(self, message):
        print(message) # Print to console
        with open(self.log_path, 'a') as f:
            f.write(message + "\n") # Write to file

@torch.no_grad()
def compute_ade_fde_real(pred_xy, target_xy, mask):
    """
    pred_xy:   [B, A, H, 2]  (denormalized)
    target_xy: [B, A, H, 2]  (denormalized)
    mask:      [B, A, H]
    """
    diff = pred_xy - target_xy           # [B, A, H, 2]
    dist = torch.sqrt((diff ** 2).sum(dim=-1))  # [B, A, H]

    # ensure float mask
    mask = mask.to(pred_xy.dtype)

    ade = (dist * mask).sum() / (mask.sum() + 1e-6)

    last_dist = dist[:, :, -1]               # [B, A]
    last_mask = mask[:, :, -1]
    fde = (last_dist * last_mask).sum() / (last_mask.sum() + 1e-6)

    return ade.item(), fde.item()


def custom_collate(batch):
    'Necessary while LIDAR data list of variable length lists. If LIDAR is converted to BEV, this is no longer necessary'
    collated = {}

    # stack tensor-like things
    collated['obs_pose'] = torch.stack([b['obs_pose'] for b in batch])
    collated['obs_mask'] = torch.stack([b['obs_mask'] for b in batch])
    collated['targets']  = torch.stack([b['targets'] for b in batch])
    collated['org_obs_pose'] = torch.stack([b['org_obs_pose'] for b in batch])
    collated['org_targets']  = torch.stack([b['org_targets'] for b in batch])
    collated['targets_mask']  = torch.stack([b['targets_mask'] for b in batch])
    collated['idx'] = np.stack([b['idx'] for b in batch])

    # sensor data is not always used
    if 'camera' in batch[0]:
        collated['camera'] = torch.stack([b['camera'] for b in batch])
    if 'lidar' in batch[0]:
        collated['lidar'] = [b['lidar'] for b in batch]
    if 'bev' in batch[0]:
        collated['bev'] = np.stack([b['bev'] for b in batch])

    return collated

class NuScenesDataset(Dataset):
    def __init__(self, data_pth, raw_data_dir, use_camera=False, use_lidar=False, use_bev=False, norm_stats=True):
        self.use_camera = use_camera
        self.use_lidar = use_lidar
        self.use_bev = use_bev
        
        # load dataset
        data = np.load(data_pth, allow_pickle=True)

        # pose normalization statistics
        self.norm_stats = norm_stats

        # agent masks
        self.obs_mask = torch.from_numpy(data['obs_mask'])                  # (n_examples, n_history, MAX_OBSTACLES)
        self.targets_mask = torch.from_numpy(data['targets_mask'])[:, :n_horizon, :]          # (n_examples, n_horizon, MAX_OBSTACLES)

        # agent type
        self.obs_type = data['obs_type' ]                                   # strings: (n_examples, n_history, MAX_OBSTACLES)

        # ego-centric pose
        self.obs_pose = torch.from_numpy(data['obs_pose']).float()          # (n_examples, n_history, MAX_OBSTACLES, 7)
        self.targets = torch.from_numpy(data['targets'])[:, :n_horizon, :].float()            # (n_examples, n_horizon, MAX_OBSTACLES, 7)
        
        # global frame pose
        self.ego_pose = torch.from_numpy(data['ego_pose']).float()          # (n_examples, n_history, 7)
        self.raw_obs_pose = torch.from_numpy(data['raw_obs_pose']).float()  # (n_examples, n_history, MAX_OBSTACLES, 7)
        self.ego_target = torch.from_numpy(data['ego_target'])[:, :n_horizon, :].float()      # (n_examples, n_horizon, 7)
        self.raw_target = torch.from_numpy(data['raw_target'])[:, :n_horizon, :].float()      # (n_examples, n_horizon, MAX_OBSTACLES, 7)
        
        # optional sensor data
        if self.use_camera: self.camera_files = data['camera']        # filepaths: (n_examples, n_history)
        if self.use_lidar: self.lidar_files = data['lidar']            # filepaths: (n_examples, n_history)
        if self.use_bev:
            if 'bev' in data: # necessary for old versions where bev is not in data
                self.bev = data['bev']                                          # None or (n_examples, n_history, 4, 256, 256)
            else:
                raise IndexError('Data type bev is not in this dataset.')

        self.n_samples = self.obs_pose.shape[0]
        self.raw_data_dir = raw_data_dir
        
         # --- IMPORTANT: RESNET-18 PREPROCESSING PIPELINE ---
        # 1. Resize to 256 (preserves aspect ratio)
        # 2. Crop center 224x224
        # 3. Convert to Tensor (0-1 float)
        # 4. Normalize with ImageNet mean/std
        self.resnet_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return self.n_samples

    def camera_loader(self, path):
        # based on pytorch pil_loader: https://docs.pytorch.org/vision/main/_modules/torchvision/datasets/folder.html#ImageFolder
        assert path.endswith('.jpg'), 'Unsupported filetype {}'.format(file_name)

        full_path = os.path.join(self.raw_data_dir, path)
        with open(full_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        return to_tensor(img)

    def lidar_loader(self, path):
        # based on nuscenes LidarPointCloud loader: https://github.com/nutonomy/nuscenes-devkit/blob/d9de17a73bdc06ce97a02f77ae7edb9b0406e851/python-sdk/nuscenes/utils/data_classes.py#L247
        assert path.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)

        full_path = os.path.join(self.raw_data_dir, path)
        scan = np.fromfile(full_path, dtype=np.float32)
        points = scan.reshape((-1, 5))[:, :4] # keep only x, y, z, intensity
        return torch.from_numpy(points)

    def __getitem__(self, idx):
        # Normalized pose (xyz only)
        norm_obs_pose = self.normalize_positions(self.obs_pose[idx])
        norm_targets  = self.normalize_positions(self.targets[idx])
        
        sample = {
            # agent masks
            'obs_mask': self.obs_mask[idx],         # (n_history, MAX_OBSTACLES)
            'targets_mask': self.targets_mask[idx], # (n_horizon, MAX_OBSTACLES) 
            # raw ego-centric poses
            'org_obs_pose': self.obs_pose[idx],         # (n_history, MAX_OBSTACLES, 7)
            'org_targets': self.targets[idx],           # (n_horizon, MAX_OBSTACLES, 7)
            # normalized ego-centric poses
            "obs_pose": norm_obs_pose,             # normalized xyz
            "targets": norm_targets,               # normalized xyz
            'idx': idx,                             # scalar
        }
        if self.use_camera:
            camera_seq = torch.stack([self.camera_loader(f) for f in self.camera_files[idx]])          
            sample.update(camera=camera_seq)        # list of n_history tensors
        if self.use_lidar:
            lidar_seq  = [self.lidar_loader(f).tolist() for f in self.lidar_files[idx]]
            sample.update(lidar=lidar_seq)          # list of n_history tensors
        if self.use_bev:
            sample.update(bev=self.bev[idx])        # (n_history, 4, 256, 256)
        
        return sample

    def get_obs_type(self, idx):
        return self.obs_type[idx]           # (n_history, MAX_OBSTACLES)
    
    def get_raw_data(self, idx):
        return {
            'ego_pose':self.ego_pose[idx],              # (n_history, 7)
            'raw_obs_pose':self.raw_obs_pose[idx],      # (n_history, MAX_OBSTACLES, 7)
            'ego_target':self.ego_target[idx],          # (n_horizon, 7)
            'raw_target':self.raw_target[idx],          # (n_horizon, MAX_OBSTACLES, 7)
        }
    

    def compute_normalization_stats(self, max_samples=None):
        """
        Compute mean and std for obstacle positions (x,y,z) and optionally yaw.
        Only uses training data. Uses obs_mask to ignore padded entries.

        max_samples: optionally limit number of samples for speed.
        """
        # Accumulators
        sum_pos = torch.zeros(3)
        sum_pos_sq = torch.zeros(3)
        count = 0

        N = self.n_samples if max_samples is None else min(self.n_samples, max_samples)

        for idx in range(N):
            # (H, K, 7)
            poses = self.obs_pose[idx][..., :3]       # (x,y,z)
            mask  = self.obs_mask[idx]                # (H,K)

            valid = mask.bool().reshape(-1)
            valid_poses = poses.reshape(-1, 3)[valid]

            if valid_poses.numel() == 0:
                continue

            sum_pos += valid_poses.sum(dim=0)
            sum_pos_sq += (valid_poses ** 2).sum(dim=0)
            count += valid_poses.shape[0]

        mean = sum_pos / count
        var  = (sum_pos_sq / count) - (mean ** 2)
        std  = torch.sqrt(var + 1e-8)

        stats = {
            "pos_mean": mean,
            "pos_std": std,
            "count": count,
        }
        return stats

    def set_norm_stats(self, stats):
        self.norm_stats = stats

    def get_norm_stats(self, stats):
        return self.norm_stats
    
    def normalize_positions(self, pos):
        """
        pos: (H, K, 7) or (T, K, 7) — only normalizes :3 (xyz).
        Returns a NEW tensor. Does not modify input.
        """
        if self.norm_stats is None:
            return pos  # No-op if not provided

        mean = self.norm_stats["pos_mean"].to(pos.device)
        std  = self.norm_stats["pos_std"].to(pos.device)

        out = pos.clone()
        out[..., :3] = (out[..., :3] - mean) / std
        return out

# --- helpers: loss + metrics consistent with your tensor shapes ---

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
    return (loss * m).sum() / (m.sum() + 1e-6)


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

import matplotlib.pyplot as plt

# plot for only one object in one sample
def plot_trajectories(pred_traj, target_traj, mask, title="Trajectories", RUN_NAME="model", filename = "model_default"):
    """
    pred_traj: [Future, 2]
    target_traj: [Future, 7]
    """
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    mask = mask > 0
    # Plot predicted trajectory
    
    plt.plot(pred_traj[:, 0].detach().cpu(), pred_traj[:, 1].detach().cpu(), 'ro--', label="Predicted", alpha=0.6)
    
    # Add index labels for Predicted
    for i, (x, y) in enumerate(pred_traj.detach().cpu()):
        plt.text(x, y, str(i), fontsize=9, color='black', alpha=0.8)

    # Plot ground truth trajectory
    #plt.plot(target_traj[:, 0].detach().cpu(), target_traj[:, 1].detach().cpu(), 'go--', label="Ground Truth")
    plt.plot(target_traj[mask, 0].detach().cpu(), target_traj[mask, 1].detach().cpu(), 'go--', label="Ground Truth", alpha=0.6)

    # Add index labels for Predicted
    for i, (x, y) in enumerate(target_traj[mask].detach().cpu()):
        plt.text(x, y, str(i), fontsize=9, color='blue', alpha=0.8)

    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    save_path = f'plot_figures/{RUN_NAME}/{filename}.png'
   
    os.makedirs(f"plot_figures/{RUN_NAME}", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    #plt.show()

# def run_act_steps(model, batch, carry):
#     """
#     If halt_max_steps==1, this is just one forward.
#     If >1, it loops until all sequences halted or max steps hit.
#     """
#     outputs = None
#     print("Running ACT steps...", model.config.halt_max_steps)
#     for _ in range(model.config.halt_max_steps):
#         carry, outputs = model(carry, batch)
#         # if training and ACT enabled, you can early stop when all halted
#         if carry.halted.all():
#             break
#     return carry, outputs

def run_act_steps(
    model,
    model_input,            # {"obs_pose","obs_mask"} on device
    carry,
    *,
    targets=None,           # [B,H,A,7] optional
    targets_mask=None,      # [B,H,A] optional
    out_slice=2,
    optimizer=None,
    do_opt_step=False,
    grad_clip=1.0,
    early_stop=True,
):
    outputs = None
    losses = []

    for _ in range(model.config.halt_max_steps):
        carry, outputs = model(carry, model_input)

        if targets is not None and targets_mask is not None:
            pred = outputs["pred"]
            loss = traj_loss_smooth_l1_from_batch(pred, targets, targets_mask, out_slice=out_slice)
            losses.append(loss)

            if do_opt_step:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        if early_stop and carry.halted.all():
            break

    loss_avg = None
    if losses:
        loss_avg = torch.stack([l.detach() for l in losses]).mean().item()

    return carry, outputs, {"loss_avg": loss_avg, "segments": len(losses)}



# Assumes these exist elsewhere in your codebase:
# - TRM_ACT_NuScenes
# - Logger
# - tr_dataloader, val_dataloader, tr_dataset, val_dataset
# - traj_loss_smooth_l1_from_batch
# - compute_ade_fde
# - run_act_steps
# - plot_trajectories


def select_debug_batch(dataloader, seed=None):
    """
    Choose a random batch index from the dataloader and return that batch.
    Used for plotting before training and after each epoch so we can track
    how predictions evolve on the same batch over time.
    """
    # if seed is not None:
    #     rng = np.random.RandomState(seed)
    #     debug_idx = rng.randint(len(dataloader))
    # else:
    #     debug_idx = random.randint(0, len(dataloader) - 1)

    # it = iter(dataloader)
    # debug_batch = None
    # for _ in range(debug_idx + 1):
    #     debug_batch = next(it)
    debug_batch = next(iter(dataloader))
    return debug_batch, 0


def plot_debug_batch(model, batch, device, epoch, run_name, out_slice):
    """
    Run the model on a fixed batch and plot the first few agents' trajectories.
    Called before training/resume (with the current model state) and after each epoch.
    """
    model.eval()
    with torch.no_grad():
        obs_pose = batch["obs_pose"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        targets = batch["targets"].to(device)
        targets_mask = batch.get("targets_mask", None)
        if targets_mask is None:
            targets_mask = (targets[..., :2].abs().sum(dim=-1) > 1e-3).to(obs_pose.dtype)
        else:
            targets_mask = targets_mask.to(device)

        model_input = {"obs_pose": obs_pose, "obs_mask": obs_mask}
        carry = model.initial_carry(model_input)
        # carry, outputs = run_act_steps(model, model_input, carry)
        # pred = outputs["pred"]  # [B, A, H, out_dim]
        carry, outputs, stats = run_act_steps(
                                            model,
                                            model_input,
                                            carry,
                                            targets=targets,
                                            targets_mask=targets_mask,
                                            out_slice=out_slice,
                                            optimizer=None,
                                            do_opt_step=False,      # <-- key for eval
                                            grad_clip=None,
                                            early_stop=False
                                    )

        pred = outputs["pred"]

        B, A, H, _ = pred.shape
        max_agents_to_plot = min(4, A)

        for i in range(max_agents_to_plot):
            pred_xy = pred[0, i, :, :2].detach().cpu()
            gt_xy = targets[0, :, i, :2].detach().cpu()
            targets_mask_cpu = targets_mask[0, :, i].detach().cpu()
            plot_trajectories(
                pred_xy,
                gt_xy,
                targets_mask_cpu,
                title=f"[Debug] Agent {i} @ epoch {epoch}",
                RUN_NAME=run_name,
                filename=f"{run_name}_debug_epoch{epoch}_agent{i}",
            )


def train(args, tr_dataset, val_dataset, test_dataset, ood_dataset, tr_dataloader, val_dataloader, test_dataloader, ood_dataloader, stats, mean_xy, std_xy):
    RUN_NAME = args.run_name
    LOG_DIR = "logs"
    CKPT_DIR = "checkpoints"
    TBOARD_DIR = "tboard"
    run_ckpt_dir = os.path.join(CKPT_DIR, RUN_NAME)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TBOARD_DIR, exist_ok=True)

    # Seeds for reproducibility (also used for picking debug batch)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    tbd_writer = SummaryWriter(os.path.join(TBOARD_DIR, f'{RUN_NAME}_{datetimestr}'))

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

    # --- Plot BEFORE training/resume (using current model state) ---
    plot_debug_batch(
        model,
        debug_batch,
        device,
        epoch=start_epoch,
        run_name=RUN_NAME,
        out_slice=config_dict["out_slice"],
    )

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs):
        logger.log(f"\n=== Starting Epoch {epoch+1}/{args.epochs} ===")
        model.train()

        # step-wise stats
        running_loss = 0.0
        running_ade = 0.0
        running_fde = 0.0
        running_ade_real = 0.0
        running_fde_real = 0.0
        running_count = 0

        # epoch-wise stats
        tr_ade_sum = 0.0
        tr_fde_sum = 0.0
        tr_ade_real_sum = 0.0
        tr_fde_real_sum = 0.0
        tr_loss_sum = 0.0
        tr_n = 0

        for batch_idx, batch in enumerate(tr_dataloader):
            obs_pose = batch["obs_pose"].to(device)              # [B, Hist, A, 7]
            obs_mask = batch["obs_mask"].to(device)              # [B, Hist, A]
            targets = batch["targets"].to(device)                # [B, Fut, A, 7]
            targets_mask = batch.get("targets_mask", None)
            if targets_mask is None:
                targets_mask = (targets[..., :2].abs().sum(dim=-1) > 1e-3).to(obs_pose.dtype)
            else:
                targets_mask = targets_mask.to(device)

            model_input = {
                "obs_pose": obs_pose,
                "obs_mask": obs_mask,
            }

            carry = model.initial_carry(model_input)
            # carry, outputs = run_act_steps(model, model_input, carry)
            # pred = outputs["pred"]  # [B, A, H, out_dim]

            # loss = traj_loss_smooth_l1_from_batch(
            #     pred, targets, targets_mask, out_slice=config_dict["out_slice"]
            # )

            # optimizer.zero_grad(set_to_none=True)
            # loss.backward()
            # clip_grad_norm_(model.parameters(), 1.0)
            # optimizer.step()

            # global_step += 1
            # running_loss += loss.item()
            
            carry, outputs, stats = run_act_steps(
                                            model,
                                            model_input,
                                            carry,
                                            targets=targets,
                                            targets_mask=targets_mask,
                                            out_slice=config_dict["out_slice"],
                                            optimizer=optimizer,
                                            do_opt_step=True,      # <-- key for training
                                            grad_clip=1.0,
                                            early_stop=True
                                    )
            global_step += stats["segments"]
            running_loss += stats["loss_avg"]
            pred = outputs["pred"]
            ade, fde = compute_ade_fde(
                pred, targets, targets_mask, out_slice=config_dict["out_slice"]
            )

            # AFTER you get pred, targets, targets_mask
            # pred: [B, A, H, 2] (normalized)

            pred_xy_denorm = pred[..., :2] * std_xy + mean_xy               # [B, A, H, 2]
            targets_xy_denorm = (targets[..., :2] * std_xy + mean_xy)       # [B, H, A, 2]
            #targets_xy_denorm = targets_xy_denorm.permute(0, 2, 1, 3)       # [B, A, H, 2]

            ade_real, fde_real = compute_ade_fde(
                pred_xy_denorm, targets_xy_denorm, targets_mask, out_slice=config_dict["out_slice"]
            )

            running_ade += ade
            running_fde += fde
            running_ade_real += ade_real
            running_fde_real += fde_real
            running_count += 1

            if global_step % 10 == 0:
                logger.log(
                    f"Epoch [{epoch+1}] Step [{global_step}] "
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
            tr_loss_sum += stats["loss_avg"]
            tr_n += 1

        # tensorboard logging
        tbd_writer.add_scalar(f"Loss/train", tr_loss_sum / max(tr_n, 1), epoch+1)
        tbd_writer.add_scalar(f"ADE/train", tr_ade_sum / max(tr_n, 1), epoch+1)
        tbd_writer.add_scalar(f"FDE/train", tr_fde_sum / max(tr_n, 1), epoch+1)
        tbd_writer.add_scalar(f"ADE_real/train", tr_ade_real_sum / max(tr_n, 1), epoch+1)
        tbd_writer.add_scalar(f"FDE_real/train", tr_fde_real_sum / max(tr_n, 1), epoch+1)

        # --- Validation ---
        logger.log("Running Validation...")
        model.eval()
        val_ade_sum = 0.0
        val_fde_sum = 0.0
        val_ade_real_sum = 0.0
        val_fde_real_sum = 0.0
        val_loss_sum = 0.0
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

                model_input = {"obs_pose": obs_pose, "obs_mask": obs_mask}
                carry = model.initial_carry(model_input)
                # carry, outputs = run_act_steps(model, model_input, carry)
                # pred = outputs["pred"]

                carry, outputs, stats = run_act_steps(
                                            model,
                                            model_input,
                                            carry,
                                            targets=targets,
                                            targets_mask=targets_mask,
                                            out_slice=config_dict["out_slice"],
                                            optimizer=None,
                                            do_opt_step=False,      # <-- key for eval
                                            grad_clip=None,
                                            early_stop=False
                                    )

                pred = outputs["pred"]

                ade, fde = compute_ade_fde(
                    pred, targets, targets_mask, out_slice=config_dict["out_slice"]
                )
                val_ade_sum += ade
                val_fde_sum += fde

                pred_xy_denorm = pred[..., :2] * std_xy + mean_xy               # [B, A, H, 2]
                targets_xy_denorm = (targets[..., :2] * std_xy + mean_xy)       # [B, H, A, 2]
                #targets_xy_denorm = targets_xy_denorm.permute(0, 2, 1, 3)       # [B, A, H, 2]
                #print()
                ade_real, fde_real = compute_ade_fde(
                    pred_xy_denorm, targets_xy_denorm, targets_mask, out_slice=config_dict["out_slice"]
                )

                val_ade_real_sum += ade_real
                val_fde_real_sum += fde_real

                # batch_loss = traj_loss_smooth_l1_from_batch(
                #     pred, targets, targets_mask, out_slice=config_dict["out_slice"]
                # )
                #val_loss_sum += batch_loss.item()
                val_loss_sum += stats["loss_avg"]

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
            model,
            debug_batch,
            device,
            epoch=epoch+1,
            run_name=RUN_NAME,
            out_slice=config_dict["out_slice"],
        )

        # --- Save epoch checkpoint ---
        epoch_ckpt_path = os.path.join(run_ckpt_dir, f"epoch_{epoch+1}.pth")
        ckpt_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "epoch": epoch,  # 0-based
            "config": config_dict,
            "val_loss": val_loss,
            "val_ade": val_ade,
            "val_fde": val_fde,
            "best_val_loss": best_val_loss,
        }
        torch.save(ckpt_payload, epoch_ckpt_path)
        logger.log(f"Saved epoch checkpoint to {epoch_ckpt_path}")

        # --- Save/update last checkpoint for resume ---
        #torch.save(ckpt_payload, last_ckpt_path)
        #logger.log(f"Updated last checkpoint at {last_ckpt_path}")

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

        # --- OOD ---
        if ood_dataloader is not None:
            logger.log("Running OOD evaluation...")
            model.eval()
            ood_ade_sum = 0.0
            ood_fde_sum = 0.0
            ood_ade_real_sum = 0.0
            ood_fde_real_sum = 0.0
            ood_loss_sum = 0.0
            ood_n = 0

            with torch.no_grad():
                for batch in ood_dataloader:
                    obs_pose = batch["obs_pose"].to(device)
                    obs_mask = batch["obs_mask"].to(device)
                    targets = batch["targets"].to(device)
                    targets_mask = batch.get("targets_mask", None)
                    if targets_mask is None:
                        targets_mask = (targets[..., :2].abs().sum(dim=-1) > 1e-3).to(obs_pose.dtype)
                    else:
                        targets_mask = targets_mask.to(device)

                    model_input = {"obs_pose": obs_pose, "obs_mask": obs_mask}
                    carry = model.initial_carry(model_input)
                    # carry, outputs = run_act_steps(model, model_input, carry)
                    # pred = outputs["pred"]

                    carry, outputs, stats = run_act_steps(
                                            model,
                                            model_input,
                                            carry,
                                            targets=targets,
                                            targets_mask=targets_mask,
                                            out_slice=config_dict["out_slice"],
                                            optimizer=None,
                                            do_opt_step=False,      # <-- key for eval
                                            grad_clip=None,
                                            early_stop=False
                                    )

                    pred = outputs["pred"]

                    ade, fde = compute_ade_fde(
                        pred, targets, targets_mask, out_slice=config_dict["out_slice"]
                    )
                    ood_ade_sum += ade
                    ood_fde_sum += fde

                    pred_xy_denorm = pred[..., :2] * std_xy + mean_xy               # [B, A, H, 2]
                    targets_xy_denorm = (targets[..., :2] * std_xy + mean_xy)       # [B, H, A, 2]
                    #targets_xy_denorm = targets_xy_denorm.permute(0, 2, 1, 3)       # [B, A, H, 2]

                    ade_real, fde_real = compute_ade_fde(
                        pred_xy_denorm, targets_xy_denorm, targets_mask, out_slice=config_dict["out_slice"]
                    )

                    ood_ade_real_sum += ade_real
                    ood_fde_real_sum += fde_real

                    # batch_loss = traj_loss_smooth_l1_from_batch(
                    #     pred, targets, targets_mask, out_slice=config_dict["out_slice"]
                    # )
                    # ood_loss_sum += batch_loss.item()
                    ood_loss_sum += stats["loss_avg"]


                    ood_n += 1

            ood_ade = ood_ade_sum / max(ood_n, 1)
            ood_fde = ood_fde_sum / max(ood_n, 1)
            ood_ade_real = ood_ade_real_sum / max(ood_n, 1)
            ood_fde_real = ood_fde_real_sum / max(ood_n, 1)
            ood_loss = ood_loss_sum / max(ood_n, 1)

            logger.log(
                f"OOD Eval Results - Epoch {epoch+1}: "
                f"Loss: {ood_loss:.4f} | "
                f"ADE: {ood_ade:.4f} | "
                f"FDE: {ood_fde:.4f} | "
                f"Real ADE: {ood_ade_real:.4f} | "
                f"Real FDE: {ood_fde_real:.4f}"
            )

            # tensorboard logging
            tbd_writer.add_scalar(f"Loss/ood",ood_loss,epoch+1)
            tbd_writer.add_scalar(f"ADE/ood",ood_ade,epoch+1)
            tbd_writer.add_scalar(f"FDE/ood",ood_fde,epoch+1)
            tbd_writer.add_scalar(f"ADE_real/ood",ood_ade_real,epoch+1)
            tbd_writer.add_scalar(f"FDE_real/ood",ood_fde_real,epoch+1)

            # --- Plot debug batch AFTER this epoch ---
            plot_debug_batch(
                model,
                ood_debug_batch,
                device,
                epoch=epoch+1,
                run_name=RUN_NAME + "_ood",
                out_slice=config_dict["out_slice"],
            )

        logger.log("-" * 30)

    logger.log("Training Complete.")

    # tboard
    tbd_writer.flush()
    tbd_writer.close()

def load_dataset(split_type="standard", batch_size=16, use_camera=False, use_lidar=False, use_bev=False):
    print("Loading Dataset...")
        
    
    train_data_pth = f'/home/vilin/Rapid_Adapt_SM/src/data/{split_type}/train.npz'
    val_data_pth = f'/home/vilin/Rapid_Adapt_SM/src/data/{split_type}/val.npz'
    test_data_pth = f'/home/vilin/Rapid_Adapt_SM/src/data/{split_type}/test.npz'
    ood_data_pth = f'/home/vilin/Rapid_Adapt_SM/src/data/{split_type}/ood.npz'
    
    raw_data_dir = '/home/vilin/Rapid_Adapt_SM/raw_data/nuscenes'
    
    
        
    print(f'Loading train dataset...')
    tr_dataset = NuScenesDataset(train_data_pth, raw_data_dir, use_camera=use_camera, use_lidar=use_lidar, use_bev=use_bev)
    print('Loaded!')
    stats = tr_dataset.compute_normalization_stats()
    print(f"Computed normalization stats: {stats}")
    tr_dataset.set_norm_stats(stats)
    print('Updated train dataset with normalization stats!')

    print(f'Loading val dataset...')
    val_dataset = NuScenesDataset(val_data_pth, raw_data_dir, use_camera=use_camera, use_lidar=use_lidar, use_bev=use_bev, norm_stats=stats)
    print(f'Loaded! {len(val_dataset)}')

    print(f'Loading test dataset...')
    test_dataset = NuScenesDataset(test_data_pth, raw_data_dir, use_camera=use_camera, use_lidar=use_lidar, use_bev=use_bev, norm_stats=stats)
    print(f'Loaded! {len(test_dataset)}')

    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
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
    

    tr_dataset, val_dataset, test_dataset, ood_dataset, tr_dataloader, val_dataloader, test_dataloader, ood_dataloader, stats, mean_xy, std_xy = load_dataset(args.split_type, args.config_batch_size, args.camera, args.lidar, args.bev)
    train(args, tr_dataset, val_dataset, test_dataset, ood_dataset, tr_dataloader, val_dataloader, test_dataloader, ood_dataloader, stats, mean_xy, std_xy)
