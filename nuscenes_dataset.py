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
    def __init__(self, data_pth, raw_data_dir, n_history, n_horizon, use_camera=False, use_lidar=False, use_bev=False, norm_stats=True):
        self.use_camera = use_camera
        self.use_lidar = use_lidar
        self.use_bev = use_bev

        self.n_history = n_history
        self.n_horizon = n_horizon
        
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
        # single agent only
        norm_obs_pose = self.normalize_positions(self.obs_pose[idx,:,0,:])
        norm_targets  = self.normalize_positions(self.targets[idx,:,0,:])
        sample = {
            # agent masks
            'obs_mask': self.obs_mask[idx,:,0].unsqueeze(1),         # (n_history,1)
            'targets_mask': self.targets_mask[idx,:,0].unsqueeze(1), # (n_horizon,1) 
            # raw ego-centric poses
            'org_obs_pose': self.obs_pose[idx,:,0,:].unsqueeze(1),         # (n_history, 1, 7)
            'org_targets': self.targets[idx,:,0,:].unsqueeze(1),           # (n_horizon, 1, 7)
            # normalized ego-centric poses
            "obs_pose": norm_obs_pose.unsqueeze(1),             # normalized xyz
            "targets": norm_targets.unsqueeze(1),               # normalized xyz
            'idx': idx,                             # scalar
        }

        # # Normalized pose (xyz only)
        # norm_obs_pose = self.normalize_positions(self.obs_pose[idx])
        # norm_targets  = self.normalize_positions(self.targets[idx])
        
        # sample = {
        #     # agent masks
        #     'obs_mask': self.obs_mask[idx],         # (n_history, MAX_OBSTACLES)
        #     'targets_mask': self.targets_mask[idx], # (n_horizon, MAX_OBSTACLES) 
        #     # raw ego-centric poses
        #     'org_obs_pose': self.obs_pose[idx],         # (n_history, MAX_OBSTACLES, 7)
        #     'org_targets': self.targets[idx],           # (n_horizon, MAX_OBSTACLES, 7)
        #     # normalized ego-centric poses
        #     "obs_pose": norm_obs_pose,             # normalized xyz
        #     "targets": norm_targets,               # normalized xyz
        #     'idx': idx,                             # scalar
        # }
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