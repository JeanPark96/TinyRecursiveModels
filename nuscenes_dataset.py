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
import h5py
import torch
import numpy as np

DYNAMIC_TYPES = ['animal',
                 'adult', 'child', 'construction_worker', 'personal_mobility', 'police_officer', 'stroller', 'wheelchair',
                 'bicycle', 'bendy', 'rigid', 'car', 'construction', 'ambulance', 'police', 'motorcycle', 'trailer', 'truck',]

def load_dataset(args):
    print("Loading Dataset...")

    split_dir = args.split_type
    if args.bev:
        split_dir = f'bev-{split_dir}'
    if args.camera_FL or args.camera_FR or args.camera_B or args.camera_BL or args.camera_BR:
        split_dir = f'cam-{split_dir}'
    
    train_data_pth = f'/home/vilin/Rapid_Adapt_SM/src/data/{split_dir}/train.npz'
    val_data_pth = f'/home/vilin/Rapid_Adapt_SM/src/data/{split_dir}/val.npz'
    test_data_pth = f'/home/vilin/Rapid_Adapt_SM/src/data/{split_dir}/test.npz'
    ood_data_pth = f'/home/vilin/Rapid_Adapt_SM/src/data/{split_dir}/ood.npz'
    
    raw_data_dir = '/home/vilin/Rapid_Adapt_SM/raw_data/nuscenes'

    camera = {'F':args.camera_F,
              'FL':args.camera_FL,
              'FR':args.camera_FR,
              'B':args.camera_B,
              'BL':args.camera_BL,
              'BR':args.camera_BR}
    
    train_vid_feat_path = f"/home/vilin/Rapid_Adapt_SM/src/data/{args.split_type}_resnet_feat18/camera_features_train.h5"
    val_vid_feat_path = f"/home/vilin/Rapid_Adapt_SM/src/data/{args.split_type}_resnet_feat18/camera_features_val.h5"
    test_vid_feat_path = f"/home/vilin/Rapid_Adapt_SM/src/data/{args.split_type}_resnet_feat18/camera_features_test.h5"
    ood_vid_feat_path = f"/home/vilin/Rapid_Adapt_SM/src/data/{args.split_type}_resnet_feat18/camera_features_ood.h5"

        
    print(f'Loading train dataset...')
    tr_dataset = NuScenesDataset(train_data_pth, raw_data_dir, args.n_history, args.n_horizon, args.max_obstacles, args.max_predict, args.dynamic_only, use_camera=camera, use_lidar=args.lidar, use_bev=args.bev)
    print('Loaded!')
    stats = tr_dataset.compute_normalization_stats()
    print(f"Computed normalization stats: {stats}")
    tr_dataset.set_norm_stats(stats)
    print('Updated train dataset with normalization stats!')

    print(f'Loading val dataset...')
    val_dataset = NuScenesDataset(val_data_pth, raw_data_dir, args.n_history, args.n_horizon, args.max_obstacles, args.max_predict, args.dynamic_only, use_camera=camera, use_lidar=args.lidar, use_bev=args.bev, norm_stats=stats)
    print(f'Loaded! {len(val_dataset)}')

    print(f'Loading test dataset...')
    test_dataset = NuScenesDataset(test_data_pth, raw_data_dir, args.n_history, args.n_horizon, args.max_obstacles, args.max_predict, args.dynamic_only, use_camera=camera, use_lidar=args.lidar, use_bev=args.bev, norm_stats=stats)
    print(f'Loaded! {len(test_dataset)}')

    tr_dataloader = DataLoader(tr_dataset, batch_size=args.config_batch_size, shuffle=True, collate_fn=custom_collate, drop_last=True) # need to trop last for asynchronous deep supervision
    val_dataloader = DataLoader(val_dataset, batch_size=args.config_batch_size, shuffle=False, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=args.config_batch_size, shuffle=False, collate_fn=custom_collate)

    pos_mean = stats["pos_mean"]
    pos_std  = stats["pos_std"]
    mean_xy = pos_mean[:2]                         # [2]
    std_xy  = pos_std[:2]                          # [2]

    print("Denormalize params: ", mean_xy, std_xy)

    if 'standard' not in args.split_type:
        print(f'Loading ood dataset...')
        ood_dataset = NuScenesDataset(ood_data_pth, raw_data_dir, args.n_history, args.n_horizon, args.max_obstacles, args.max_predict, args.dynamic_only, use_camera=camera, use_lidar=args.lidar, use_bev=args.bev, norm_stats=stats)
        print(f'Loaded ood dataset! {len(ood_dataset)}')
        ood_dataloader = DataLoader(ood_dataset, batch_size=args.config_batch_size, shuffle=False, collate_fn=custom_collate)
    else:
        ood_dataset = None
        ood_dataloader = None

    return tr_dataset, val_dataset, test_dataset, ood_dataset, tr_dataloader, val_dataloader, test_dataloader, ood_dataloader, stats, mean_xy, std_xy

class HDF5FeatureLoader:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file = None
        self.dset = None

    def _open_file(self):
        # We lazily open the file in the worker process to avoid pickling errors
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.dset = self.h5_file['features']

    def get_features(self, idx):
        """
        Returns torch tensor for sample `idx`.
        Shape: [T, 512, 9, 16]
        """
        self._open_file()
        
        # Read from disk (fast slice)
        # Convert fp16 back to fp32 for PyTorch training stability
        data = self.dset[idx].astype(np.float32)
        
        return torch.from_numpy(data)

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

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
    collated['targets_idx'] = torch.stack([b['targets_idx'] for b in batch])

    # sensor data is not always used
    if 'camera_F' in batch[0]:
        collated['camera_F'] = torch.stack([b['camera_F'] for b in batch])
    if 'camera_FL' in batch[0]:
        collated['camera_FL'] = torch.stack([b['camera_FL'] for b in batch])
    if 'camera_FR' in batch[0]:
        collated['camera_FR'] = torch.stack([b['camera_FR'] for b in batch])
    if 'camera_B' in batch[0]:
        collated['camera_B'] = torch.stack([b['camera_B'] for b in batch])
    if 'camera_BL' in batch[0]:
        collated['camera_BL'] = torch.stack([b['camera_BL'] for b in batch])
    if 'camera_BR' in batch[0]:
        collated['camera_BR'] = torch.stack([b['camera_BR'] for b in batch])
    if 'lidar' in batch[0]:
        collated['lidar'] = [b['lidar'] for b in batch]
    if 'bev' in batch[0]:
        collated['bev'] = np.stack([b['bev'] for b in batch])
    if 'camera_F_features' in batch[0]:
        collated['camera_F_features'] = torch.stack([b['camera_F_features'] for b in batch])
    if 'camera_FL_features' in batch[0]:
        collated['camera_FL_features'] = torch.stack([b['camera_FL_features'] for b in batch])
    if 'camera_FR_features' in batch[0]:
        collated['camera_FR_features'] = torch.stack([b['camera_FR_features'] for b in batch])
    if 'camera_B_features' in batch[0]:
        collated['camera_B_features'] = torch.stack([b['camera_B_features'] for b in batch])
    if 'camera_BL_features' in batch[0]:
        collated['camera_BL_features'] = torch.stack([b['camera_BL_features'] for b in batch])
    if 'camera_BR_features' in batch[0]:
        collated['camera_BR_features'] = torch.stack([b['camera_BR_features'] for b in batch])

    return collated

class NuScenesDataset(Dataset):
    def __init__(self, data_pth, raw_data_dir, n_history, n_horizon, max_obstacles, max_predict, dynamic_only=False, use_camera=False, use_lidar=False, use_bev=False, use_preprocessed=False, feature_path=None, norm_stats=True):      
        self.use_camera_F = use_camera['F']
        self.use_camera_FL = use_camera['FL']
        self.use_camera_FR = use_camera['FR']
        self.use_camera_B = use_camera['B']
        self.use_camera_BL = use_camera['BL']
        self.use_camera_BR = use_camera['BR']
        self.use_lidar = use_lidar
        self.use_bev = use_bev
        self.use_preprocessed = use_preprocessed

        self.n_history = n_history
        self.n_horizon = n_horizon
        
        # load dataset
        data = np.load(data_pth, allow_pickle=True)
        assert n_horizon <= data['targets_mask'].shape[1]
        assert n_history <= data['obs_mask'].shape[1]
        assert max_obstacles == data['targets_mask'].shape[2], f"Max obstacles must be {data['targets_mask'].shape[2]}"
        assert max_predict <= max_obstacles
        
        # pose normalization statistics
        self.norm_stats = norm_stats

        # agent type
        self.obs_type = data['obs_type'] # strings: (n_examples, total_obstacles)

        # history agent mask
        self.obs_mask = torch.from_numpy(data['obs_mask'])[:, -n_history:, :].reshape(-1, n_history, max_obstacles)  # (n_examples, n_history, max_obstacles)
        
        # history ego-centric pose
        self.obs_pose = torch.from_numpy(data['obs_pose'])[:, -n_history:, :, :].reshape(-1, n_history, max_obstacles, 7).float() # (n_examples, n_history, max_obstacles, 7)
        
        # history global frame pose
        self.ego_pose = torch.from_numpy(data['ego_pose'])[:, -n_history:, :].reshape(-1, n_history, 7).float() # (n_examples, n_history, 7)
        self.raw_obs_pose = torch.from_numpy(data['raw_obs_pose'])[:, -n_history:, :, :].reshape(-1, n_history, max_obstacles, 7).float() # (n_examples, n_history, max_obstacles, 7)

        # select agents to predict based on current time step
        candidate_obstacles = self.obs_mask[:, -1, :] # (n_examples, max_obstacles)
        if dynamic_only:
            dynamic_idx = torch.from_numpy(np.isin(self.obs_type, DYNAMIC_TYPES)) # (n_examples, max_obstacles)
            candidate_obstacles = candidate_obstacles & dynamic_idx # (n_examples, max_obstacles)
        dists = torch.linalg.norm(self.obs_pose[:, -1, :, :2], dim=-1) # (n_examples, max_obstacles)
        masked_dists = dists.masked_fill(candidate_obstacles==0, float("inf")) # (n_examples, max_obstacles)
        sorted_idx = masked_dists.argsort(dim=1) # (n_examples, max_obstacles)
        self.target_idx = sorted_idx[:, :max_predict] # (n_examples, max_predict) may or may not all be valid obstacles, need dto refer to mask
        assert self.target_idx.shape[1] == max_predict

        # future agent mask
        self.targets_mask = np.take_along_axis(data['targets_mask'][:, :n_horizon, :], self.target_idx[:, None, :].numpy(), axis=2)
        self.targets_mask = torch.from_numpy(self.targets_mask).reshape(-1, n_horizon, max_predict) # (n_examples, n_horizon, max_predict)

        # future ego-centric pose
        self.targets = np.take_along_axis(data['targets'][:, :n_horizon, :, :], self.target_idx[:, None, :, None].numpy(), axis=2)
        self.targets = torch.from_numpy(self.targets).reshape(-1, n_horizon, max_predict, 7).float() # (n_examples, n_horizon, max_predict, 7)

        # future global frame pose
        self.ego_target = torch.from_numpy(data['ego_target'])[:, :n_horizon, :].reshape(-1, n_horizon, 7).float() # (n_examples, n_horizon, 7)
        self.raw_target = np.take_along_axis(data['raw_target'][:, :n_horizon, :, :], self.target_idx[:, None, :, None].numpy(), axis=2)
        self.raw_target = torch.from_numpy(self.raw_target).reshape(-1, n_horizon, max_predict, 7).float() # (n_examples, n_horizon, max_predict, 7)

        # optional sensor data
        if self.use_camera_F: self.camera_F_files = data['camera'] if 'camera' in data else data['camera_F']        # filepaths: (n_examples, n_history)
        if self.use_camera_FL: self.camera_FL_files = data['camera_FL'] # filepaths: (n_examples, n_history)
        if self.use_camera_FR: self.camera_FR_files = data['camera_FR'] # filepaths: (n_examples, n_history)
        if self.use_camera_B: self.camera_B_files = data['camera_B'] # filepaths: (n_examples, n_history)
        if self.use_camera_BL: self.camera_BL_files = data['camera_BL'] # filepaths: (n_examples, n_history)
        if self.use_camera_BR: self.camera_BR_files = data['camera_BR'] # filepaths: (n_examples, n_history)
        if self.use_lidar: self.lidar_files = data['lidar']             # filepaths: (n_examples, n_history)
        if self.use_bev:
            if 'bev' in data: # necessary for old versions where bev is not in data
                self.bev = self.unwrap_optional_array(data['bev'])     # None or (n_examples, n_history, 4, 256, 256)
            else:
                raise IndexError('Data type bev is not in this dataset.')

        self.n_samples = self.obs_pose.shape[0]
        self.raw_data_dir = raw_data_dir
                
        # Initialize Feature Loader if using preprocessed features
        if (self.use_camera_F or self.use_camera_FL or self.use_camera_FR or \
            self.use_camera_B or self.use_camera_BL or self.use_camera_BR) and \
            self.use_preprocessed:
            if feature_path is None:
                raise ValueError("You must provide 'feature_path' when use_preprocessed=True")
            
            self.feature_loader = HDF5FeatureLoader(feature_path)

    def __len__(self):
        return self.n_samples

    def unwrap_optional_array(self, x):
        if x.dtype == object and x.shape == () and x.item() is None:
            return None
        return x

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
            # selected agents to predict
            'targets_idx': self.target_idx[idx], # (max_predict)
            # agent masks
            'obs_mask': self.obs_mask[idx],         # (n_history, max_obstacles)
            'targets_mask': self.targets_mask[idx], # (n_horizon, max_predict) 
            # raw ego-centric poses
            'org_obs_pose': self.obs_pose[idx],         # (n_history, max_obstacles, 7)
            'org_targets': self.targets[idx],           # (n_horizon, max_predict, 7)
            # normalized ego-centric poses
            "obs_pose": norm_obs_pose,             # normalized xyz
            "targets": norm_targets,               # normalized xyz
            'idx': idx,                             # scalar
        }

        # Camera Logic
        if self.use_camera_F:
            if self.use_preprocessed:
                raise NotImplementedError
                # FAST: Load from HDF5
                # Since you have separate files for train/val, 
                # idx 0 in this dataset is guaranteed to be row 0 in the HDF5 file.
                sample['camera_F_features'] = self.feature_loader.get_features(idx)
            else:
                # SLOW: Load raw JPGs
                camera_F_seq = torch.stack([self.camera_loader(f) for f in self.camera_F_files[idx]])          
                sample.update(camera_F=camera_F_seq)        # list of n_history tensors
        if self.use_camera_FL:
            if self.use_preprocessed:
                raise NotImplementedError
                sample['camera_FL_features'] = self.feature_loader.get_features(idx)
            else:
                camera_FL_seq = torch.stack([self.camera_loader(f) for f in self.camera_FL_files[idx]])          
                sample.update(camera_FL=camera_FL_seq)        # list of n_history tensors
        if self.use_camera_FR:
            if self.use_preprocessed:
                raise NotImplementedError
                sample['camera_FR_features'] = self.feature_loader.get_features(idx)
            else:
                camera_FR_seq = torch.stack([self.camera_loader(f) for f in self.camera_FR_files[idx]])          
                sample.update(camera_FR=camera_FR_seq)        # list of n_history tensors
        if self.use_camera_B:
            if self.use_preprocessed:
                raise NotImplementedError
                sample['camera_B_features'] = self.feature_loader.get_features(idx)
            else:
                camera_B_seq = torch.stack([self.camera_loader(f) for f in self.camera_B_files[idx]])          
                sample.update(camera_B=camera_B_seq)        # list of n_history tensors
        if self.use_camera_BL:
            if self.use_preprocessed:
                raise NotImplementedError
                sample['camera_BL_features'] = self.feature_loader.get_features(idx)
            else:
                camera_BL_seq = torch.stack([self.camera_loader(f) for f in self.camera_BL_files[idx]])          
                sample.update(camera_BL=camera_BL_seq)        # list of n_history tensors
        if self.use_camera_BR:
            if self.use_preprocessed:
                raise NotImplementedError
                sample['camera_BR_features'] = self.feature_loader.get_features(idx)
            else:
                camera_BR_seq = torch.stack([self.camera_loader(f) for f in self.camera_BR_files[idx]])          
                sample.update(camera_BR=camera_BR_seq)        # list of n_history tensors
        
        if self.use_lidar:
            lidar_seq  = [self.lidar_loader(f).tolist() for f in self.lidar_files[idx]]
            sample.update(lidar=lidar_seq)          # list of n_history tensors
        if self.use_bev:
            sample.update(bev=self.bev[idx])        # (n_history, 4, 256, 256)
        
        return sample

    def get_obs_type(self, idx):
        '''
        idx: scalar or list of indices
        '''
        return self.obs_type[idx]           # (max_obstacles)
    
    def get_raw_data(self, idx):
        '''
        idx: scalar or list of indices
        '''
        return {
            'ego_pose':self.ego_pose[idx],              # (n_history, 7)
            'raw_obs_pose':self.raw_obs_pose[idx],      # (n_history, max_obstacles, 7)
            'ego_target':self.ego_target[idx],          # (n_horizon, 7)
            'raw_target':self.raw_target[idx],          # (n_horizon, max_predict, 7)
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