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
from nuscenes.map_expansion.map_api import NuScenesMap
from tqdm import tqdm
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    if args.map:
        split_dir = f"loc-{split_dir}"

    
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
    tr_dataset = NuScenesDataset(train_data_pth, raw_data_dir, args.n_history, args.n_horizon, args.max_obstacles, args.max_predict, args.dynamic_only, use_camera=camera, use_lidar=args.lidar, use_bev=args.bev, use_map=args.map, feature_set=args.feature_set)
    print('Loaded!')
    stats = tr_dataset.compute_normalization_stats()
    print(f"Computed normalization stats: {stats}")
    tr_dataset.set_norm_stats(stats)
    print('Updated train dataset with normalization stats!')

    print(f'Loading val dataset...')
    val_dataset = NuScenesDataset(val_data_pth, raw_data_dir, args.n_history, args.n_horizon, args.max_obstacles, args.max_predict, args.dynamic_only, use_camera=camera, use_lidar=args.lidar, use_bev=args.bev, use_map=args.map, feature_set=args.feature_set, norm_stats=stats)
    print(f'Loaded! {len(val_dataset)}')

    print(f'Loading test dataset...')
    test_dataset = NuScenesDataset(test_data_pth, raw_data_dir, args.n_history, args.n_horizon, args.max_obstacles, args.max_predict, args.dynamic_only, use_camera=camera, use_lidar=args.lidar, use_bev=args.bev, use_map=args.map, feature_set=args.feature_set, norm_stats=stats)
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
        ood_dataset = NuScenesDataset(ood_data_pth, raw_data_dir, args.n_history, args.n_horizon, args.max_obstacles, args.max_predict, args.dynamic_only, use_camera=camera, use_lidar=args.lidar, use_bev=args.bev, use_map=args.map, feature_set=args.feature_set, norm_stats=stats)
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
        # We don't pre-load datasets here to avoid pickling issues with DataLoader workers

    def _open_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r", swmr=True, libver="latest")

    def get_features(self, idx, sensor_name):
        """
        Returns torch tensor for sample `idx` and specific `sensor_name`.
        Shape: [T, 512, 9, 16]
        """
        self._open_file()
        
        # Access specific camera dataset (e.g., 'camera_F')
        if sensor_name not in self.h5_file:
            raise KeyError(f"Sensor {sensor_name} not found in HDF5 file.")

        # Read from disk
        data = self.h5_file[sensor_name][idx]
        
        return torch.from_numpy(data)

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

def quat_to_yaw_torch(qw, qx, qy, qz):
    """
    Compute yaw from quaternion (w,x,y,z). Fully vectorized.
    qw,qx,qy,qz can be any broadcastable shape.

    Returns:
        yaw: same broadcasted shape, radians in [-pi, pi]
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return torch.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi_torch(x: torch.Tensor) -> torch.Tensor:
    return (x + np.pi) % (2 * np.pi) - np.pi

def batch_hpnet_map_to_ego_per_timestep(
    batch_map_features,
    batch_ego_pose,
    inplace: bool = False,
):
    """
    Convert HPNet-style GLOBAL map features into ego-centric map features
    for EACH timestep in the history.

    This is designed to be consistent with your dataloader usage:
        batch['map_features'] is a list of dicts
        each dict has keys:
            'city', 'lane', 'centerline', ('centerline','lane'), ('lane','lane')

    It returns the same structure, but with:
        data['lane']['position']     : [H, L, 2]
        data['lane']['heading']      : [H, L]
        data['centerline']['position']: [H, N, 2]
        data['centerline']['heading'] : [H, N]

    All other fields remain unchanged:
        lane.length, lane.is_intersection, lane.turn_direction, lane.traffic_control
        centerline.length, centerline.num_nodes
        edge indices

    Args:
        batch_map_features:
            list length B of per-sample map dicts (GLOBAL coords)
        batch_ego_pose:
            torch.Tensor [B,H,7] = [x,y,z,qw,qx,qy,qz] in GLOBAL coords
            Must correspond to the same history timesteps used for agents.
        inplace:
            if True, modifies the dicts in batch_map_features directly.
            if False, returns a new list with shallow-copied dicts.

    Returns:
        batch_map_features_ego:
            list length B of per-sample dicts, now with per-timestep ego-centric
            lane/centerline positions/headings.
    """

    assert isinstance(batch_map_features, list), "batch_map_features must be a list of dicts"
    assert isinstance(batch_ego_pose, torch.Tensor), "batch_ego_pose must be a torch.Tensor"
    assert batch_ego_pose.dim() == 3 and batch_ego_pose.shape[-1] == 7, \
        "batch_ego_pose must have shape [B,H,7]"

    B, H, _ = batch_ego_pose.shape

    # yaw per timestep
    qw = batch_ego_pose[:, :, 3]
    qx = batch_ego_pose[:, :, 4]
    qy = batch_ego_pose[:, :, 5]
    qz = batch_ego_pose[:, :, 6]
    ego_yaw = quat_to_yaw_torch(qw, qx, qy, qz)  # [B,H]

    ego_xy = batch_ego_pose[:, :, :2]  # [B,H,2]

    # We will return a list of dicts, same as input
    out_list = []

    for b in range(B):
        d = batch_map_features[b]

        # -----------------------------
        # Copy dict structure if needed
        # -----------------------------
        if inplace:
            d_out = d
        else:
            # Shallow copies: we will overwrite only the tensors we transform
            d_out = dict(d)
            d_out["lane"] = dict(d["lane"])
            d_out["centerline"] = dict(d["centerline"])

            # keep tuple keys intact (same as your expected usage)
            d_out[("centerline", "lane")] = dict(d[("centerline", "lane")])
            d_out[("lane", "lane")] = dict(d[("lane", "lane")])

        # -----------------------------
        # Pull global map tensors
        # -----------------------------
        lane_pos_g = d["lane"]["position"]          # [L,2]
        lane_head_g = d["lane"]["heading"]          # [L]
        cl_pos_g = d["centerline"]["position"]      # [N,2]
        cl_head_g = d["centerline"]["heading"]      # [N]

        # Device/dtype consistency
        device = lane_pos_g.device
        dtype = lane_pos_g.dtype

        # ego pose for this sample
        ex = ego_xy[b, :, 0].to(device=device, dtype=dtype)  # [H]
        ey = ego_xy[b, :, 1].to(device=device, dtype=dtype)  # [H]
        yaw = ego_yaw[b].to(device=device, dtype=dtype)      # [H]

        c = torch.cos(yaw)  # [H]
        s = torch.sin(yaw)  # [H]

        # -----------------------------
        # Transform lane positions: [H,L,2]
        # -----------------------------
        L = lane_pos_g.shape[0]

        dx = lane_pos_g[None, :, 0] - ex[:, None]  # [H,L]
        dy = lane_pos_g[None, :, 1] - ey[:, None]  # [H,L]

        lane_x = c[:, None] * dx + s[:, None] * dy
        lane_y = -s[:, None] * dx + c[:, None] * dy
        lane_pos_ego = torch.stack([lane_x, lane_y], dim=-1)  # [H,L,2]

        # lane headings: [H,L]
        lane_head_ego = wrap_to_pi_torch(lane_head_g[None, :] - yaw[:, None])

        # -----------------------------
        # Transform centerline positions: [H,N,2]
        # -----------------------------
        N = cl_pos_g.shape[0]

        dx = cl_pos_g[None, :, 0] - ex[:, None]  # [H,N]
        dy = cl_pos_g[None, :, 1] - ey[:, None]  # [H,N]

        cl_x = c[:, None] * dx + s[:, None] * dy
        cl_y = -s[:, None] * dx + c[:, None] * dy
        cl_pos_ego = torch.stack([cl_x, cl_y], dim=-1)  # [H,N,2]

        cl_head_ego = wrap_to_pi_torch(cl_head_g[None, :] - yaw[:, None])

        # -----------------------------
        # Write back into dict
        # -----------------------------
        d_out["lane"]["position"] = lane_pos_ego
        d_out["lane"]["heading"] = lane_head_ego

        d_out["centerline"]["position"] = cl_pos_ego
        d_out["centerline"]["heading"] = cl_head_ego

        out_list.append(d_out)

    return out_list

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
    collated['ego_pose'] = torch.stack([b['ego_pose'] for b in batch])

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
    if 'map_features' in batch[0]:
        batch_map = [b["map_features"] for b in batch]        # list of dicts
        batch_ego = torch.stack([b["ego_pose"] for b in batch], dim=0)  # [B,H,7]
        collated["map_features"] = batch_hpnet_map_to_ego_per_timestep(
            batch_map_features=batch_map,
            batch_ego_pose=batch_ego,   # [B,H,7]
            inplace=True
        )

    return collated

class NuScenesDataset(Dataset):
    def __init__(self, data_pth, raw_data_dir, n_history, n_horizon, max_obstacles, max_predict, dynamic_only=False, use_camera=None, use_lidar=False, use_bev=False, use_map=False, feature_set=None, use_preprocessed=False, feature_path=None, norm_stats=None):
        '''
        data_pth: path to raw data
        data_dir: root of raw data
        n_history: length of history
        n_horizon: length of future
        max_obstacles: max number of agents in history
        max_predict: max number of agents to predict
        dynamic_only: predict only dynamic agents
        use_camera: use camera data
        use_lidar: use lidar data
        use_bev: use BEV representation of lidar data
        use_map: use map data
        feature_set: identifier for set of map features to use
        use_preprocessed: use preprocessed camera data
        feature_path: path for camera features
        norm_stats: mean, std to norm pose by
        '''

        self.use_camera_F = use_camera['F'] if use_camera is not None else False
        self.use_camera_FL = use_camera['FL'] if use_camera is not None else False
        self.use_camera_FR = use_camera['FR'] if use_camera is not None else False
        self.use_camera_B = use_camera['B'] if use_camera is not None else False
        self.use_camera_BL = use_camera['BL'] if use_camera is not None else False
        self.use_camera_BR = use_camera['BR'] if use_camera is not None else False
        self.use_lidar = use_lidar
        self.use_bev = use_bev
        self.use_map = use_map
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
            dynamic_obstacles = torch.from_numpy(np.isin(self.obs_type, DYNAMIC_TYPES)) # (n_examples, max_obstacles)
            candidate_obstacles = candidate_obstacles & dynamic_obstacles # (n_examples, max_obstacles)
        dists = torch.linalg.norm(self.obs_pose[:, -1, :, :2], dim=-1) # (n_examples, max_obstacles)
        masked_dists = dists.masked_fill(candidate_obstacles==0, float("inf")) # (n_examples, max_obstacles)
        sorted_idx = masked_dists.argsort(dim=1) # (n_examples, max_obstacles)
        self.target_idx, _ = sorted_idx[:, :max_predict].sort(dim=1) # (n_examples, max_predict) may or may not all be valid obstacles, need to refer to mask
        assert self.target_idx.shape[1] == max_predict

        # future agent mask
        self.targets_mask = data['targets_mask'][:, :n_horizon, :] # (n_examples, n_horizon, max_obstacles)
        if dynamic_only:
            self.targets_mask = self.targets_mask * dynamic_obstacles[:,None,:].numpy()
            self.obs_mask = self.obs_mask * dynamic_obstacles[:,None,:]
        self.targets_mask = np.take_along_axis(self.targets_mask, self.target_idx[:, None, :].numpy(), axis=2)
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

        # Map features
        if self.use_map:
            assert feature_set is not None, 'Must define map features to use'
            assert feature_set in ['hpnet']
            self.map_names = data['map'] # strings: (n_examples, )
            self._turn_direction_type = ['NONE', 'LEFT', 'RIGHT']
            self.map_features = self.get_map_features(feature_set)
            
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
            # ego pose for extra analysis
            "ego_pose": self.ego_pose[idx],
        }

        # Helper to handle Feature vs Raw loading
        def load_camera_data(sensor_name, file_paths):
            if self.use_preprocessed:
                # Load features from HDF5 using the sensor name as key
                # Returns: [T, 512, H, W]
                return self.feature_loader.get_features(idx, sensor_name)
            else:
                # Load raw JPGs
                # Returns: [T, 3, H, W]
                return torch.stack([self.camera_loader(f) for f in file_paths[idx]])

        # Camera Logic
        if self.use_camera_F:
            sample['camera_F'] = load_camera_data('camera_F', self.camera_F_files)
            if self.use_preprocessed: 
                sample['camera_F_features'] = sample.pop('camera_F') # Rename key if needed by collate

        if self.use_camera_FL:
            sample['camera_FL'] = load_camera_data('camera_FL', self.camera_FL_files)
            if self.use_preprocessed: 
                sample['camera_FL_features'] = sample.pop('camera_FL')

        if self.use_camera_FR:
            sample['camera_FR'] = load_camera_data('camera_FR', self.camera_FR_files)
            if self.use_preprocessed: 
                sample['camera_FR_features'] = sample.pop('camera_FR')

        if self.use_camera_B:
            sample['camera_B'] = load_camera_data('camera_B', self.camera_B_files)
            if self.use_preprocessed: 
                sample['camera_B_features'] = sample.pop('camera_B')

        if self.use_camera_BL:
            sample['camera_BL'] = load_camera_data('camera_BL', self.camera_BL_files)
            if self.use_preprocessed: 
                sample['camera_BL_features'] = sample.pop('camera_BL')

        if self.use_camera_BR:
            sample['camera_BR'] = load_camera_data('camera_BR', self.camera_BR_files)
            if self.use_preprocessed: 
                sample['camera_BR_features'] = sample.pop('camera_BR')
        
        if self.use_lidar:
            lidar_seq  = [self.lidar_loader(f).tolist() for f in self.lidar_files[idx]]
            sample.update(lidar=lidar_seq)          # list of n_history tensors
        if self.use_bev:
            sample.update(bev=self.bev[idx])        # (n_history, 4, 256, 256)
        if self.use_map:
            sample.update(map_features=self.map_features[idx])
        
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

    def get_map_features(self, feature_set):
        if feature_set == 'hpnet':
            return self.get_hpnet_map_features()
        else:
            raise NotImplementedError
        
    def lane_has_stop_line_fast(self, nusc_map, pts_t, n_check=3):
        """
        Check only the last few points of the lane centerline.
        """
        if pts_t is None or pts_t.shape[0] < 2:
            return False

        pts = pts_t.detach().cpu().numpy()
        idxs = np.linspace(max(0, pts.shape[0] - 1 - 10), pts.shape[0] - 1, n_check).astype(int)

        for j in idxs:
            x, y = float(pts[j, 0]), float(pts[j, 1])
            layers = nusc_map.layers_on_point(x, y)
            if len(layers.get("stop_line", [])) > 0:
                return True
        return False

    def get_hpnet_map_features(self, margin=50.0, save=True, pth='map_features/hpnet.pt'):
        print('Processing HPNet-style map features...')
        if os.path.isfile(pth):
            print('HPNet-style map features already exist. Loading...')
            map_features = torch.load(pth)
            return map_features

        if save:
            dr = os.path.dirname(pth)
            os.makedirs(dr, exist_ok=True)

        map_features = []
        for idx in tqdm(range(self.n_samples)):
            data = {
                'city': self.map_names[idx].split('-')[0],
                'lane': {},
                'centerline': {},
                ('centerline', 'lane'): {},
                ('lane', 'lane'): {}
            }
            nusc_map = NuScenesMap(dataroot=self.raw_data_dir, map_name=self.map_names[idx])

            # agent history info
            global_ego = self.ego_pose[idx] # H, 7
            global_agent = self.raw_obs_pose[idx] # H, A, 7
            agent_mask = self.obs_mask[idx] # H, A

            # bounding box at current timestep
            valid_positions = global_agent[-1][agent_mask[-1]>0] # <A, 7

            left_boundary = min(valid_positions[:,0])
            right_boundary = max(valid_positions[:,0])
            down_boundary = min(valid_positions[:,1])
            up_boundary = max(valid_positions[:,1])
        
            lane_tokens = nusc_map.get_records_in_radius(
                x = (left_boundary + right_boundary) / 2,
                y = (down_boundary + up_boundary) / 2,
                radius = max((right_boundary - left_boundary) / 2, (up_boundary - down_boundary) / 2) + margin,
                layer_names=['lane', 'lane_connector']
            )

            all_lane_tokens = lane_tokens['lane'] + lane_tokens['lane_connector']
            num_lanes = len(all_lane_tokens)

            # lane-level tensors
            lane_position = torch.zeros(num_lanes, 2, dtype=torch.float)
            lane_heading = torch.zeros(num_lanes, dtype=torch.float)
            lane_length = torch.zeros(num_lanes, dtype=torch.float)
            lane_is_intersection = torch.zeros(num_lanes, dtype=torch.uint8)
            lane_turn_direction = torch.zeros(num_lanes, dtype=torch.uint8)
            lane_traffic_control = torch.zeros(num_lanes, dtype=torch.uint8)

            num_centerlines = torch.zeros(num_lanes, dtype=torch.long)
            centerline_position: List[Optional[torch.Tensor]] = [None] * num_lanes
            centerline_heading: List[Optional[torch.Tensor]] = [None] * num_lanes
            centerline_length: List[Optional[torch.Tensor]] = [None] * num_lanes

            lane_token_to_idx = {t: i for i, t in enumerate(all_lane_tokens)}

            lane_adjacent_edge_index = []
            lane_predecessor_edge_index = []
            lane_successor_edge_index = []
            for i, lane_token in enumerate(all_lane_tokens):
                lane_record = nusc_map.get('lane', lane_token) \
                    if lane_token in lane_tokens['lane'] \
                    else nusc_map.get('lane_connector', lane_token)

                # centerline
                arclines = nusc_map.get_arcline_path(lane_token)

                # nuScenes returns a list of arcline dicts; typically length=1
                arc = arclines[0]
                x, y, yaw = arc["start_pose"]          # yaw is in radians in nuScenes maps
                shape = arc["shape"]                   # e.g., "LSL"
                r = float(arc["radius"])
                seg_lens = [float(v) for v in arc["segment_length"]]

                # Sampling resolution (meters). Adjust if you want denser/sparser polylines.
                ds = 1.0

                pts = [(x, y)]

                def rot2(vx, vy, a):
                    ca, sa = np.cos(a), np.sin(a)
                    return ca * vx - sa * vy, sa * vx + ca * vy

                # unit vectors given heading yaw
                def fwd(th):
                    return np.cos(th), np.sin(th)

                def left(th):
                    return -np.sin(th), np.cos(th)

                # reconstruct centerline (in Argoverese, this is given by map_api.get_lane_segment_centerline without any extra work needed)
                cur_x, cur_y, cur_yaw = x, y, yaw

                for seg_type, L in zip(shape, seg_lens):
                    if L <= 1e-6:
                        continue

                    if seg_type == "S":
                        # Straight segment
                        n = max(1, int(np.ceil(L / ds)))
                        step = L / n
                        fx, fy = fwd(cur_yaw)
                        for _ in range(n):
                            cur_x += fx * step
                            cur_y += fy * step
                            pts.append((cur_x, cur_y))

                    elif seg_type in ("L", "R"):
                        # Arc segment: angle = arc_length / radius
                        dtheta = L / r
                        sign = +1.0 if seg_type == "L" else -1.0

                        # center of rotation
                        lx, ly = left(cur_yaw)
                        if seg_type == "L":
                            cx = cur_x + r * lx
                            cy = cur_y + r * ly
                        else:
                            cx = cur_x - r * lx
                            cy = cur_y - r * ly

                        # vector from center to current position
                        vx = cur_x - cx
                        vy = cur_y - cy

                        n = max(1, int(np.ceil(abs(dtheta) * r / ds)))  # ~ arc length / ds
                        step_ang = sign * (abs(dtheta) / n)

                        for _ in range(n):
                            vx, vy = rot2(vx, vy, step_ang)
                            cur_x = cx + vx
                            cur_y = cy + vy
                            cur_yaw += step_ang
                            pts.append((cur_x, cur_y))

                    else:
                        raise ValueError(f"Unknown arcline segment type '{seg_type}' in shape='{shape}'")

                pts_t = torch.from_numpy(np.asarray(pts, dtype=np.float32)).float()

                # If we ended up with too few points, make it non-empty but safe
                if pts_t.shape[0] < 2:
                    centerline_position[i] = pts_t[:, :2]
                    centerline_heading[i] = torch.zeros(0)
                    centerline_length[i] = torch.zeros(0)
                    num_centerlines[i] = pts_t.shape[0] - 1
                    lane_position[i] = pts_t[:1, :2].mean(dim=0) if pts_t.shape[0] > 0 else torch.zeros(2)
                    lane_heading[i] = 0.0
                    lane_length[i] = 0.0
                else:
                    pts_t = pts_t[:, :2]  # [M, 2]

                    deltas = pts_t[1:] - pts_t[:-1]                 # [M-1, 2]
                    center_pos = (pts_t[1:] + pts_t[:-1]) / 2           # [M-1,2]
                    headings = torch.atan2(deltas[:, 1], deltas[:, 0])
                    seg_len = torch.norm(deltas, dim=-1)            # [M-1]

                    centerline_position[i] = center_pos
                    centerline_heading[i] = headings
                    centerline_length[i] = seg_len
                    num_centerlines[i] = pts_t.shape[0] - 1

                    center_index = int(num_centerlines[i]/2)
                    lane_position[i] = pts_t[center_index]
                    lane_heading[i] = torch.atan2(pts_t[center_index + 1, 1] - pts_t[center_index, 1], 
                                                 pts_t[center_index + 1, 0] - pts_t[center_index, 0])
                    lane_length[i] = seg_len.sum()

                    lane_is_intersection[i] = int(lane_token in lane_tokens['lane_connector'])
                    if headings.numel() < 2: # headings: [M-1] tensor for this lane (segment headings)
                        lane_turn_direction[i] = self._turn_direction_type.index('NONE')
                    else:
                        hd = headings.detach().cpu().numpy()
                        hd = np.unwrap(hd)  # remove +/-pi discontinuities
                        delta = float(hd[-1] - hd[0])
                        thresh = 0.26 # threshold in radians (~15 degrees)
                        if abs(delta) < thresh:
                            lane_turn_direction[i] = self._turn_direction_type.index('NONE')
                        elif delta > 0:
                            lane_turn_direction[i] = self._turn_direction_type.index('LEFT')
                        else:
                            lane_turn_direction[i] = self._turn_direction_type.index('RIGHT')
                    # lane traffic control is extremely slow at the moment (multiple days)
                    # if lane_token not in lane_tokens['lane_connector']:
                    #     lane_traffic_control[i] = 0
                    # else:
                    #     lane_traffic_control[i] = int(self.lane_has_stop_line_fast(nusc_map, pts_t))
                    lane_traffic_control[i] = 0

            # pack tensors
            data['lane'] = {
                'num_nodes': num_lanes,
                'position': lane_position,
                'heading': lane_heading,
                'length': lane_length,
                'is_intersection': lane_is_intersection,
                'turn_direction': lane_turn_direction,
                'traffic_control': lane_traffic_control
            }

            data['centerline'] = {
                'position': torch.cat(centerline_position, dim=0),
                'heading': torch.cat(centerline_heading, dim=0),
                'length': torch.cat(centerline_length, dim=0),
                'num_nodes': num_centerlines.sum().item()
            }

            centerline_to_lane_edge_index = torch.stack([torch.arange(num_centerlines.sum(), dtype=torch.long), torch.arange(num_lanes, dtype=torch.long).repeat_interleave(num_centerlines)], dim=0)
            data['centerline', 'lane']['centerline_to_lane_edge_index'] = centerline_to_lane_edge_index

            # not easily available in nuscenes
            data[('lane','lane')] = {
                'adjacent_edge_index': torch.empty(2, 0, dtype=torch.long),
                'predecessor_edge_index': torch.empty(2, 0, dtype=torch.long),
                'successor_edge_index': torch.empty(2, 0, dtype=torch.long),
            }

            map_features.append(data)

        if save:
            torch.save(map_features, os.path.join(pth))
                
        return map_features