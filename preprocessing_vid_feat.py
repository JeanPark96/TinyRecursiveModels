import os
from scipy import stats
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import h5py
from tqdm import tqdm
import sys
from nuscenes_dataset import NuScenesDataset
import argparse

# Add project root to path so we can import dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# def extract_to_hdf5(dataset, output_path, batch_size=16, feat_h=288, feat_w=512, device='cuda'):
#     """
#     Extracts ResNet features and stores them in a single HDF5 file.
#     Structure: Dataset 'features' with shape [N_samples, T, 512, 9, 16]
#     """
#     #os.makedirs(output_path, exist_ok=True)

#     # 1. Setup Feature Extractor (ResNet18)
#     resnet = models.resnet18(pretrained=True)
#     # Remove last 2 layers (AvgPool, FC) -> Output stride 32
#     backbone = nn.Sequential(*list(resnet.children())[:-2])
#     device = torch.device(f"cuda:2")
#     torch.cuda.set_device(2)
#     backbone.to(device)
#     backbone.eval()
    
#     # 2. Define Transform (Resize to 512x288 for 16:9 aspect ratio)
#     preprocess = transforms.Compose([
#         transforms.Resize((feat_h, feat_w)),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     # 3. Setup DataLoader
#     # Note: We rely on the raw images here, so ensure dataset returns them
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
#     # 4. Initialize HDF5 File
#     # We need to know the total size first.
#     N_samples = len(dataset)
#     # Assuming T=10 history logic from your config, check dataset.n_history if available
#     # Or just inspect the first batch.
#     print("Checking data shape...")
#     first_batch = next(iter(loader))
#     # camera shape: [B, T, C, H, W]
#     _, T, _, _, _ = first_batch['camera'].shape
    
#     # Feature map shape: 512 channels, 9 height, 16 width
#     feature_shape = (N_samples, T, 512, feat_h // 32, feat_w // 32)
#     print(f"Creating HDF5 container with shape: {feature_shape}")
    
#     with h5py.File(output_path, 'w') as f:
#         # utilize float16 ('f2') to save 50% disk space (ResNet features are robust to fp16)
#         dset = f.create_dataset('features', shape=feature_shape, dtype='float16', chunks=(1, T, 512, feat_h // 32, feat_w // 32))
        
#         print("Starting extraction...")
#         start_idx = 0
        
#         with torch.no_grad():
#             for batch in tqdm(loader):
#                 # Get images: [B, T, C, H, W]
#                 # Assuming dataset returns 0-1 tensors. If 0-255 uint8, divide by 255 first.
#                 imgs = batch['camera'].to(device)
#                 B = imgs.shape[0]
                
#                 # Flatten T: [B*T, C, H, W]
#                 imgs_flat = imgs.view(B * T, 3, imgs.shape[-2], imgs.shape[-1])
                
#                 # Transform (Resize/Norm)
#                 imgs_ready = preprocess(imgs_flat)
                
#                 # Extract: [B*T, 512, 9, 16]
#                 feats = backbone(imgs_ready)
#                 # Reshape back: [B, T, 512, feat_height, feat_width]
#                 feats = feats.view(B, T, 512, feat_h // 32, feat_w // 32)
                
#                 # Write to HDF5 (CPU numpy)
#                 end_idx = start_idx + B
#                 dset[start_idx:end_idx] = feats.cpu().numpy().astype('float16')
#                 start_idx = end_idx

#     print(f"Saved to {output_path}")
class HDF5FeatureLoader:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file = None
        # We don't pre-load datasets here to avoid pickling issues with DataLoader workers

    def _open_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

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
        data = self.h5_file[sensor_name][idx].astype(np.float32)
        
        return torch.from_numpy(data)

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

def extract_to_hdf5(dataset, output_path, batch_size=16, feat_h=288, feat_w=512, device='cuda'):
    """
    Extracts ResNet features for all enabled cameras and stores them in an HDF5 file.
    Structure: 
        file['camera_F'] -> [N_samples, T, 512, 9, 16]
        file['camera_FL'] -> [N_samples, T, 512, 9, 16]
        ...
    """
    # 1. Setup Feature Extractor (ResNet18)
    resnet = models.resnet18(pretrained=True)
    # Remove last 2 layers (AvgPool, FC) -> Output stride 32
    backbone = nn.Sequential(*list(resnet.children())[:-2])
    
    # Handle device selection more robustly
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    backbone.eval()
    
    # 2. Define Transform (Resize to 512x288 for 16:9 aspect ratio)
    preprocess = transforms.Compose([
        transforms.Resize((feat_h, feat_w)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Setup DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    # 4. Initialize HDF5 File
    N_samples = len(dataset)
    print("Checking data shape...")
    first_batch = next(iter(loader))
    
    # Identify which cameras are present in the batch
    camera_keys = [key for key in first_batch.keys() if 'camera_' in key and 'features' not in key]
    print(f"Found cameras: {camera_keys}")

    # Get T from the first available camera
    _, T, _, _, _ = first_batch[camera_keys[0]].shape
    print("timesteps:", T)
    # Feature map shape: 512 channels, 9 height, 16 width
    feature_shape = (N_samples, T, 512, feat_h // 32, feat_w // 32)
    chunk_shape = (1, T, 512, feat_h // 32, feat_w // 32)
    
    print(f"Creating HDF5 container with shape per camera: {feature_shape}")
    
    with h5py.File(output_path, 'w') as f:
        # Create datasets for each camera
        dsets = {}
        for cam_name in camera_keys:
            dsets[cam_name] = f.create_dataset(
                cam_name, 
                shape=feature_shape, 
                dtype='float16', 
                chunks=chunk_shape
            )
        
        print("Starting extraction...")
        start_idx = 0
        
        with torch.no_grad():
            for batch in tqdm(loader):
                # Calculate Batch Size from the first camera key
                B = batch[camera_keys[0]].shape[0]
                end_idx = start_idx + B

                for cam_name in camera_keys:
                    # Get images: [B, T, C, H, W]
                    imgs = batch[cam_name].to(device)
                    
                    # Flatten T: [B*T, C, H, W]
                    imgs_flat = imgs.view(B * T, 3, imgs.shape[-2], imgs.shape[-1])
                    
                    # Transform (Resize/Norm)
                    imgs_ready = preprocess(imgs_flat)
                    
                    # Extract: [B*T, 512, 9, 16]
                    feats = backbone(imgs_ready)
                    
                    # Reshape back: [B, T, 512, feat_height, feat_width]
                    feats = feats.view(B, T, 512, feat_h // 32, feat_w // 32)
                    
                    # Write to specific HDF5 dataset
                    dsets[cam_name][start_idx:end_idx] = feats.cpu().numpy().astype('float16')
                
                start_idx = end_idx

    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_type', type=str, default='standard', help='Dataset split methods. OOD splits of type oodType are named with convention oodType-oodSubType, where oodSubType does not appear in the ID train/val/test distribution.',
                                    choices=['standard',
                                             'city-boston', 'city-singapore',
                                             'map-boston-seaport', 'map-singapore-onenorth', 'map-singapore-queensto', 'map-singapore-hollandv',
                                             'object-animal', 'object-child', 'object-construction_worker', 'object-personal_mobility', 'object-police_officer', 'object-stroller', 'object-wheelchair',
                                             'object-debris', 'object-bicycle_rack',
                                             'object-bendy', 'object-ambulance', 'object-police'],)
    args = parser.parse_args()
    # Initialize your dataset in 'raw' mode (loading jpgs)
    # Update paths to your real data
    SAMPLE_FREQ = 2
    max_obstacles = 30#30
    n_history = 2*SAMPLE_FREQ # current time inclusive
    n_horizon = 6*SAMPLE_FREQ
    # Optional CUDA debug envs (you can comment these out if you don't want sync execution)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    src_root = "/home/vilin/Rapid_Adapt_SM/src/data"
    dst_root = "/home/vilin/Rapid_Adapt_SM/src/data"

    camera = {'F':True, 'FL':True, 'FR':True, 'B':True, 'BL':True, 'BR':True}

    for split in ['standard', 'city-boston', 'city-singapore','map-boston-seaport', 'map-singapore-onenorth', 'map-singapore-queensto', 'map-singapore-hollandv',
                    'object-animal', 'object-child', 'object-construction_worker', 'object-personal_mobility', 'object-police_officer', 'object-stroller', 'object-wheelchair',
                    'object-debris', 'object-bicycle_rack', 'object-bendy', 'object-ambulance', 'object-police']:
        for ts in ["train", "val", "test", "ood"]:
            if 'standard' in split and ts == 'ood':
                continue   
            if os.path.exists(f"{dst_root}/cam-{split}_resnet_feat18/all_camera_features_{ts}.h5"):
                print(f"Features for {split} {ts} already exist, skipping...")
                continue     
            dataset = NuScenesDataset(
                data_pth=f'{dst_root}/cam-{split}/{ts}.npz', 
                raw_data_dir="/home/vilin/Rapid_Adapt_SM/raw_data/nuscenes", 
                # raw_data_dir=f"{src_root}/cam-{split}", 
                max_obstacles=max_obstacles, n_history=n_history, n_horizon=n_horizon, use_camera=camera, use_lidar=False, use_bev=False, use_preprocessed=False, feature_path=None, norm_stats=None)
            os.makedirs(f"{dst_root}/cam-{split}_resnet_feat18", exist_ok=True)
            extract_to_hdf5(dataset, f"{dst_root}/cam-{split}_resnet_feat18/all_camera_features_{ts}.h5")