import os
import torch
import numpy as np
from tqdm import tqdm
from nuscenes_dataset import NuScenesDataset


def export_to_social_lstm_format(nuscenes_dataset, output_dir="data/social_lstm_input"):
    """
    Exports NuScenesDataset samples to individual text files compatible with Social LSTM.
    
    Format required by Social LSTM utils.py:
    [frame_num] [ped_id] [y] [x]
    
    Args:
        nuscenes_dataset: Your instantiated NuScenesDataset
        output_dir: Directory to save the .txt files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Exporting {len(nuscenes_dataset)} samples to {output_dir}...")

    # Iterate through every sample in your dataset
    for idx in tqdm(range(len(nuscenes_dataset))):
        data = nuscenes_dataset[idx]
        
        # 1. Get raw coordinates (Metric space)
        # Concatenate History (obs) and Future (targets)
        # obs_pose shape: (History, Max_Agents, 7)
        # targets shape: (Horizon, Max_Agents, 7)
        history_pose = data['org_obs_pose'] 
        future_pose = data['org_targets']
        
        # Combine to get full trajectory: (Total_Len, Max_Agents, 7)
        full_pose = torch.cat((history_pose, future_pose), dim=0)
        
        # 2. Get Masks to find valid agents
        history_mask = data['obs_mask']
        future_mask = data['targets_mask']
        full_mask = torch.cat((history_mask, future_mask), dim=0)
        
        total_frames, max_agents, _ = full_pose.shape
        lines = []

        # 3. Iterate over each agent
        for agent_idx in range(max_agents):
            # Skip agents that don't appear at all in this sequence
            if full_mask[:, agent_idx].sum() == 0:
                continue
            
            # 4. Iterate over each frame for this agent
            for t in range(total_frames):
                # Only write data if the agent is valid (visible) in this frame
                if full_mask[t, agent_idx] == 1:
                    # Extract x and y (Indices 0 and 1 in your dataset)
                    # Note: Your dataset seems to use [x, y, z...].
                    # Social LSTM utils.py reads column 2 as 'y' and 3 as 'x'.
                    # So we write: frame_num, agent_id, y, x
                    
                    x_val = full_pose[t, agent_idx, 0].item()
                    y_val = full_pose[t, agent_idx, 1].item()
                    
                    # Create the line string
                    # Format: frame_num ped_id y x
                    line = f"{t} {agent_idx} {y_val:.4f} {x_val:.4f}"
                    lines.append(line)

        # 5. Save to a text file
        # We save one file per sample. The Social LSTM dataloader will treat 
        # each file as a dataset containing one sequence.
        if len(lines) > 0:
            file_path = os.path.join(output_dir, f"seq_{idx:06d}.txt")
            with open(file_path, "w") as f:
                f.write("\n".join(lines))

    print("Export complete.")

if __name__ == "__main__":
    SAMPLE_FREQ = 2
    max_obstacles = 30#30
    n_history = 2*SAMPLE_FREQ # current time inclusive
    n_horizon = 6*SAMPLE_FREQ
    #src_root = "/home/vilin/Rapid_Adapt_SM/src/data"
    src_root = "/home/hlpark/common-data/trm/data"
    #dst_root = "/home/hlpark/common-data/trm/data/social_lstm_xy"
    dst_root = "/home/vilin/Rapid_Adapt_SM/src/data/social_lstm_xy"

    camera = {'F':False, 'FL':False, 'FR':False, 'B':False, 'BL':False, 'BR':False}

    for split in ['standard', 'city-boston', 'city-singapore','map-boston-seaport', 'map-singapore-onenorth', 'map-singapore-queensto', 'map-singapore-hollandv',
                    'object-animal', 'object-child', 'object-construction_worker', 'object-personal_mobility', 'object-police_officer', 'object-stroller', 'object-wheelchair',
                    'object-debris', 'object-bicycle_rack', 'object-bendy', 'object-ambulance', 'object-police']:
        for ts in ["train", "val", "test", "ood"]:
            if 'standard' in split and ts == 'ood':
                continue   
            if os.path.exists(f"{dst_root}/{split}/{ts}"):
                print(f"Features for {split} {ts} already exist, skipping...")
                continue     
            dataset = NuScenesDataset(
                data_pth=f'{src_root}/{split}/{ts}.npz', 
                raw_data_dir="/home/vilin/Rapid_Adapt_SM/raw_data/nuscenes", 
                # raw_data_dir=f"{src_root}/cam-{split}", 
                max_obstacles=max_obstacles, n_history=n_history, n_horizon=n_horizon, use_camera=camera, use_lidar=False, use_bev=False, use_preprocessed=False, feature_path=None, norm_stats=None)
            
            # dataset = NuScenesDataset(
            #     data_pth=f'{dst_root}/{split}/{ts}.npz',
            #     raw_data_dir="/home/vilin/Rapid_Adapt_SM/raw_data/nuscenes",
            #     n_history=n_history,  # Example
            #     n_horizon=n_horizon, # Example
            #     max_obstacles=max_obstacles
            # )

            export_to_social_lstm_format(dataset, output_dir=f"{dst_root}/{split}/{ts}")