import os
import torch
import numpy as np
from tqdm import tqdm
from nuscenes_dataset import NuScenesDataset

def export_valid_ids_and_trajs(dataset, output_root):
    """
    Exports trajectories to .txt files compatible with Social LSTM.
    Handles the dimension mismatch between History (All Neighbors) and Future (Targets Only).
    
    Output Format: seq_XXXXXX.txt
    Row Format: [frame_num] [ped_id] [y] [x]
    """
    # Create 'feat' directory inside the split folder
    feat_dir = os.path.join(output_root, "feat")
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)
    
    print(f"Exporting {len(dataset)} samples to {feat_dir}...")

    for idx in tqdm(range(len(dataset))):
        data = dataset[idx]
        
        # 1. Unpack Data
        # Shape: [History, 30, 7]
        hist_pose = data['org_obs_pose']  
        #print(hist_pose[:,19,:2])

        # Shape: [Horizon, 8, 7]
        fut_pose = data['org_targets']    
        # Shape: [8] (The original IDs of the targets)
        target_idxs = data['targets_idx'] 
        
        # Masks
        hist_mask = data['obs_mask']      # [History, 30]
        fut_mask = data['targets_mask']   # [Horizon, 8]
        #print(hist_mask[:,19])
        
        n_hist = hist_pose.shape[0]
        n_fut = fut_pose.shape[0]
        n_total = n_hist + n_fut
        max_agents = hist_pose.shape[1] # Should be 30
        
        # 2. Reconstruct Full Matrix [Total_Len, 30, 7]
        # We create a container for all 30 agents over the full timeline
        full_pose = torch.zeros((n_total, max_agents, 7))
        full_mask = torch.zeros((n_total, max_agents))
        
        # A. Fill History (0 to n_hist) for ALL 30 agents
        full_pose[:n_hist, :, :] = hist_pose
        full_mask[:n_hist, :] = hist_mask
        
        # B. Fill Future (n_hist to end) ONLY for the 8 Targets
        # We must map the 8 targets back to their original slots using target_idxs
        for i, real_id in enumerate(target_idxs):
            real_id = real_id.item()
            # Place the i-th target's future into the real_id slot
            full_pose[n_hist:, real_id, :] = fut_pose[:, i, :]
            full_mask[n_hist:, real_id] = fut_mask[:, i]

        # 3. Write to String
        lines = []
        
        # Iterate over all 30 potential agents
        for agent_id in range(max_agents):
            # If agent never appears, skip
            if full_mask[:, agent_id].sum() == 0:
                continue
                
            for t in range(n_total):
                
                if full_mask[t, agent_id] == 1:
                    # Extract X and Y (Assuming index 0=x, 1=y in your 7-dim vector)
                    # Note: You requested [y] [x] order
                    x_val = full_pose[t, agent_id, 0].item()
                    y_val = full_pose[t, agent_id, 1].item()
                    
                    # if agent_id == 19:
                    #     print(f"Agent 19 at time {t}: x={x_val}, y={y_val}, mask={full_mask[t, agent_id].item()}")
                    
                    # Format: frame_num ped_id y x
                    line = f"{t} {agent_id} {y_val:.4f} {x_val:.4f}"
                    lines.append(line)
        # 4. Save File
        if len(lines) > 0:
            file_path = os.path.join(feat_dir, f"seq_{idx:06d}.txt")
            with open(file_path, "w") as f:
                f.write("\n".join(lines))

    print("Export complete.")


if __name__ == "__main__":
    # --- Configuration ---
    SAMPLE_FREQ = 2
    max_obstacles = 30
    max_predict = 8 # Ensure this matches your Dataset default
    n_history = 2 * SAMPLE_FREQ 
    n_horizon = 6 * SAMPLE_FREQ
    
    src_root = "/home/hlpark/common-data/trm/data"
    dst_root = "/home/vilin/Rapid_Adapt_SM/src/data/social_lstm_xy_8pred"

    camera = {'F':False, 'FL':False, 'FR':False, 'B':False, 'BL':False, 'BR':False}

    # --- Loop Over Splits ---
    # split_list = ['standard'] 
    #split_list = ['standard', 'city-boston', 'city-singapore'] # Add others as needed

    for split in ['standard', 'city-boston', 'city-singapore','map-boston-seaport', 'map-singapore-onenorth', 'map-singapore-queensto', 'map-singapore-hollandv',
                    'object-animal', 'object-child', 'object-construction_worker', 'object-personal_mobility', 'object-police_officer', 'object-stroller', 'object-wheelchair',
                    'object-debris', 'object-bicycle_rack', 'object-bendy', 'object-ambulance', 'object-police']:
        for ts in ["train", "val", "test", "ood"]:
            if 'standard' in split and ts == 'ood':
                continue   
            if os.path.exists(f"{dst_root}/{split}/{ts}"):
                print(f"Features for {split} {ts} already exist, skipping...")
                continue  
                
            # Define Input/Output Paths
            input_npz = f'{src_root}/{split}/{ts}.npz'
            output_dir = f"{dst_root}/{split}/{ts}"
            
            if not os.path.exists(input_npz):
                print(f"Skipping {input_npz} (Not found)")
                continue

            print(f"Processing: {split}/{ts}...")
            
            # Initialize Dataset
            # Note: We pass max_predict=8 here to match your logic
            dataset = NuScenesDataset(
                data_pth=input_npz,
                raw_data_dir="/home/vilin/Rapid_Adapt_SM/raw_data/nuscenes", 
                n_history=n_history, 
                n_horizon=n_horizon, 
                max_obstacles=max_obstacles, 
                max_predict=max_predict, # <--- IMPORTANT
                dynamic_only=False, 
                use_camera=camera, 
                use_lidar=False, 
                use_bev=False, 
                use_map=False, 
                norm_stats=None
            )
            
            # Run Export
            export_valid_ids_and_trajs(dataset, output_root=output_dir)
            