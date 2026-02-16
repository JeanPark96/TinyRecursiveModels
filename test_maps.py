from nuscenes_dataset import NuScenesDataset, custom_collate
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch


import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_hpnet_nusc_features(dataset, batch, idx=0, title=None, t=3, pth='test_map_feats.png'):
    """
    dataset: your dataset object (needed for agent/ego positions)
    map_features: output list from get_hpnet_map_features()
    idx: which sample to plot
    """

    data = batch['map_features'][idx]

    # -----------------------------
    # Pull out map tensors
    # -----------------------------
    lane_pos = data["lane"]["position"][t]              # [H,L,2]
    lane_heading = data["lane"]["heading"][t]           # [H,L]
    lane_is_int = data["lane"]["is_intersection"]       # [L]
    lane_turn = data["lane"]["turn_direction"]          # [L] 0/1/2
    lane_tc = data["lane"]["traffic_control"]           # [L]

    cl_pos = data["centerline"]["position"][t]          # [H,N,2] (segment midpoints!)
    cl_heading = data["centerline"]["heading"][t]       # [H,N]
    cl_len = data["centerline"]["length"]               # [N]

    # edge index: [2, N]
    e = data[("centerline", "lane")]["centerline_to_lane_edge_index"]
    cl_to_lane = e[1]  # [N] lane index for each centerline segment

    # -----------------------------
    # Pull out agent positions
    # -----------------------------
    global_agent = batch['obs_pose'][idx]       # [H,A,7]
    agent_mask = batch['obs_mask'][idx]         # [H,A]

    agents_xy = global_agent[t, :, :2].detach().cpu().numpy()
    valid = agent_mask[t] > 0
    agents_xy = agents_xy[valid.detach().cpu().numpy()]

    # -----------------------------
    # Move tensors to numpy
    # -----------------------------
    lane_pos_np = lane_pos.detach().cpu().numpy()
    cl_pos_np = cl_pos.detach().cpu().numpy()
    cl_to_lane_np = cl_to_lane.detach().cpu().numpy()

    lane_is_int_np = lane_is_int.detach().cpu().numpy()
    lane_tc_np = lane_tc.detach().cpu().numpy()

    # -----------------------------
    # Plot
    # -----------------------------
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Plot centerline segment midpoints.
    # We'll color by: lane_connector (intersection) vs lane.
    is_int_for_cl = lane_is_int_np[cl_to_lane_np]  # [N]

    # non-connector
    mask0 = (is_int_for_cl == 0)
    ax.scatter(
        -cl_pos_np[mask0, 1],
        cl_pos_np[mask0, 0],
        s=2,
        alpha=0.35,
        label="centerline segs (lane)",
    )

    # connector
    mask1 = (is_int_for_cl > 0)
    ax.scatter(
        -cl_pos_np[mask1, 1],
        cl_pos_np[mask1, 0],
        s=6,
        alpha=0.7,
        label="centerline segs (lane_connector)",
    )

    # Plot lane node positions
    ax.scatter(
        -lane_pos_np[:, 1],
        lane_pos_np[:, 0],
        s=40,
        marker="x",
        label="lane nodes",
    )

    # Highlight lanes with traffic control
    tc_mask = lane_tc_np > 0
    if tc_mask.any():
        ax.scatter(
            -lane_pos_np[tc_mask, 1],
            lane_pos_np[tc_mask, 0],
            s=80,
            marker="o",
            facecolors="none",
            linewidths=2,
            label="traffic_control lanes",
        )

    # Plot agents
    ax.scatter(
        -agents_xy[:, 1],
        agents_xy[:, 0],
        s=30,
        label="agents(t)",
    )

    # Plot ego
    ax.scatter(
        [0],
        [0],
        s=120,
        marker="*",
        label="ego(t)",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y")
    ax.set_ylabel("x")

    if title is None:
        title = f"nuScenes HPNet map features | idx={idx} | lanes={lane_pos.shape[0]} | cl_segs={cl_pos.shape[0]}"
    ax.set_title(title)

    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    plt.savefig(pth)
    plt.close()

train_data_pth = f'/home/vilin/Rapid_Adapt_SM/src/data/loc-cam-standard/train.npz'
raw_data_dir = '/home/vilin/Rapid_Adapt_SM/raw_data/nuscenes'

print(f'Loading train dataset...')
tr_dataset = NuScenesDataset(train_data_pth, raw_data_dir, 4, 12, 30, 8, use_map=True, feature_set='hpnet')
tr_dataloader = DataLoader(tr_dataset, batch_size=20, shuffle=True, collate_fn=custom_collate)

batch = next(iter(tr_dataloader))
print(batch.keys())

print('items in batch:', len(batch['map_features'])) # list of dicts

sample_num = 0
print(f'\nExample map features at index {sample_num}...')
print(batch['map_features'][sample_num].keys())
print('\tcity:', batch['map_features'][sample_num]['city'])
print('\tlane dict:', batch['map_features'][sample_num]['lane'].keys())
print('\tlane position:', batch['map_features'][sample_num]['lane']['position'].shape)
print('\tcenterline dict:', batch['map_features'][sample_num]['centerline'].keys())
print('\t(centerline, lane) dict:', batch['map_features'][sample_num][('centerline', 'lane')].keys())
print('\t(lane, lane) dict:', batch['map_features'][sample_num][('lane', 'lane')].keys())
print("\tnum_centerline_nodes:", batch['map_features'][sample_num]["centerline"]["num_nodes"])

for s in range(20):
    plot_hpnet_nusc_features(tr_dataset, batch, s, title=f"Sample {s}", pth=f'map_features/{s}_map_feats.png')

    idx = batch['idx']
    obs_mask = batch['obs_mask']
    obs_pose = batch['obs_pose']
    t = 3
    obs_types = tr_dataset.get_obs_type(idx[s])
    color_list = ['b','g','r','c','m','y','tab:orange','tab:brown','tab:gray']
    color_map = {}
    for i in range(30):
        if obs_mask[s,t,i] == 1:# and obs_pose[s,-1,i,0]>0:
            if obs_types[i] in color_map:
                plt.scatter(-obs_pose[s,t,i,1], obs_pose[s,t,i,0], color=f'{color_map[obs_types[i]]}', marker='.')
            else:
                plt.scatter(-obs_pose[s,t,i,1], obs_pose[s,t,i,0], color=f'{color_list[0]}', marker='.', label=obs_types[i])
                color_map[obs_types[i]] = color_list[0]
                color_list = color_list[1:]
    plt.xlabel('Right Direction (-y)')
    plt.ylabel('Forward Direction (+x)')
    plt.scatter(0,0, color='k', marker='*', label='ego')
    plt.legend()
    plt.savefig(f'map_features/{s}_positions.png')
    plt.close()