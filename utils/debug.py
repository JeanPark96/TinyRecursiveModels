import matplotlib.pyplot as plt
import torch
import os
import random
import numpy as np

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

    B, _, A, _ = debug_batch['targets'].shape
    max_agents_to_plot = min(4, A)
    max_samples_to_plot = min(5, B)

    rdm_agents = random.sample(range(A), max_agents_to_plot)
    rdm_samples = random.sample(range(B), max_samples_to_plot)

    return debug_batch, 0, rdm_agents, rdm_samples

# plot for only one object in one sample
def plot_trajectories(hist_traj, hist_masks, pred_traj, target_traj, target_masks, obs_types, title="Trajectories", RUN_NAME="model", filename = "model_default"):
    """
    hist_traj: [History, AgentsToPlot, 2]
    pred_traj: [AgentsToPlot, Future, 2]
    target_traj: [Future, AgentsToPlot, 2]
    """
    alpha=1.0
    off='  '

    hist_masks = hist_masks > 0
    target_masks = target_masks > 0
    n_hist, n_agents, _ = hist_traj.shape

    hist_traj = hist_traj.detach().cpu()
    pred_traj = pred_traj.detach().cpu()
    target_traj = target_traj.detach().cpu()

    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")

    # Plot ego
    plt.plot(0, 0, 'k*', alpha=alpha, label='ego')
    
    for j, a in enumerate(range(n_agents)):
        obs_type = obs_types[a]
        hist_mask = hist_masks[:,a]
        target_mask = target_masks[:,a]

        if j > 0:
            label=None

        # Plot past trajectory with index labels
        if j == 0:
            label='History'
        plt.plot(hist_traj[hist_mask, a, 0], hist_traj[hist_mask, a, 1], 'b.-', alpha=alpha, label=label)
        
        for i, (x, y) in enumerate(hist_traj[hist_mask, a]):
            lbl = off+str(-n_hist+i)
            if i == 0: lbl += f' ({obs_type})'
            plt.text(x, y, lbl, fontsize=9, color='black', alpha=alpha)

        # Plot predicted trajectory with index labels
        if j == 0:
            label='Predicted'
        plt.plot(pred_traj[a, :, 0], pred_traj[a, :, 1], 'r.--', label=label, alpha=alpha)
        
        for i, (x, y) in enumerate(pred_traj[a]):
            plt.text(x, y, off+str(i), fontsize=9, color='black', alpha=alpha)

        # Plot ground truth trajectory with index labels
        if j == 0:
            label='Ground Truth'
        plt.plot(target_traj[target_mask, a, 0], target_traj[target_mask, a, 1], 'g.--', label=label, alpha=alpha)

        for i, (x, y) in enumerate(target_traj[target_mask, a]):
            plt.text(x, y, off+str(i), fontsize=9, color='black', alpha=alpha)

        # Plot lines between history and future
        # mask between last history and first future guaranteed to be true (otherwise, sample would not exist)
        plt.plot([hist_traj[-1, a, 0], pred_traj[a, 0, 0]], # x
                [hist_traj[-1, a, 1], pred_traj[a, 0, 1]], # y
                'r--', alpha=alpha)
        plt.plot([hist_traj[-1, a, 0], target_traj[0, a, 0]], # x
                [hist_traj[-1, a, 1], target_traj[0, a, 1]], # y
                'g--', alpha=alpha)

    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    save_path = f'plot_figures/{RUN_NAME}/{filename}.png'

    os.makedirs(f"plot_figures/{RUN_NAME}", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    #plt.show()

def plot_debug_batch(train_state, dataset, batch, rdm_agents, rdm_samples, device, epoch, run_name, out_slice, together=True):
    """
    Run the model on a fixed batch and plot the first few agents' trajectories.
    Called before training/resume (with the current model state) and after each epoch.

    together: plot all agents on top of each other if true
    """
    train_state.model.eval()
    with torch.no_grad():
        obs_pose = batch["obs_pose"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        targets = batch["targets"].to(device)
        targets_mask = batch.get("targets_mask", None)
        if targets_mask is None:
            targets_mask = (targets[..., :2].abs().sum(dim=-1) > 1e-3).to(obs_pose.dtype)
        else:
            targets_mask = targets_mask.to(device)
        sample_idx = batch["idx"]                            # [B]
        obs_types = dataset.get_obs_type(sample_idx)      # [B, A]    

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

        pred = outputs["pred"]
        
        if together:
            for s in rdm_samples:
                hist_xy = obs_pose[s, :, rdm_agents, :2].detach().cpu()
                hist_mask_cpu = obs_mask[s, :, rdm_agents].detach().cpu()
                gt_xy = targets[s, :, rdm_agents, :2].detach().cpu()
                targets_mask_cpu = targets_mask[s, :, rdm_agents].detach().cpu()
                pred_xy = pred[s, rdm_agents, :, :2].detach().cpu()
                types = obs_types[s, rdm_agents]

                plot_trajectories(
                    hist_xy,
                    hist_mask_cpu,
                    pred_xy,
                    gt_xy,
                    targets_mask_cpu,
                    obs_types=types,
                    title=f"[Debug] epoch {epoch} sample {s}",
                    RUN_NAME=run_name,
                    filename=f"{run_name}_debug_epoch{epoch}_sample{s}",
                )
        else:
            for s in rdm_samples:
                for a in rdm_agents:
                    hist_xy = obs_pose[s, :, a, :2].detach().cpu().unsqueeze(1)
                    hist_mask_cpu = obs_mask[s, :, a].detach().cpu().unsqueeze(1)
                    gt_xy = targets[s, :, a, :2].detach().cpu().unsqueeze(1)
                    targets_mask_cpu = targets_mask[s, :, a].detach().cpu().unsqueeze(1)
                    pred_xy = pred[s, a, :, :2].detach().cpu().unsqueeze(0)
                    types = np.expand_dims(np.array(obs_types[s, a]), 0)

                    plot_trajectories(
                        hist_xy,
                        hist_mask_cpu,
                        pred_xy,
                        gt_xy,
                        targets_mask_cpu,
                        obs_types=types,
                        title=f"[Debug] agent {a} @ epoch {epoch} sample {s}",
                        RUN_NAME=run_name,
                        filename=f"{run_name}_debug_epoch{epoch}_sample{s}_agent{a}",
                    )