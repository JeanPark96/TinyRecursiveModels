import matplotlib.pyplot as plt
import torch
import os
import random

# plot for only one object in one sample
def plot_trajectories(hist_traj, hist_mask, pred_traj, target_traj, target_mask, obs_type, title="Trajectories", RUN_NAME="model", filename = "model_default"):
    """
    hist_traj: [History, 2]
    pred_traj: [Future, 2]
    target_traj: [Future, 2]
    """
    alpha=1.0

    hist_mask = hist_mask > 0
    target_mask = target_mask > 0
    n_hist, _ = hist_traj.shape

    hist_traj = hist_traj.detach().cpu()
    pred_traj = pred_traj.detach().cpu()
    target_traj = target_traj.detach().cpu()

    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")

    # Plot past trajectory with index labels
    plt.plot(hist_traj[hist_mask, 0], hist_traj[hist_mask, 1], 'go-', alpha=alpha)
    
    for i, (x, y) in enumerate(hist_traj[hist_mask]):
        plt.text(x, y, '  '+str(-n_hist+i), fontsize=9, color='black', alpha=alpha)

    # Plot predicted trajectory with index labels
    plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'ro--', label=f"Predicted ({obs_type})", alpha=alpha)
    
    for i, (x, y) in enumerate(pred_traj):
        plt.text(x, y, '  '+str(i), fontsize=9, color='black', alpha=alpha)

    # Plot ground truth trajectory with index labels
    plt.plot(target_traj[target_mask, 0], target_traj[target_mask, 1], 'go--', label=f"Ground Truth ({obs_type})", alpha=alpha)

    for i, (x, y) in enumerate(target_traj[target_mask]):
        plt.text(x, y, '  '+str(i), fontsize=9, color='black', alpha=alpha)

    # Plot lines between history and future
    # mask between last history and first future guaranteed to be true (otherwise, sample would not exist)
    plt.plot([hist_traj[-1, 0], pred_traj[0,0]], # x
             [hist_traj[-1, 1], pred_traj[0,1]], # y
             'r--', alpha=alpha)
    plt.plot([hist_traj[-1, 0], target_traj[0,0]], # x
             [hist_traj[-1, 1], target_traj[0,1]], # y
             'g--', alpha=alpha)

    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    save_path = f'plot_figures/{RUN_NAME}/{filename}.png'
   
    os.makedirs(f"plot_figures/{RUN_NAME}", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    #plt.show()

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

def plot_debug_batch(train_state, dataset, batch, device, epoch, run_name, out_slice):
    """
    Run the model on a fixed batch and plot the first few agents' trajectories.
    Called before training/resume (with the current model state) and after each epoch.
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

        B, A, H, _ = pred.shape
        max_agents_to_plot = min(4, A)
        max_samples_to_plot = min(5, B)

        rdm_agents = random.sample(range(A), max_agents_to_plot)
        rdm_samples = random.sample(range(B), max_samples_to_plot)
        
        for s in rdm_samples:
            for i in rdm_agents:
                hist_xy = obs_pose[s, :, i, :2].detach().cpu()
                hist_mask_cpu = obs_mask[s, :, i].detach().cpu()
                gt_xy = targets[s, :, i, :2].detach().cpu()
                targets_mask_cpu = targets_mask[s, :, i].detach().cpu()
                pred_xy = pred[s, i, :, :2].detach().cpu()
                obs_type = obs_types[s, i]

                plot_trajectories(
                    hist_xy,
                    hist_mask_cpu,
                    pred_xy,
                    gt_xy,
                    targets_mask_cpu,
                    obs_type=obs_type,
                    title=f"[Debug] Agent {i} @ epoch {epoch} sample {s}",
                    RUN_NAME=run_name,
                    filename=f"{run_name}_debug_epoch{epoch}_sample{s}_agent{i}",
                )