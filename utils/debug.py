import matplotlib.pyplot as plt
import torch
import os

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

def plot_debug_batch(train_state, batch, device, epoch, run_name, out_slice):
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