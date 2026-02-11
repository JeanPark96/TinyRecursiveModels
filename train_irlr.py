"""
Training script for IRLR + SnAG PyramidBoundaryPredictor
on Charades-STA with I3D features and GloVe text tokenization.

Usage:
    python train_irlr.py --run_name irlr_exp1 --epochs 30 --lr 1e-4
"""

import os
import sys
import argparse
import random
import math
import json
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Dataset
from dataset.build_charades_sta_snag import load_dataset

# Loss functions
from models.act_losses import compute_dense_tr_loss

# IRLR + SnAG model
from models.irlr import IRLRConfig, IRLRWithSnAG, make_gt_mask, compute_mask_loss

# Metrics
from utils.metric import (
    compute_localization_metric,
    compute_retrieval_metrics,
    decode_dense_candidates,
)

# Logger
from utils.log import Logger


# =============================================================================
# Learning rate schedule
# =============================================================================

def cosine_schedule_with_warmup(current_step, base_lr, num_warmup_steps, num_training_steps, min_ratio=0.01):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))


# =============================================================================
# Single training step
# =============================================================================

def train_step(model, batch, optimizer, cfg, device, global_step,
               lr_schedule=False, base_lr=1e-4, warmup_steps=500, total_steps=10000):
    model.train()

    # ---- Move to device ----
    video_emb = batch["video_emb"].to(device)           # (B, T, 2048)
    query_tokens = batch["query_tokens"].to(device)     # (B, L, 300)
    text_emb = batch["text_emb"].to(device)             # (B, 300)
    query_mask = batch["query_mask"].to(device)         # (B, L)
    video_mask = batch["video_mask"].to(device)         # (B, T)
    i0 = batch["i0"].to(device)                         # (B,)
    i1 = batch["i1"].to(device)                         # (B,)
    duration = batch["duration"].to(device)             # (B,)

    B = video_emb.shape[0]
    N = video_emb.shape[1]

    # ---- Forward ----
    cls_logits, reg_offsets, masks = model(
        video_emb, query_tokens, text_emb, query_mask, video_mask
    )
    # cls_logits:  tuple of (B, T_i) per pyramid level
    # reg_offsets: tuple of (B, T_i, 2) per pyramid level
    # masks:       list of R masks, each (B, N)

    # ---- 1. Mask loss (IRLR deep supervision) ----
    gt_mask = make_gt_mask(i0, i1, N).to(device)
    mask_loss, per_iter_mask_losses = compute_mask_loss(masks, gt_mask, cfg.R, cfg.sigma_max, cfg.alpha_entropy)

    # ---- 2. Dense loss (SnAG cls + reg) across pyramid levels ----
    total_dense_loss = torch.tensor(0.0, device=device)
    dense_details = {}
    base_stride = 1.0

    for lvl, (lvl_logits, lvl_offsets) in enumerate(zip(cls_logits, reg_offsets)):
        curr_stride = base_stride * (2 ** lvl)

        # Interpolate video_mask for downsampled levels
        if lvl == 0:
            lvl_mask = video_mask
        else:
            lvl_mask = F.interpolate(
                video_mask.unsqueeze(1).float(),
                size=lvl_logits.shape[1],
                mode='nearest'
            ).squeeze(1).bool()

        lvl_loss, lvl_info = compute_dense_tr_loss(
            lvl_logits, lvl_offsets,
            i0.float(), i1.float(),       # pass clip indices as "seconds"
            lvl_mask,
            sec_per_step=curr_stride,     # stride converts indices to level grid
        )
        total_dense_loss = total_dense_loss + lvl_loss
        dense_details[f"L_cls_lvl{lvl}"] = lvl_info.get("L_cls", torch.tensor(0.0)).item()
        dense_details[f"L_reg_lvl{lvl}"] = lvl_info.get("L_reg", torch.tensor(0.0)).item()

    # ---- 3. Total loss ----
    total_loss = cfg.alpha_mask * mask_loss + total_dense_loss

    # ---- Backward ----
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # ---- LR schedule ----
    if lr_schedule:
        lr = cosine_schedule_with_warmup(global_step, base_lr, warmup_steps, total_steps)
    else:
        lr = base_lr
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    optimizer.step()

    # ---- Metrics (no grad) ----
    metrics = {
        "total_loss": total_loss.item(),
        "mask_loss": mask_loss.item(),
        "dense_loss": total_dense_loss.item(),
        "lr": lr,
    }
    metrics.update(dense_details)
    for r, ml in enumerate(per_iter_mask_losses):
        metrics[f"mask_iter{r}"] = ml

    return metrics, cls_logits, reg_offsets


# =============================================================================
# Evaluation step
# =============================================================================

@torch.no_grad()
def eval_step(model, batch, cfg, device):
    model.eval()

    video_emb = batch["video_emb"].to(device)
    query_tokens = batch["query_tokens"].to(device)
    text_emb = batch["text_emb"].to(device)
    query_mask = batch["query_mask"].to(device)
    video_mask = batch["video_mask"].to(device)
    i0 = batch["i0"].to(device)
    i1 = batch["i1"].to(device)
    duration = batch["duration"].to(device)

    N = video_emb.shape[1]

    cls_logits, reg_offsets, masks = model(
        video_emb, query_tokens, text_emb, query_mask, video_mask
    )

    # Mask loss
    gt_mask = make_gt_mask(i0, i1, N).to(device)
    mask_loss, _ = compute_mask_loss(masks, gt_mask, cfg.R, cfg.sigma_max)

    # Dense loss
    total_dense_loss = torch.tensor(0.0, device=device)
    base_stride = 1.0
    for lvl, (lvl_logits, lvl_offsets) in enumerate(zip(cls_logits, reg_offsets)):
        curr_stride = base_stride * (2 ** lvl)
        if lvl == 0:
            lvl_mask = video_mask
        else:
            lvl_mask = F.interpolate(
                video_mask.unsqueeze(1).float(),
                size=lvl_logits.shape[1],
                mode='nearest'
            ).squeeze(1).bool()
        lvl_loss, _ = compute_dense_tr_loss(
            lvl_logits, lvl_offsets,
            i0.float(), i1.float(),
            lvl_mask,
            sec_per_step=curr_stride,
        )
        total_dense_loss = total_dense_loss + lvl_loss

    total_loss = cfg.alpha_mask * mask_loss + total_dense_loss

    metrics = {
        "total_loss": total_loss.item(),
        "mask_loss": mask_loss.item(),
        "dense_loss": total_dense_loss.item(),
    }
    return metrics, cls_logits, reg_offsets


# =============================================================================
# Main training loop
# =============================================================================

def train(args, tr_ds, val_ds, test_ds, tr_loader, val_loader, test_loader):
    RUN_NAME = args.run_name
    LOG_DIR = args.log_dir
    CKPT_DIR = args.checkpoint_dir
    TBOARD_DIR = args.tboard_dir
    run_ckpt_dir = os.path.join(CKPT_DIR, RUN_NAME)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TBOARD_DIR, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(args.gpu_id)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ---- Auto-detect input dimensions from a real batch ----
    sample_batch = next(iter(tr_loader))
    vid_dim = sample_batch["video_emb"].shape[-1]
    word_dim = sample_batch["query_tokens"].shape[-1]
    global_dim = sample_batch["text_emb"].shape[-1]
    print(f"Auto-detected dims: video={vid_dim}, word={word_dim}, global={global_dim}")

    # ---- Config ----
    cfg = IRLRConfig(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_z_tokens=args.num_z_tokens,
        R=args.R,
        ffn_ratio=4,
        dropout=args.dropout,
        sigma_max=args.sigma_max,
        alpha_mask=args.alpha_mask,
        alpha_entropy=args.alpha_entropy,
        video_feat_dim=vid_dim,
        text_word_dim=word_dim,
        text_global_dim=global_dim,
    )

    # ---- Checkpoint resume ----
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    ckpt = None

    if args.resume:
        checkpoint_files = [f for f in os.listdir(run_ckpt_dir)
                           if f.startswith("epoch_") and f.endswith(".pth")]
        last_ckpt_path = os.path.join(run_ckpt_dir, "last.pth")
        target_ckpt_path = None

        if os.path.exists(last_ckpt_path):
            target_ckpt_path = last_ckpt_path
        elif checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            target_ckpt_path = os.path.join(run_ckpt_dir, checkpoint_files[-1])

        if target_ckpt_path:
            print(f"Resuming from checkpoint: {target_ckpt_path}")
            ckpt = torch.load(target_ckpt_path, map_location="cpu", weights_only=False)
            start_epoch = ckpt.get("epoch", 0) + 1
            global_step = ckpt.get("global_step", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
        else:
            print(f"No checkpoint found in {run_ckpt_dir}; starting from scratch.")

    # ---- Save config ----
    config_dict = {
        "hidden_size": cfg.hidden_size,
        "num_heads": cfg.num_heads,
        "num_z_tokens": cfg.num_z_tokens,
        "R": cfg.R,
        "sigma_max": cfg.sigma_max,
        "alpha_mask": cfg.alpha_mask,
        "alpha_entropy": cfg.alpha_entropy,
        "dropout": cfg.dropout,
        "video_feat_dim": cfg.video_feat_dim,
        "text_word_dim": cfg.text_word_dim,
        "text_global_dim": cfg.text_global_dim,
        "num_pyramid_levels": args.num_pyramid_levels,
        "head_n_layers": args.head_n_layers,
        "lr": args.lr,
        "batch_size": args.config_batch_size,
        "epochs": args.epochs,
    }
    config_path = os.path.join("./config", f"{RUN_NAME}.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)

    # ---- Logger ----
    logger = Logger(LOG_DIR, RUN_NAME, config_dict, resume=args.resume)
    if ckpt is not None:
        logger.log(f"\n--- RESUMING AT EPOCH {start_epoch} ---")
    else:
        logger.log("--- STARTING NEW SESSION ---")

    # ---- TensorBoard ----
    datetimestr = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    run_base_path = os.path.join(TBOARD_DIR, RUN_NAME)
    if args.resume and os.path.exists(run_base_path):
        subdirs = [os.path.join(run_base_path, d) for d in os.listdir(run_base_path)
                   if os.path.isdir(os.path.join(run_base_path, d))]
        if subdirs:
            tboard_run_path = max(subdirs, key=os.path.getmtime)
        else:
            tboard_run_path = os.path.join(run_base_path, datetimestr)
    else:
        tboard_run_path = os.path.join(run_base_path, datetimestr)
    tbd_writer = SummaryWriter(tboard_run_path)

    logger.log(f"Train samples: {len(tr_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")

    # ---- Model ----
    model = IRLRWithSnAG(
        cfg,
        num_pyramid_levels=args.num_pyramid_levels,
        head_n_layers=args.head_n_layers,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Model parameters: {total_params:,}")
    logger.log(f"  IRLR: video {vid_dim}->{cfg.hidden_size}, R={cfg.R}, K={cfg.num_z_tokens}")
    logger.log(f"  SnAG: {args.num_pyramid_levels} pyramid levels, {args.head_n_layers} head layers")

    # Print parameter breakdown
    irlr_params = sum(p.numel() for p in model.irlr.parameters() if p.requires_grad)
    snag_params = sum(p.numel() for p in model.snag_head.parameters() if p.requires_grad)
    logger.log(f"  IRLR params: {irlr_params:,}")
    logger.log(f"  SnAG params: {snag_params:,}")

    # ---- Optimizer ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    steps_per_epoch = len(tr_loader)
    total_training_steps = steps_per_epoch * args.epochs
    warmup_steps = args.warmup_steps

    logger.log(f"Steps/epoch: {steps_per_epoch}, Total steps: {total_training_steps}, Warmup: {warmup_steps}")

    # ================================================================
    # Training loop
    # ================================================================
    for epoch in range(start_epoch, args.epochs):
        logger.log(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        model.train()

        # Running accumulators (reset every N steps for logging)
        running_loss = running_mask = running_dense = 0.0
        running_thr3 = running_thr5 = running_thr7 = 0.0
        running_count = 0

        # Epoch-level accumulators
        ep_loss = ep_thr3 = ep_thr5 = ep_thr7 = 0.0
        ep_n = 0

        for batch_idx, batch in enumerate(tr_loader):
            global_step += 1

            metrics, cls_logits, reg_offsets = train_step(
                model, batch, optimizer, cfg, device, global_step,
                lr_schedule=args.lr_schedule,
                base_lr=args.lr,
                warmup_steps=warmup_steps,
                total_steps=total_training_steps,
            )

            # ---- Localization metrics (level 0) ----
            with torch.no_grad():
                pred = cls_logits[0]        # (B, T) at full resolution
                extra = reg_offsets[0]      # (B, T, 2)
                video_mask_dev = batch["video_mask"].to(device)
                gt_start = batch["start_sec"].to(device)
                gt_end = batch["end_sec"].to(device)
                max_dur = batch["duration"].to(device)

                loc_metric, stats = compute_localization_metric(
                    pred, video_mask_dev, gt_start, gt_end,
                    offsets=extra, max_dur=max_dur, option="dense_head"
                )

            # Accumulate
            running_loss += metrics["total_loss"]
            running_mask += metrics["mask_loss"]
            running_dense += metrics["dense_loss"]
            running_thr3 += loc_metric["R1@IoU=0.3"]
            running_thr5 += loc_metric["R1@IoU=0.5"]
            running_thr7 += loc_metric["R1@IoU=0.7"]
            running_count += 1

            ep_loss += metrics["total_loss"]
            ep_thr3 += loc_metric["R1@IoU=0.3"]
            ep_thr5 += loc_metric["R1@IoU=0.5"]
            ep_thr7 += loc_metric["R1@IoU=0.7"]
            ep_n += 1

            # ---- Log every N steps ----
            if global_step % args.log_every == 0:
                c = max(running_count, 1)
                logger.log(
                    f"  Step {global_step} | "
                    f"Loss: {running_loss/c:.4f} (mask={running_mask/c:.4f} dense={running_dense/c:.4f}) | "
                    f"R1@0.3: {running_thr3/c:.4f} | "
                    f"R1@0.5: {running_thr5/c:.4f} | "
                    f"R1@0.7: {running_thr7/c:.4f} | "
                    f"LR: {metrics['lr']:.6f}"
                )

                # Print sample predictions
                n_print = min(4, pred.shape[0])
                for i in range(n_print):
                    print(f"    S{i}: Pred [{stats['pred_s'][i]:.2f} - {stats['pred_e'][i]:.2f}] "
                          f"| GT [{stats['gt_s'][i]:.2f} - {stats['gt_e'][i]:.2f}] "
                          f"| IoU: {stats['ious'][i]:.3f}")

                # TensorBoard step-level
                tbd_writer.add_scalar("step/total_loss", running_loss/c, global_step)
                tbd_writer.add_scalar("step/mask_loss", running_mask/c, global_step)
                tbd_writer.add_scalar("step/dense_loss", running_dense/c, global_step)
                tbd_writer.add_scalar("step/R1@0.5", running_thr5/c, global_step)
                tbd_writer.add_scalar("step/lr", metrics["lr"], global_step)

                running_loss = running_mask = running_dense = 0.0
                running_thr3 = running_thr5 = running_thr7 = 0.0
                running_count = 0

        # ---- Epoch-level training metrics ----
        c = max(ep_n, 1)
        tbd_writer.add_scalar("Loss/train", ep_loss / c, epoch + 1)
        tbd_writer.add_scalar("R1@IoU=0.3/train", ep_thr3 / c, epoch + 1)
        tbd_writer.add_scalar("R1@IoU=0.5/train", ep_thr5 / c, epoch + 1)
        tbd_writer.add_scalar("R1@IoU=0.7/train", ep_thr7 / c, epoch + 1)

        # ================================================================
        # Validation
        # ================================================================
        logger.log("Running Validation...")
        model.eval()

        val_loss_sum = val_thr3 = val_thr5 = val_thr7 = 0.0
        val_mask_sum = val_dense_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for batch in val_loader:
                metrics, cls_logits, reg_offsets = eval_step(model, batch, cfg, device)

                pred = cls_logits[0]
                extra = reg_offsets[0]
                video_mask_dev = batch["video_mask"].to(device)
                gt_start = batch["start_sec"].to(device)
                gt_end = batch["end_sec"].to(device)
                max_dur = batch["duration"].to(device)

                loc_metric, stats = compute_localization_metric(
                    pred, video_mask_dev, gt_start, gt_end,
                    offsets=extra, max_dur=max_dur, option="dense_head"
                )

                val_loss_sum += metrics["total_loss"]
                val_mask_sum += metrics["mask_loss"]
                val_dense_sum += metrics["dense_loss"]
                val_thr3 += loc_metric["R1@IoU=0.3"]
                val_thr5 += loc_metric["R1@IoU=0.5"]
                val_thr7 += loc_metric["R1@IoU=0.7"]
                val_n += 1

        c = max(val_n, 1)
        val_loss_avg = val_loss_sum / c
        logger.log(
            f"Val Epoch {epoch+1}: "
            f"Loss: {val_loss_avg:.4f} (mask={val_mask_sum/c:.4f} dense={val_dense_sum/c:.4f}) | "
            f"R1@0.3: {val_thr3/c:.4f} | "
            f"R1@0.5: {val_thr5/c:.4f} | "
            f"R1@0.7: {val_thr7/c:.4f}"
        )

        tbd_writer.add_scalar("Loss/val", val_loss_avg, epoch + 1)
        tbd_writer.add_scalar("R1@IoU=0.3/val", val_thr3 / c, epoch + 1)
        tbd_writer.add_scalar("R1@IoU=0.5/val", val_thr5 / c, epoch + 1)
        tbd_writer.add_scalar("R1@IoU=0.7/val", val_thr7 / c, epoch + 1)

        # ================================================================
        # Checkpointing
        # ================================================================
        ckpt_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config_dict,
            "val_loss": val_loss_avg,
            "val_R1@0.3": val_thr3 / c,
            "val_R1@0.5": val_thr5 / c,
            "val_R1@0.7": val_thr7 / c,
            "best_val_loss": best_val_loss,
        }

        # Save epoch checkpoint
        epoch_ckpt_path = os.path.join(run_ckpt_dir, f"epoch_{epoch+1}.pth")
        torch.save(ckpt_payload, epoch_ckpt_path)
        logger.log(f"Saved checkpoint: {epoch_ckpt_path}")

        # Save best model
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            ckpt_payload["best_val_loss"] = best_val_loss
            best_path = os.path.join(run_ckpt_dir, "best.pth")
            torch.save(ckpt_payload, best_path)
            logger.log(f"New best model (val_loss={best_val_loss:.4f}) saved to {best_path}")

        logger.log("-" * 60)

    # ================================================================
    # Test evaluation (after training)
    # ================================================================
    logger.log("\n=== Final Test Evaluation ===")
    best_path = os.path.join(run_ckpt_dir, "best.pth")
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model"])
        logger.log(f"Loaded best model from {best_path}")

    model.eval()
    test_loss_sum = test_thr3 = test_thr5 = test_thr7 = 0.0
    test_n = 0

    with torch.no_grad():
        for batch in test_loader:
            metrics, cls_logits, reg_offsets = eval_step(model, batch, cfg, device)

            pred = cls_logits[0]
            extra = reg_offsets[0]
            video_mask_dev = batch["video_mask"].to(device)
            gt_start = batch["start_sec"].to(device)
            gt_end = batch["end_sec"].to(device)
            max_dur = batch["duration"].to(device)

            loc_metric, _ = compute_localization_metric(
                pred, video_mask_dev, gt_start, gt_end,
                offsets=extra, max_dur=max_dur, option="dense_head"
            )

            test_loss_sum += metrics["total_loss"]
            test_thr3 += loc_metric["R1@IoU=0.3"]
            test_thr5 += loc_metric["R1@IoU=0.5"]
            test_thr7 += loc_metric["R1@IoU=0.7"]
            test_n += 1

    c = max(test_n, 1)
    logger.log(
        f"TEST RESULTS: "
        f"Loss: {test_loss_sum/c:.4f} | "
        f"R1@0.3: {test_thr3/c:.4f} | "
        f"R1@0.5: {test_thr5/c:.4f} | "
        f"R1@0.7: {test_thr7/c:.4f}"
    )

    tbd_writer.flush()
    tbd_writer.close()
    logger.log("Training complete.")


# =============================================================================
# Param logging
# =============================================================================

def print_model_params(model, logger):
    print(f"{'LAYER NAME':<50} {'SHAPE':<25} {'PARAMS'}")
    print("-" * 85)
    total_params = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        shape = list(param.shape)
        count = param.numel()
        total_params += count
        print(f"{name:<50} {str(shape):<25} {count:,}")
    print("-" * 85)
    print(f"Total Trainable Params: {total_params:,}")
    logger.log(f"Total Trainable Params: {total_params:,}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRLR + SnAG Training on Charades-STA")

    # Run management
    parser.add_argument("--run_name", type=str, default="irlr_exp1")
    parser.add_argument("--log_dir", type=str, default="./irlr_log")
    parser.add_argument("--checkpoint_dir", type=str, default="./irlr_checkpoints")
    parser.add_argument("--tboard_dir", type=str, default="./irlr_tboard")
    parser.add_argument("--data_root", type=str, default="/home/hlpark/common-data/jean")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4501)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--config_batch_size", type=int, default=16)
    parser.add_argument("--lr_schedule", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)

    # IRLR architecture
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_z_tokens", type=int, default=8)
    parser.add_argument("--R", type=int, default=4, help="Number of refinement iterations")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sigma_max", type=float, default=3.0)
    parser.add_argument("--alpha_mask", type=float, default=1.0, help="Weight for mask loss")
    parser.add_argument("--alpha_entropy", type=float, default=0.1, help="Weight for entropy regularization")

    # SnAG head
    parser.add_argument("--num_pyramid_levels", type=int, default=3)
    parser.add_argument("--head_n_layers", type=int, default=2)

    # Dataset
    parser.add_argument("--feature_type", type=str, default="i3d")

    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load dataset
    tr_ds, val_ds, test_ds, tr_loader, val_loader, test_loader = load_dataset(args, args.config_batch_size)

    # Train
    train(args, tr_ds, val_ds, test_ds, tr_loader, val_loader, test_loader)
