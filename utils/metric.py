import random
import numpy as np
import json
import torch
from torch import nn

def soft_nms_1d_pytorch(segs, scores, iou_thresh=0.5, sigma=0.5, min_score=0.001, method=2):
    """
    Pure PyTorch implementation of 1D Soft-NMS.
    segs: (N, 2) [start, end]
    scores: (N,)
    method: 1 (linear), 2 (gaussian)
    """
    # 1. Sort by score descending
    indices = scores.argsort(descending=True)
    segs = segs[indices]
    scores = scores[indices]
    
    final_segs = []
    final_scores = []
    
    while len(scores) > 0 and scores[0] > min_score:
        # Pick best
        best_seg = segs[0]
        best_score = scores[0]
        
        final_segs.append(best_seg)
        final_scores.append(best_score)
        
        # If only one left, break
        if len(scores) == 1:
            break
            
        # Remove best from list
        segs = segs[1:]
        scores = scores[1:]
        
        # Compute IoU with best_seg
        # Intersection
        tt1 = torch.maximum(best_seg[0], segs[:, 0])
        tt2 = torch.minimum(best_seg[1], segs[:, 1])
        intersection = (tt2 - tt1).clamp(min=0)
        
        # Union
        len_best = best_seg[1] - best_seg[0]
        len_others = segs[:, 1] - segs[:, 0]
        union = len_best + len_others - intersection
        
        iou = intersection / union.clamp(min=1e-6)
        
        # Soft-NMS Decay
        if method == 1: # Linear
            weight = torch.ones_like(iou)
            weight[iou > iou_thresh] -= iou[iou > iou_thresh]
        else: # Gaussian (Standard for SnAG/DecafNet)
            weight = torch.exp(-(iou * iou) / sigma)
            
        scores = scores * weight
        
        # Re-sort to bring next best to top
        new_idx = scores.argsort(descending=True)
        segs = segs[new_idx]
        scores = scores[new_idx]

    return torch.stack(final_segs), torch.stack(final_scores)
@torch.no_grad()
def decode_dense_candidates(cls_logits, reg_offsets, mask, duration, max_vid_len=256):
    """
    Generates ALL candidate segments.
    Returns:
        proposals: (N, 2) in Seconds
        scores: (N,)
    """
    # 1. Get Scores
    scores = torch.sigmoid(cls_logits) * mask.float() # (B, T)
    
    B, T = scores.shape
    candidates_list = []
    
    for i in range(B):
        # 2. Filter low scores to speed up NMS (Standard trick)
        # SOTA usually keeps top 100 or scores > 0.01
        valid_indices = torch.nonzero(scores[i] > 0.01).squeeze(1)
        
        if len(valid_indices) == 0:
            # Fallback if nothing is confident: pick max
            valid_indices = torch.argmax(scores[i]).unsqueeze(0)

        # 3. Decode specific indices
        idx_t = valid_indices.float()
        
        # Get Offsets: (N_valid, 2)
        cur_offsets = reg_offsets[i][valid_indices] 
        off_l = cur_offsets[:, 0]
        off_r = cur_offsets[:, 1]
        
        # Grid Indices -> Normalized Time
        # Start = (t - off_l) / 256
        # End   = (t + off_r) / 256
        pred_s = (idx_t - off_l) / float(max_vid_len)
        pred_e = (idx_t + off_r) / float(max_vid_len)
        
        # Convert to Seconds
        dur = duration[i]
        pred_s = pred_s * dur
        pred_e = pred_e * dur
        
        # Clamp
        pred_s = pred_s.clamp(min=0, max=dur)
        pred_e = pred_e.clamp(min=0, max=dur)
        
        # Stack [Start, End]
        proposals = torch.stack([pred_s, pred_e], dim=1)
        prop_scores = scores[i][valid_indices]
        
        candidates_list.append((proposals, prop_scores))
        
    return candidates_list

        
@torch.no_grad()
def compute_retrieval_metrics(
    proposals_list, # List of (segs, scores) tuples from decode function
    gt_start,       # (B,) Tensor
    gt_end,          # (B,) Tensor
    iou_thresh=0.1, # Standard for temporal grounding
    sigma=0.9, 
    method=2
):
    """
    Computes R@1 and R@5 at IoU=0.3, 0.5, 0.7 and returns batch statistics.
    """
    # Metrics accumulator
    batch_metrics = {
        "R1@IoU=0.3": [], "R1@IoU=0.5": [], "R1@IoU=0.7": [],
        "R5@IoU=0.3": [], "R5@IoU=0.5": [], "R5@IoU=0.7": [],
        "mIoU": []
    }
    
    # Batch Stats accumulator
    stat_pred_s = []
    stat_pred_e = []
    stat_gt_s = []
    stat_gt_e = []
    stat_ious = []
    
    for i, (segs, scores) in enumerate(proposals_list):
        # 1. Apply Soft-NMS (Standard SOTA processing)
        # Uses the python implementation we discussed
        nms_segs, nms_scores = soft_nms_1d_pytorch(
            segs, scores, 
            iou_thresh=iou_thresh, 
            sigma=sigma, 
            method=method
        )
        
        # 2. Extract Top-K
        # If NMS removed everything (rare), fallback to [0,0]
        if len(nms_segs) == 0:
            print("Warning: NMS removed everything. Fallback to [0,0]")
            top_1_seg = torch.tensor([0.0, 0.0], device=gt_start.device)
            top_5_segs = top_1_seg.unsqueeze(0)
        else:
            top_1_seg = nms_segs[0]       # Best prediction
            top_5_segs = nms_segs[:5]     # Top-5 candidates
        
        # 3. Get Ground Truth for this sample
        g_s = gt_start[i]
        g_e = gt_end[i]
        
        # 4. Compute IoU for Top-5
        # Intersection
        inter_min = torch.maximum(top_5_segs[:, 0], g_s)
        inter_max = torch.minimum(top_5_segs[:, 1], g_e)
        inter_len = (inter_max - inter_min).clamp(min=0)
        
        # Union
        pred_len = top_5_segs[:, 1] - top_5_segs[:, 0]
        gt_len = g_e - g_s
        union_len = pred_len + gt_len - inter_len
        
        ious = inter_len / union_len.clamp(min=1e-6)
        
        # 5. Compute Metrics
        top1_iou = ious[0].item()
        batch_metrics["mIoU"].append(top1_iou) 
        
        for thresh in [0.3, 0.5, 0.7]:
            # R@1
            r1 = 1.0 if top1_iou >= thresh else 0.0
            batch_metrics[f"R1@IoU={thresh}"].append(r1)
            # R@5
            r5 = 1.0 if (ious >= thresh).any() else 0.0
            batch_metrics[f"R5@IoU={thresh}"].append(r5)
            
        # 6. Store Stats (Top-1 prediction only)
        stat_pred_s.append(top_1_seg[0].cpu())
        stat_pred_e.append(top_1_seg[1].cpu())
        stat_gt_s.append(g_s.cpu())
        stat_gt_e.append(g_e.cpu())
        stat_ious.append(ious[0].cpu()) # Store Top-1 IoU

    # Average metrics over batch
    final_metrics = {k: torch.tensor(v).mean().item() for k, v in batch_metrics.items()}
    
    # Pack Stats
    batch_stats = {
        "pred_s": torch.stack(stat_pred_s),
        "pred_e": torch.stack(stat_pred_e),
        "gt_s": torch.stack(stat_gt_s),
        "gt_e": torch.stack(stat_gt_e),
        "ious": torch.stack(stat_ious)
    }
    
    return final_metrics, batch_stats

def temporal_iou_1d(pred_s, pred_e, gt_s, gt_e):
    """
    Computes IoU between two time spans.
    Input shapes: (B,)
    """
    # Intersection
    inter_min = torch.maximum(pred_s, gt_s)
    inter_max = torch.minimum(pred_e, gt_e)
    inter_len = torch.clamp(inter_max - inter_min, min=0)
    
    # Union
    union_min = torch.minimum(pred_s, gt_s)
    union_max = torch.maximum(pred_e, gt_e)
    union_len = torch.clamp(union_max - union_min, min=1e-6)
    
    return inter_len / union_len

def decode_dense_predictions(cls_logits, reg_offsets, mask, sec_per_step=1.0):
    """
    Decodes dense predictions into (start, end) spans.
    """
    # 1. Sigmoid & Threshold
    scores = torch.sigmoid(cls_logits) * mask.float() # (B, T)
    
    # 2. Pick best frame (Simple Argmax)
    # For more advanced usage, you can use NMS, but Argmax works for single-instance
    best_score, best_idx = scores.max(dim=1) # (B,)
    
    # 3. Get offsets at that frame
    # reg_offsets: (B, T, 2)
    # Gather: select the offset corresponding to best_idx
    batch_indices = torch.arange(scores.shape[0], device=scores.device)
    best_offsets = reg_offsets[batch_indices, best_idx] # (B, 2)
    
    off_l = best_offsets[:, 0]
    off_r = best_offsets[:, 1]
    
    # 4. Calculate Start/End
    # Start = t - off_l
    # End   = t + off_r
    pred_s_idx = best_idx.float() - off_l
    pred_e_idx = best_idx.float() + off_r
    
    return pred_s_idx * sec_per_step, pred_e_idx * sec_per_step

@torch.no_grad()
def compute_localization_metric(
    logits,          # (B, T)
    mask,            # (B, T)
    gt_start_sec,    # (B,)
    gt_end_sec,      # (B,)
    sec_per_step=1.0, 
    max_dur=None, #(B,)
    offsets=None, # optional, valid only if dense loc prediction
    option="dense_head" 
):
    """
    Computes IoU-based metrics: R@0.3, R@0.5, R@0.7 and mIoU.
    Note: True mAP requires ranking over the entire dataset. 
    This function returns per-batch 'Recall' (Acc) which approximates mAP during training.
    """
    
    if option == "dense_head":
        if max_dur != None:
            sec_per_step =  (max_dur / 256)
            #print(sec_per_step)
        pred_s, pred_e = decode_dense_predictions(logits, offsets, mask, sec_per_step)
        scores = []
    else:
        raise ValueError(f"Unknown option: {option}")

        
    # 2. Compute IoU for the batch
    ious = temporal_iou_1d(pred_s, pred_e, gt_start_sec, gt_end_sec)
    
    # 3. Compute Metrics
    metrics = {}
    metrics["mIoU"] = ious.mean().item()
    
    # Recall at IoU thresholds (Success Rate)
    for thresh in [0.3, 0.5, 0.7]:
        # (B,) bool -> float -> mean
        r_at_k = (ious >= thresh).float().mean().item()
        metrics[f"R1@IoU={thresh}"] = r_at_k
        
    # 'mAP' Proxy: Average of Recalls [0.5 : 0.05 : 0.95] is a common video eval metric
    # But usually people want Average Precision. Since we can't rank the whole dataset here,
    # we return the Batch Average IoU and R@0.5 as the primary indicators.
    
    # If you need to accumulate for global mAP, return lists:
    batch_stats = {
        "pred_s": pred_s.cpu(),
        "pred_e": pred_e.cpu(),
        #"scores": scores.cpu(), # Confidence score for ranking
        "gt_s": gt_start_sec.cpu(),
        "gt_e": gt_end_sec.cpu(),
        "ious": ious.cpu()
    }
    
    return metrics, batch_stats