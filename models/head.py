import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.layers import CastedLinear
# ---------------------------
# localization head similar to DeCafNet and SnAG
# ---------------------------

class MaskedConv1D(nn.Module):
    """
    1D Conv that respects the mask (sets padded output to 0).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x, mask=None):
        # x: (B, C, T)
        # mask: (B, T)
        x = self.conv(x)
        if mask is not None:
            # Expand mask to (B, 1, T) and multiply
            x = x * mask.unsqueeze(1).to(x.dtype)
        return x, mask
    
class TRM_DenseHead(nn.Module):
    def __init__(self, hidden_size, num_heads=4, prior_prob=0.01):
        super().__init__()
        
        # 1. Fusion Layer: Inject 'z_H' (intent) into 'v' (video)
        # We use simple Multi-Head Attention: Query=Video, Key/Value=z_H
        self.fusion_norm = nn.LayerNorm(hidden_size)
        self.fusion_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # 2. Classification Tower (Saliency)
        self.cls_tower = nn.Sequential(
            MaskedConv1D(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_size),
            nn.ReLU(inplace=True),
            MaskedConv1D(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.cls_head = MaskedConv1D(hidden_size, 1, kernel_size=3, padding=1, bias=True)
        # Gating Mechanism
        # Projects the reasoning signal to a 0-1 scalar for every channel/time
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        # 3. Regression Tower (Offsets: dist_to_start, dist_to_end)
        self.reg_tower = nn.Sequential(
            MaskedConv1D(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_size),
            nn.ReLU(inplace=True),
            MaskedConv1D(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.reg_head = MaskedConv1D(hidden_size, 2, kernel_size=3, padding=1, bias=True)
        
        # --- Initialization (CRITICAL) ---
        # Initialize CLS bias to -log((1-p)/p) to prevent massive background loss at start
        bias_init = -np.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_head.conv.bias, bias_init)

    def forward(self, z_H, video_tokens, video_mask):
        """
        z_H: (B, Ny, D) - Reasoning tokens
        video_tokens: (B, T, D) - Video context
        video_mask: (B, T) - Boolean mask
        """
        B, T, D = video_tokens.shape
        _, T_reason, _ = z_H.shape         # 32

        # --- 1. EXPLICIT UPSAMPLING ---
        # We take the 32 reasoning tokens and stretch them to 256.
        # This gives us a "reasoning backbone" that matches the video's length.
        # (B, 32, D) -> (B, D, 32) -> (B, D, 256) -> (B, 256, D)
        z_H_up = z_H.transpose(1, 2)
        z_H_up = F.interpolate(z_H_up, size=T, mode='linear', align_corners=False)
        z_H_up = z_H_up.transpose(1, 2)
        
        # --- A. FUSION ---
        # Inject reasoning into video stream
        # Q = Video, K = z_H, V = z_H
        # Output is (B, T, D) - "Video frame t attended to relevant reasoning tokens"
        v_norm = self.fusion_norm(video_tokens)
        fused_v, _ = self.fusion_attn(query=v_norm, key=z_H_up, value=z_H_up)
        
        # Residual connection + Norm? (Optional, usually simple add is fine here)
        #x = video_tokens + fused_v
        # x = fused_v
        # Compute the Gate
        # "How much should I trust the raw video at this pixel?"
        # shape: (B, T, D)
        gate = self.gate_proj(z_H_up)
        
        # 3. Apply Gating (The "Spotlight")
        # We allow the Raw Video features to pass through, but ONLY where the gate is open.
        # plus we keep the reasoning signal itself.
        # Formula: (Video * Gate) + Reasoning
        x = (video_tokens * gate) + fused_v
        # Prepare for Conv1D: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        
        # --- B. CLASSIFICATION ---
        # "Is frame t inside the action?"
        cls_feat = x
        for layer in self.cls_tower:
            if isinstance(layer, MaskedConv1D):
                cls_feat, _ = layer(cls_feat, video_mask)
            else:
                cls_feat = layer(cls_feat)
        
        cls_logits, _ = self.cls_head(cls_feat, video_mask)
        cls_logits = cls_logits.transpose(1, 2).squeeze(-1) # (B, T)

        # --- C. REGRESSION ---
        # "Where does the action start/end relative to t?"
        reg_feat = x
        for layer in self.reg_tower:
            if isinstance(layer, MaskedConv1D):
                reg_feat, _ = layer(reg_feat, video_mask)
            else:
                reg_feat = layer(reg_feat)

        reg_offsets, _ = self.reg_head(reg_feat, video_mask)
        reg_offsets = reg_offsets.transpose(1, 2) # (B, T, 2)
        # Apply ReLU because distances must be positive
        # reg_offsets = F.relu(reg_offsets)
        reg_offsets = torch.exp(reg_offsets)
        
        return cls_logits, reg_offsets
    

# ---------------------------
# old head
# ---------------------------
class MLPBoundaryHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or d_model
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.start = nn.Linear(d_model, 1)
        self.end   = nn.Linear(d_model, 1)

    def forward(self, H, mask=None):
        # H: (B,T,D)
        x = self.mlp(self.norm(H))  # (B,T,D)
        s = self.start(x).squeeze(-1)  # (B,T)
        e = self.end(x).squeeze(-1)    # (B,T)
        if mask is not None:
            s = s.masked_fill(~mask, -1e9)
            e = e.masked_fill(~mask, -1e9)
        return s, e

class StartConditionedBoundaryHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.base = MLPBoundaryHead(d_model, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.end2 = nn.Linear(d_model, 1)

    def forward(self, H, mask=None):
        start_logits, end_logits0 = self.base(H, mask=mask)

        # soft start embedding (differentiable)
        ps = torch.softmax(start_logits, dim=-1)  # already masked if mask applied
        s_emb = (ps[:, :, None] * H).sum(dim=1)   # (B,D)

        # condition end prediction using start embedding
        x = self.fuse(torch.cat([H, s_emb[:, None, :].expand_as(H)], dim=-1))
        end_logits = self.end2(x).squeeze(-1)

        if mask is not None:
            end_logits = end_logits.masked_fill(~mask, -1e9)
        return start_logits, end_logits
    
class TRM_LatentHead(nn.Module):
    def __init__(self, hidden_size, max_frames):
        super().__init__()
        self.max_frames = max_frames
        
        # 1. Center Frame Classifier (Logits per frame)
        # Input: z_H -> Output: Score for every frame (T)
        self.center_cls = CastedLinear(hidden_size, max_frames, bias=True)
        
        # 2. Dense Regression Head (Offsets per frame)
        # Input: z_H -> Output: [Left_Offset, Right_Offset] for EVERY frame
        # We output T * 2 values and reshape them to (B, T, 2)
        self.reg_head = nn.Sequential(
            CastedLinear(hidden_size, hidden_size, bias=True),
            nn.ReLU(),
            CastedLinear(hidden_size, max_frames * 2, bias=True) 
        )

    def forward(self, z_H, mask=None):
        # z_H shape: (Batch, Hidden) or (Batch, Num_Tokens, Hidden)
        
        # 1. Logits: (Batch, ..., Max_Frames)
        center_logits = self.center_cls(z_H)
        
        # 2. Regs: (Batch, ..., Max_Frames * 2) -> Reshape to (Batch, ..., Max_Frames, 2)
        reg_preds_flat = self.reg_head(z_H)
        
        # Dynamically reshape keeping leading dimensions
        # If z_H is (B, H), shape becomes (B, T, 2)
        # If z_H is (B, S, H), shape becomes (B, S, T, 2)
        out_shape = reg_preds_flat.shape[:-1] + (self.max_frames, 2)
        reg_preds = reg_preds_flat.view(out_shape)
        
        # --- THE FIX: Force non-negative offsets ---
        reg_preds = F.relu(reg_preds)
        
        # Apply mask if provided
        if mask is not None:
            # mask: (Batch, T)
            # Expand mask to match z_H dimensions if needed
            if center_logits.ndim == 3 and mask.ndim == 2:
                mask_expanded = mask.unsqueeze(1) # (B, 1, T)
            else:
                mask_expanded = mask

            # Pad mask if needed
            if mask_expanded.shape[-1] < self.max_frames:
                pad_amt = self.max_frames - mask_expanded.shape[-1]
                mask_expanded = F.pad(mask_expanded, (0, pad_amt), value=0)
            
            # Slice to max_frames
            mask_slice = mask_expanded[..., :self.max_frames].bool()
            
            # Apply Mask
            center_logits = center_logits.masked_fill(~mask_slice, -1e4)

        return center_logits, reg_preds

import math
class Scale(nn.Module):
    """
    A learnable scale parameter.
    This is critical for regression heads to match the range of ground truth offsets.
    """
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale

class TRM_SnAGLikeHead(nn.Module):
    def __init__(self, hidden_size, kernel_size=3, prior_prob=0.01):
        super().__init__()
        
        # SnAG uses 2 layers of (Conv -> LayerNorm -> ReLU)
        # Note: SnAG uses 'MaskedConv1D', but since we mask the input/output manually,
        # standard Conv1d is mathematically equivalent for the valid regions.
        
        # --- 1. Classification Branch ---
        self.cls_convs = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size, stride=1, padding=kernel_size//2, bias=False),
            nn.LayerNorm(hidden_size), # LayerNorm typically applied to (Batch, Time, Channels) in PyTorch
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size, stride=1, padding=kernel_size//2, bias=False),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.cls_head = nn.Conv1d(hidden_size, 1, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

        # --- 2. Regression Branch ---
        self.reg_convs = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size, stride=1, padding=kernel_size//2, bias=False),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size, stride=1, padding=kernel_size//2, bias=False),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.reg_head = nn.Conv1d(hidden_size, 2, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
        # CRITICAL: Learnable Scale for Regression
        self.reg_scale = Scale(init_value=1.0)
        
        self.prior_prob = prior_prob
        self.init_weights()

    def init_weights(self):
        # 1. Classification Prior
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.cls_head.bias, bias_value)
        
        # 2. Regression Init (Zero bias helps stability with Scale)
        torch.nn.init.constant_(self.reg_head.bias, 0)
        
        # Normal init for weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.normal_(m.weight, std=0.01)

    def forward(self, z_H, mask=None):
        # z_H: (Batch, Time, Hidden)
        
        # PyTorch Conv1d needs (Batch, Hidden, Time)
        x = z_H.transpose(1, 2)
        
        # --- Classification ---
        # Note: PyTorch LayerNorm expects channels LAST. 
        # But Conv1d outputs channels MIDDLE. We must swap for Norm, then swap back.
        # To avoid complex swapping logic in Sequential, we implement loop manually.
        
        # CLS Path
        c_feat = x
        for layer in self.cls_convs:
            if isinstance(layer, nn.LayerNorm):
                c_feat = c_feat.transpose(1, 2) # (B, T, H)
                c_feat = layer(c_feat)
                c_feat = c_feat.transpose(1, 2) # (B, H, T)
            else:
                c_feat = layer(c_feat)
        
        cls_logits = self.cls_head(c_feat) # (B, 1, T)

        # REG Path
        r_feat = x
        for layer in self.reg_convs:
            if isinstance(layer, nn.LayerNorm):
                r_feat = r_feat.transpose(1, 2)
                r_feat = layer(r_feat)
                r_feat = r_feat.transpose(1, 2)
            else:
                r_feat = layer(r_feat)
                
        reg_out = self.reg_head(r_feat) # (B, 2, T)
        
        # Apply Scale AND ReLU
        #
        reg_offsets = F.relu(self.reg_scale(reg_out))
        
        # Transpose back to (B, T, C)
        cls_logits = cls_logits.transpose(1, 2).squeeze(-1) # (B, T)
        reg_offsets = reg_offsets.transpose(1, 2)           # (B, T, 2)
        
        # --- Masking ---
        if mask is not None:
            # Expand mask to match output time dimension if needed
            T_out = cls_logits.shape[1]
            if mask.shape[1] < T_out:
                mask = F.pad(mask, (0, T_out - mask.shape[1]), value=0)
            
            mask_bool = mask[:, :T_out].bool()
            
            # Mask Logits (-inf for Focal Loss)
            cls_logits = cls_logits.masked_fill(~mask_bool, -1e4)
            
            # Mask Regression (0.0 for clean gradients)
            reg_mask = mask_bool.unsqueeze(-1)
            reg_offsets = reg_offsets.masked_fill(~reg_mask, 0.0)
            
        return cls_logits, reg_offsets