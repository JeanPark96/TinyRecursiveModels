import torch
import torch.nn as nn



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
        
        # --- A. FUSION ---
        # Inject reasoning into video stream
        # Q = Video, K = z_H, V = z_H
        # Output is (B, T, D) - "Video frame t attended to relevant reasoning tokens"
        v_norm = self.fusion_norm(video_tokens)
        fused_v, _ = self.fusion_attn(query=v_norm, key=z_H, value=z_H)
        
        # Residual connection + Norm? (Optional, usually simple add is fine here)
        x = video_tokens + fused_v
        
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
        reg_offsets = F.relu(reg_offsets)
        
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
