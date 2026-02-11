"""
IRLR Module: Iterative Refinement with Latent Reasoning
Combines recursive mask refinement with SnAG PyramidBoundaryPredictor for
temporal moment localization on Charades-STA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

from models.decafnet_head import PyramidBoundaryPredictor


# =============================================================================
# Config
# =============================================================================

@dataclass
class IRLRConfig:
    hidden_size: int = 256
    num_heads: int = 8
    num_z_tokens: int = 8
    R: int = 4
    ffn_ratio: int = 4
    dropout: float = 0.1
    sigma_max: float = 3.0
    alpha_mask: float = 1.0       # weight for mask loss
    alpha_entropy: float = 0.1    # weight for entropy regularization
    # Input dimensions (set from actual batch shapes):
    video_feat_dim: int = 2048
    text_word_dim: int = 300      # query_tokens last dim
    text_global_dim: int = 300    # text_emb last dim


# =============================================================================
# Utilities
# =============================================================================

def make_gt_mask(i0, i1, num_clips):
    """Create binary ground-truth mask from start/end clip indices."""
    device = i0.device
    clip_indices = torch.arange(num_clips, device=device).unsqueeze(0)
    start = i0.unsqueeze(1).float()
    end = i1.unsqueeze(1).float()
    gt_mask = ((clip_indices >= start) & (clip_indices <= end)).float()
    return gt_mask


def blur_mask(m_gt, sigma):
    """Apply Gaussian blur to a 1D mask for coarse-to-fine supervision."""
    if sigma <= 0:
        return m_gt
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = torch.arange(kernel_size, dtype=m_gt.dtype, device=m_gt.device)
    x = x - kernel_size // 2
    kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)
    padding = kernel_size // 2
    blurred = F.conv1d(m_gt.unsqueeze(1), kernel, padding=padding).squeeze(1)
    return blurred.clamp(0.0, 1.0)


def mask_entropy(m):
    """Binary entropy of mask values in [0, 1]. High = uncertain, low = sharp."""
    mc = m.clamp(1e-6, 1 - 1e-6)
    return -(mc * mc.log() + (1 - mc) * (1 - mc).log()).mean()


def compute_mask_loss(masks, gt_mask, R, sigma_max, alpha_entropy=0.0):
    """
    Deep supervision loss with entropy regularization.

    BCE: each iteration gets a progressively sharper target.
    Entropy: early iterations are encouraged to be uncertain (high entropy),
             later iterations are encouraged to be sharp (low entropy).
             This prevents all iterations from collapsing to the same mask.
    """
    total_bce = 0.0
    total_entropy_reg = 0.0
    per_iter_losses = []

    for r in range(R):
        sigma = sigma_max * (1.0 - (r + 1) / R)
        target = blur_mask(gt_mask, sigma)
        iter_loss = F.binary_cross_entropy(masks[r], target)
        per_iter_losses.append(iter_loss.item())
        total_bce += iter_loss

        # Entropy regularization: target entropy decreases with iteration
        # Iter 0 (coarse) → high target entropy, Iter R-1 (fine) → low target entropy
        if alpha_entropy > 0:
            actual_ent = mask_entropy(masks[r])
            # Target entropy: linearly decrease from ~0.6 (soft) to ~0.05 (sharp)
            target_ent = 0.6 * (1.0 - (r + 1) / R) + 0.05
            total_entropy_reg += (actual_ent - target_ent) ** 2

    total_loss = total_bce / R + alpha_entropy * total_entropy_reg / R
    return total_loss, per_iter_losses


# =============================================================================
# Masked Cross-Attention
# =============================================================================

class MaskedCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = self.d_head ** -0.5
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, kv, mask):
        B, K, _ = query.shape
        N = kv.shape[1]
        H = self.num_heads
        Q = self.W_Q(query).view(B, K, H, self.d_head).transpose(1, 2)
        Key = self.W_K(kv).view(B, N, H, self.d_head).transpose(1, 2)
        V = self.W_V(kv).view(B, N, H, self.d_head).transpose(1, 2)
        attn_logits = torch.matmul(Q, Key.transpose(-1, -2)) * self.scale
        mask_clamped = mask.clamp(min=1e-6)
        mask_bias = torch.log(mask_clamped).unsqueeze(1).unsqueeze(2)
        attn_logits = attn_logits + mask_bias
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, K, -1)
        out = self.W_O(out)
        return out


# =============================================================================
# Shared Block (all operations at hidden_size)
# =============================================================================

class IRLRSharedBlock(nn.Module):
    def __init__(self, d_model, num_heads, R=4, ffn_ratio=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.R = R
        self.W_a = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.W_a.weight, gain=0.1)

        # Per-iteration mask temperatures: initialized so early iterations
        # produce soft masks (low temp → scores compressed → sigmoid ≈ 0.5)
        # and later iterations produce sharp masks (high temp → scores spread).
        #   Iter 0: scale ≈ 0.3/sqrt(d)  (soft)
        #   Iter R-1: scale ≈ 1.0/sqrt(d) (sharp)
        base = 1.0 / math.sqrt(d_model)
        init_temps = torch.linspace(0.3 * base, 1.0 * base, R)
        self.mask_scales = nn.Parameter(init_temps)

        self.masked_cross_attn = MaskedCrossAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.query_cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * ffn_ratio, d_model), nn.Dropout(dropout),
        )
        self.norm4 = nn.LayerNorm(d_model)

    def forward(self, Z, F_video, Q_words, iteration_idx, query_mask=None):
        Z_proj = self.W_a(Z)
        scores = torch.matmul(Z_proj, F_video.transpose(-1, -2)).mean(dim=1) * self.mask_scales[iteration_idx]
        m = torch.sigmoid(scores)

        cross_out = self.masked_cross_attn(Z, F_video, m)
        Z = self.norm1(Z + cross_out)

        self_out, _ = self.self_attn(Z, Z, Z)
        Z = self.norm2(Z + self_out)

        key_padding_mask = None
        if query_mask is not None:
            key_padding_mask = ~query_mask

        query_out, _ = self.query_cross_attn(Z, Q_words, Q_words, key_padding_mask=key_padding_mask)
        Z = self.norm3(Z + query_out)

        Z = self.norm4(Z + self.ffn(Z))
        return Z, m


# =============================================================================
# IRLR Module
# =============================================================================

class IRLRModule(nn.Module):
    def __init__(self, cfg: IRLRConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_size

        # Input projections
        self.video_proj = nn.Sequential(
            nn.Linear(cfg.video_feat_dim, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.text_word_proj = nn.Sequential(
            nn.Linear(cfg.text_word_dim, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.text_global_proj = nn.Sequential(
            nn.Linear(cfg.text_global_dim, d),
            nn.LayerNorm(d),
        )

        # Latent tokens
        self.latent_embeds = nn.Parameter(torch.randn(cfg.num_z_tokens, d) * 0.02)
        self.W_q_init = nn.Linear(d, d)

        # Iteration embeddings
        self.iter_embeds = nn.Parameter(torch.randn(cfg.R, 1, d) * 0.02)

        # Shared block
        self.shared_block = IRLRSharedBlock(d, cfg.num_heads, R=cfg.R, ffn_ratio=cfg.ffn_ratio, dropout=cfg.dropout)

        # Reverse cross-attention
        self.reverse_cross_attn = nn.MultiheadAttention(
            d, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.reverse_norm = nn.LayerNorm(d)

        # Feature combination: [F_video; f_prime; m*F_video] -> d
        self.feature_combine = nn.Sequential(
            nn.Linear(d * 3, d), nn.GELU(), nn.Linear(d, d),
        )
        self.output_norm = nn.LayerNorm(d)

    def forward(self, video_emb, query_tokens, text_emb, query_mask=None):
        """
        Args:
            video_emb:    (B, N, video_feat_dim)
            query_tokens: (B, L, text_word_dim)
            text_emb:     (B, text_global_dim)
            query_mask:   (B, L) bool
        Returns:
            F_enhanced:   (B, N, hidden_size)
            masks:        list of R masks, each (B, N)
        """
        F_video = self.video_proj(video_emb)
        Q_words = self.text_word_proj(query_tokens)
        q_global = self.text_global_proj(text_emb)

        B = F_video.shape[0]

        Z = (self.latent_embeds.unsqueeze(0).expand(B, -1, -1)
             + self.W_q_init(q_global).unsqueeze(1))

        masks = []
        for r in range(self.cfg.R):
            Z_input = Z + self.iter_embeds[r]
            Z, m_r = self.shared_block(Z_input, F_video, Q_words, iteration_idx=r, query_mask=query_mask)
            masks.append(m_r)

        # Reverse cross-attention
        f_prime, _ = self.reverse_cross_attn(F_video, Z, Z)
        f_prime = self.reverse_norm(F_video + f_prime)

        # Feature combination
        m_final = masks[-1].unsqueeze(-1)
        combined = torch.cat([F_video, f_prime, m_final * F_video], dim=-1)
        F_enhanced = self.feature_combine(combined)
        F_enhanced = self.output_norm(F_enhanced + F_video)

        return F_enhanced, masks


# =============================================================================
# End-to-End Model: IRLR + SnAG Pyramid Head
# =============================================================================

class IRLRWithSnAG(nn.Module):
    """
    Combines IRLR iterative feature enhancement with SnAG pyramid prediction.

    Forward flow:
        video_emb -> IRLR -> F_enhanced -> PyramidBoundaryPredictor -> (cls, reg)
                                        -> masks (for deep supervision)
    """
    def __init__(self, cfg: IRLRConfig, num_pyramid_levels=3, head_n_layers=2):
        super().__init__()
        self.irlr = IRLRModule(cfg)
        self.snag_head = PyramidBoundaryPredictor(
            embd_dim=cfg.hidden_size,
            num_levels=num_pyramid_levels,
            n_layers=head_n_layers,
        )
        self.cfg = cfg
        self.num_pyramid_levels = num_pyramid_levels

    def forward(self, video_emb, query_tokens, text_emb, query_mask, video_mask):
        """
        Args:
            video_emb:    (B, N, video_feat_dim)
            query_tokens: (B, L, text_word_dim)
            text_emb:     (B, text_global_dim)
            query_mask:   (B, L) bool
            video_mask:   (B, N) bool
        Returns:
            cls_logits:   tuple of (B, T_i) per pyramid level
            reg_offsets:  tuple of (B, T_i, 2) per pyramid level
            masks:        list of R masks from IRLR, each (B, N)
        """
        # 1. IRLR enhancement
        F_enhanced, masks = self.irlr(video_emb, query_tokens, text_emb, query_mask)

        # 2. SnAG head expects (B, C, T) and mask (B, 1, T)
        F_transposed = F_enhanced.transpose(1, 2)             # (B, C, T)
        video_mask_3d = video_mask.unsqueeze(1).float()       # (B, 1, T)

        # 3. Pyramid prediction
        cls_logits, reg_offsets = self.snag_head(F_transposed, video_mask_3d)

        return cls_logits, reg_offsets, masks
