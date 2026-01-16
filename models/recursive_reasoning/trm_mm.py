from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from pydantic import BaseModel
import random

# Import existing layers from your TRM codebase
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, CastedLinear, CastedEmbedding
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Config, 
    TinyRecursiveReasoningModel_ACTV1Block,
    TinyRecursiveReasoningModel_ACTV1InnerCarry
)

# --- 1. Helper Classes ---

class Learnable2DPositionalEncoding(nn.Module):
    def __init__(self, dim, height, width):
        super().__init__()
        self.row_embed = nn.Embedding(height, dim // 2)
        self.col_embed = nn.Embedding(width, dim // 2)
        
    def forward(self, x):
        # x: [Batch, Dim, H, W]
        B, C, H, W = x.shape
        i = torch.arange(W, device=x.device)
        j = torch.arange(H, device=x.device)
        
        # Expand dims to match batch/channel structure
        x_emb = self.col_embed(i).unsqueeze(0).repeat(H, 1, 1) # [H, W, D/2]
        y_emb = self.row_embed(j).unsqueeze(1).repeat(1, W, 1) # [H, W, D/2]
        
        # Concatenate and reshape to [1, Dim, H, W]
        pos = torch.cat([x_emb, y_emb], dim=-1).permute(2, 0, 1).unsqueeze(0) 
        return x + pos

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)
        
        # Keep layers up to the final Conv block (removes avgpool and fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze early layers for stability
        for param in self.backbone[:6].parameters():
            param.requires_grad = False
            
        # Project 512 channels to hidden_size
        self.projection = nn.Conv2d(512, output_dim, kernel_size=1)
        self.preprocess = weights.transforms()

    def forward(self, x):
        # x: [Batch, 3, H, W]
        features = self.backbone(x) 
        return self.projection(features) 

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.kv_proj = CastedLinear(hidden_size, hidden_size * 2, bias=False)
        self.o_proj = CastedLinear(hidden_size, hidden_size, bias=False)

    def forward(self, x_q, x_kv):
        B, N_Q, _ = x_q.shape
        B, N_KV, _ = x_kv.shape

        q = self.q_proj(x_q).view(B, N_Q, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x_kv).view(B, N_KV, self.num_heads, 2 * self.head_dim).transpose(1, 2)
        k, v = kv.chunk(2, dim=-1)

        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N_Q, -1)
        return self.o_proj(attn_output)

# --- 2. Main AV Module ---
class TinyRecursiveReasoningModel_AV_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config, 
                 hist_len: int, future_len: int, img_channels: int, img_size: Tuple[int, int]):
        super().__init__()
        self.config = config
        
        # --- Configurable Dimensions ---
        # NuScenes pose has 7 dims (x, y, z, w, l, h, theta)
        AGENT_INPUT_DIM = 7 
        
        # --- Encoders ---
        self.agent_encoder = nn.Sequential(
            # Flatten: hist_len * 7
            CastedLinear(hist_len * AGENT_INPUT_DIM, config.hidden_size, bias=True),
            nn.SiLU(),
            CastedLinear(config.hidden_size, config.hidden_size, bias=True)
        )

        self.image_encoder = ResNetFeatureExtractor(output_dim=config.hidden_size)
        
        # Calculate ResNet feature map size (224 / 32 = 7)
        feat_H, feat_W = img_size[0] // 32, img_size[1] // 32
        self.image_pos_enc = Learnable2DPositionalEncoding(config.hidden_size, feat_H, feat_W)

        # --- Fusion & TRM ---
        self.cross_attention = CrossAttention(config.hidden_size, config.num_heads)
        self.fusion_norm = nn.LayerNorm(config.hidden_size)

        self.L_level = nn.ModuleList([
            TinyRecursiveReasoningModel_ACTV1Block(config) 
            for _ in range(config.L_layers)
        ])
        
        self.H_init = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.L_init = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # Output Head: Predicts (x,y) for future_len steps
        self.pred_head = CastedLinear(config.hidden_size, future_len * 2, bias=True)

    def _run_reasoning_layers(self, hidden_states, input_injection):
        curr = hidden_states
        seq_info = dict(cos_sin=None) 
        for layer in self.L_level:
            curr = layer(hidden_states=curr + input_injection, **seq_info)
        return curr

    def _input_embeddings(self, obs_pose, obs_mask, image):
        # obs_pose: [Batch, Agents, Hist, 7]
        # obs_mask: [Batch, Agents, Hist]
        # image:    [Batch, 3, H, W] (Current Frame)

        B, N, T, C = obs_pose.shape

        # 1. Encode Agents
        # Flatten history: [B, N, T, 7] -> [B, N, T*7]
        agent_input = obs_pose.reshape(B, N, -1)
        agent_feats = self.agent_encoder(agent_input)

        # 2. Apply Masking
        # Determine if agent is valid at the CURRENT time (last step of history)
        # obs_mask is 1.0 for valid, 0.0 for invalid
        # We look at the last time step T-1
        valid_agent_mask = obs_mask[:, :, -1].unsqueeze(-1) # [B, N, 1]
        
        # Zero out features for invalid agents so they don't pollute the TRM
        agent_feats = agent_feats * valid_agent_mask

        # 3. Encode Image
        img_feats = self.image_encoder(image)      # [B, Dim, H/32, W/32]
        img_feats = self.image_pos_enc(img_feats)  # Add 2D PE
        img_seq = img_feats.flatten(2).transpose(1, 2) # [B, Seq, Dim]

        # 4. Cross Attention
        # Valid agents query the image. Invalid agents query too (but result is masked out later)
        fused = self.cross_attention(x_q=agent_feats, x_kv=img_seq)
        
        # Residual + Norm, then re-apply mask to ensure zeroed agents stay zeroed
        fused_input = self.fusion_norm(agent_feats + fused)
        fused_input = fused_input * valid_agent_mask
        
        return fused_input

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, 
                batch: Dict[str, torch.Tensor]):
        
        # Unpack from NuScenesMiniDataset batch keys
        # obs_pose: [Batch, Hist, Agents, 7] -> Permute to [Batch, Agents, Hist, 7]
        obs_pose = batch['obs_pose'].permute(0, 2, 1, 3)
        
        # obs_mask: [Batch, Hist, Agents] -> Permute to [Batch, Agents, Hist]
        obs_mask = batch['obs_mask'].permute(0, 2, 1)

        # camera: [Batch, Hist, 3, H, W] -> Select LAST frame for current context
        # shape becomes [Batch, 3, H, W]
        current_image = batch['camera'][:, -1, ...]

        # Generate Input Injection
        input_injection = self._input_embeddings(obs_pose, obs_mask, current_image)
        
        # --- TRM Logic ---
        z_H, z_L = carry.z_H, carry.z_L
        
        # Initialize if empty
        if z_H.numel() == 0: 
            B, N, _ = input_injection.shape
            z_H = self.H_init.expand(B, N, -1)
            z_L = self.L_init.expand(B, N, -1)

        # 1. Heavy H-cycles (No Grad)
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self._run_reasoning_layers(hidden_states=z_L, input_injection=z_H + input_injection)
                z_H = self._run_reasoning_layers(hidden_states=z_H, input_injection=z_L)

        # 2. Final H-cycle (With Grad)
        for _L_step in range(self.config.L_cycles):
            z_L = self._run_reasoning_layers(hidden_states=z_L, input_injection=z_H + input_injection)
        z_H = self._run_reasoning_layers(hidden_states=z_H, input_injection=z_L)
        
        # Output Prediction
        output_traj = self.pred_head(z_H)
        
        # Reshape to [Batch, Agents, Future, 2]
        output_traj = output_traj.view(input_injection.shape[0], input_injection.shape[1], -1, 2)

        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=z_L)
        return new_carry, output_traj