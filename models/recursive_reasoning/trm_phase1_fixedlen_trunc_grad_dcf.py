import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
# your existing utilities
from models.layers import SwiGLU, Attention, CastedLinear, CastedEmbedding, rms_norm
from models.common import trunc_normal_init_
from models.multimodal_block import TRMBlock
import numpy as np
from models.decafnet_head import PyramidBoundaryPredictor
#from utils.debug import summarize_hist
from models.trm_config import TRMLocalizerConfig
from models.blocks import sinusoid_encoding

@dataclass
class VideoTRM_InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class VideoTRM_Carry:
    inner_carry: VideoTRM_InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]
    prev_loss: torch.Tensor = None # added for Q_head training, default for back-compatability
    prev_iou: torch.Tensor = None

class HaltingHead(nn.Module):
    def __init__(self, latent_dim, prior_prob=0.001):
        super().__init__()
        
        # Output 2 values: [Halt_Logit, Continue_Logit]
        self.proj = nn.Linear(latent_dim, 2)
        
        # Initialization logic for ACT (Bias the model to NOT halt initially)
        # We want: Halt_Prob low, Continue_Prob high.
        # So: bias[0] (Halt) should be negative, bias[1] (Continue) should be positive.
        
        bias_init = -np.log((1 - prior_prob) / prior_prob) # approx -4.59 for p=0.01
        
        with torch.no_grad():
            self.proj.weight.zero_() # Zero weights for stability at start
            # Set bias: Index 0 = Halt (-5.0), Index 1 = Continue (+5.0)
            self.proj.bias[0] = bias_init   # Hard to halt
            self.proj.bias[1] = -bias_init  # Easy to continue

    def forward(self, z):
        # z shape: [Batch, Channel, Time] -> Pool over Time -> [Batch, Channel]
        z_pool = torch.mean(z, dim=2)
        
        # Output: [Batch, 2]
        return self.proj(z_pool)

# ---------------------------
# Cross-attention (y/z -> context)
# ---------------------------
class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, causal=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.output_size = hidden_size

        self.q_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.k_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.v_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.o_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.causal = causal

    def forward(self, hidden_states, context, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        ctx_len = context.shape[1]

        # hidden_states: (B,S,D), context: (B,C,D), attention_mask: (B, 1, 1, C)
        query = self.q_proj(hidden_states)
        key = self.k_proj(context)
        value = self.v_proj(context)

        # Reshape for Attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, ctx_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, ctx_len, self.num_heads, self.head_dim)

        # Transpose for SDPA: [B, H, S, D]
        query, key, value = map(lambda t: t.transpose(1, 2), (query, key, value))

        # Scaled Dot Product Attention
        # Note: If mask is boolean, True=Attend, False=Mask. 
        # If float, 0=Attend, -inf=Mask.
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=query, 
            key=key, 
            value=value, 
            attn_mask=attention_mask, 
            is_causal=self.causal
        )

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)
        
        return self.o_proj(attn_output)


#----------------------------
# Reasoning Module
#----------------------------
class VideoTRM_ReasoningModule(nn.Module):
    def __init__(self, layers: List[TRMBlock]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, vid_curr: torch.Tensor,
                vid_orig: torch.Tensor,
                vid_mask: torch.Tensor,
                z_H: torch.Tensor,
                z_L: torch.Tensor, 
                q_orig: torch.Tensor,
                q_mask: torch.Tensor,
                loop_embedding: torch.Tensor,
                **kwargs) -> torch.Tensor:
        for layer in self.layers:
            vid_curr, z_H, z_L = layer(vid_curr=vid_curr, 
                                  vid_orig=vid_orig,
                                  vid_mask=vid_mask,
                                  z_H=z_H,
                                  z_L=z_L,
                                  q_orig=q_orig,
                                  q_mask=q_mask,
                                  loop_embedding=loop_embedding
                                  )
        return vid_curr, z_H, z_L


class TRMLocalizerSync(nn.Module):
    """
    Sync TRM:
      for each L step: (y,z) := Cell(y,z, context)
    Localization uses y tokens via a selectable head (no mean pooling over y).
    """
    def __init__(self, cfg: TRMLocalizerConfig):
        super().__init__()
        self.cfg = cfg

        self.video_proj = CastedLinear(cfg.video_in_dim, cfg.hidden_size, bias=True)
        self.query_proj = CastedLinear(cfg.query_in_dim, cfg.hidden_size, bias=True)

        self.text_pos_emb = CastedEmbedding(cfg.max_text_len, cfg.hidden_size, init_std=0.02, cast_to=torch.float32)
        self.video_pos_emb = CastedEmbedding(cfg.max_frames, cfg.hidden_size, init_std=0.02, cast_to=torch.float32)

        # Reasoning Layer
        self.L_level = VideoTRM_ReasoningModule(layers = [TRMBlock(self.cfg) for _ in range(cfg.L_layers)])

        self.forward_dtype = getattr(torch, self.cfg.forward_dtype)

        #nn.Buffer compatible with torch 2.10 which is not compatible with torchtext 0.18.0 and torch 2.3.0
        # we use custom function that detects torch version and select buffer registration method
        # --- Buffers (FIXED SHAPES) ---
        # CRITICAL FIX: Initialize as (Hidden, Tokens) to match (Channel, Time) format
        self.register_buffer("H_init", torch.empty(cfg.hidden_size, cfg.num_y_tokens))
        self.register_buffer("L_init", torch.empty(cfg.hidden_size, cfg.num_z_tokens))
        # register_compat_buffer(self, "H_init", torch.empty(cfg.num_y_tokens, cfg.hidden_size),)
        # register_compat_buffer(self, "L_init", torch.empty(cfg.num_z_tokens, cfg.hidden_size),)
        trunc_normal_init_(self.H_init, std=0.02)
        trunc_normal_init_(self.L_init, std=0.02)

        # --- Localization Head (Updated) ---
        if cfg.loc_head == "dense_head": # or whatever flag you use
            self.loc_head = PyramidBoundaryPredictor(
                embd_dim=cfg.hidden_size,
                num_levels=3,  # e.g., 3
                n_layers=2        # e.g., 2
            )
        else:
            # Fallback or error
            raise ValueError(f"Unknown loc_head: {cfg.loc_head}")

        # self.q_head = CastedLinear(cfg.hidden_size, 2, bias=True)
        self.q_head = HaltingHead(cfg.hidden_size)
        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        # with torch.no_grad():
        #     self.q_head.weight.zero_()
        #     self.q_head.bias.fill_(-5)  # type: ignore
        
        # NEW: Create the time-step embedding
        # H cycles + 1 to be safe (e.g., if you run 0 to 8)
        self.step_embedding = nn.Embedding(cfg.halt_max_steps, cfg.hidden_size)
        pe = sinusoid_encoding(cfg.max_frames, cfg.hidden_size // 2) # [C, T]
        pe /= cfg.hidden_size ** 0.5 # Scale down
        self.register_buffer('video_pe', pe, persistent=False)

    def _prep_inputs(self, video_emb, text_emb, frame_mask=None, query_mask=None):
        """
        video_emb: (B,T,Dv)
        text_emb: (B,Q,Dq) or (B,Dq)
        frame_mask:   (B,T) optional
        query_mask:   (B,Q) optional
        """
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(1)

        v = self.video_proj(video_emb)    # (B,T,D)
        q = self.query_proj(text_emb)    # (B,Q,D)

        B, T, D = v.shape
        _, T_text, _ = q.shape
        if T > self.cfg.max_frames:
            v = v[:, :self.cfg.max_frames]
            T = self.cfg.max_frames
            if frame_mask is not None:
                frame_mask = frame_mask[:, :T]
        if T_text > self.cfg.max_text_len:
            q = q[:, :self.cfg.max_text_len]
            T_text = self.cfg.max_text_len
            if query_mask is not None:
                query_mask = query_mask[:, :T_text]

        # Add Positional Embeddings
        # Create range [0, 1, ... T-1]
        t_ids = torch.arange(T_text, device=q.device)
        
        # Add embeddings (casted automatically by CastedEmbedding)
        q = q + self.text_pos_emb(t_ids)
        # 2. Video (Sinusoidal)
        # v is [B, T, D], but pe is [D, Max_T]
        pe = self.video_pe.to(v.dtype) # [D, Max_T]
        
        # Handle length mismatch (Interpolate if T > Max_T, slice if T < Max_T)
        if T > pe.shape[1]:
            # Interpolate: Unsqueeze to [1, D, Max_T] -> Interpolate -> Squeeze
            pe = F.interpolate(
                pe.unsqueeze(0), size=T, mode='linear', align_corners=True
            )[0]
        
        # Slice to current length T
        pe_slice = pe[:, :T] # [D, T]
        
        # Add to v (Need to transpose pe to [T, D] to match v)
        v = v + pe_slice.transpose(0, 1)

        # 4. CRITICAL FIX: Transpose to (B, C, T) for DeCafNet Blocks
        v = v.transpose(1, 2) # [B, D, T]
        q = q.transpose(1, 2) # [B, D, Q]
        return v, q, T, frame_mask

    def empty_carry(self, batch_size:int, device: torch.device)-> VideoTRM_InnerCarry:
        return VideoTRM_InnerCarry(
            z_H = torch.empty(batch_size,  self.cfg.hidden_size, self.cfg.num_y_tokens,dtype=self.forward_dtype, device=device),
            z_L = torch.empty(batch_size,  self.cfg.hidden_size, self.cfg.num_z_tokens,dtype=self.forward_dtype, device=device))
        
    
    def reset_carry(self, reset_flag: torch.Tensor, carry: VideoTRM_InnerCarry):
        return VideoTRM_InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: VideoTRM_InnerCarry, batch: Dict[str, torch.Tensor], steps=None):
        # Unpack batch
        video_emb = batch["video_emb"]
        text_emb = batch["text_emb"]
        frame_mask = batch["video_mask"]
        query_mask = batch["query_mask"]

        v, q, T, mask = self._prep_inputs(
            video_emb, text_emb, frame_mask, query_mask
        )

        # Ensure mask is (B, 1, T)
        if mask is not None and mask.dim() == 2:
            mask_expanded = mask.unsqueeze(1) 
        else:
            mask_expanded = mask
        
        # Ensure query mask is (B, 1, Q)
        if query_mask is not None and query_mask.dim() == 2:
            q_mask_expanded = query_mask.unsqueeze(1)
        else:
            q_mask_expanded = query_mask

        debug = True
        zH_hist, zL_hist = [], []

        z_H, z_L = carry.z_H, carry.z_L
        # --- 1. INJECT STEP EMBEDDING HERE ---
        # if steps is not None:
        #     if (steps != steps[0]).any():
        #         print(f"Mixed Steps in Batch: {steps}")
        #     # Clamp steps to fit in embedding table (0 to 8)
        #     current_step_idx = steps.clamp(max=self.cfg.halt_max_steps).long()
        
        #     # Get the Time Signal
        #     # Shape: (B, Hidden) -> (B, 1, Hidden) to broadcast across tokens
        #     time_signal = self.step_embedding(current_step_idx).unsqueeze(1)
        #     z_H = z_H + time_signal
        if steps is not None:
            current_step_idx = steps.clamp(max=self.cfg.halt_max_steps).long()
            time_signal = self.step_embedding(current_step_idx).unsqueeze(2)
        def snap():
            if debug:
                zH_hist.append(z_H.detach().float().cpu())
                zL_hist.append(z_L.detach().float().cpu())
        vid_curr = v.clone()
        device = z_H.device
        snap()
        for _h in range(self.cfg.H_cycles):
            # --- FIX: Convert int to Tensor ---
            h_idx = torch.tensor([_h], device=device) # Shape: [1]
            
            # --- FIX: Lookup & Reshape for Broadcasting ---
            # Embedding: [1, Hidden] -> Unsqueeze: [1, Hidden, 1]
            # This allows it to add to z_H which is [Batch, Hidden, Tokens]
            #time_signal = self.step_embedding(h_idx).unsqueeze(2)
            vid_curr, z_H, z_L = self.L_level(vid_curr=vid_curr,
                               vid_orig=v,
                               vid_mask=mask_expanded,
                               z_H=z_H,
                                z_L=z_L,
                                q_orig=q,
                                q_mask=q_mask_expanded,
                                loop_embedding=time_signal
                               )
            snap()
        # 4. Localization Prediction (Pyramid)
        time_logits, reg_offsets = self.loc_head(vid_curr, mask_expanded)

        # 5. Halting Prediction
        q_logits = self.q_head(z_H).to(torch.float32)

        new_carry = VideoTRM_InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        # ---- Localization Prediction ----
        
        debug_stats = {}
        # if debug and len(zH_hist) > 1:
        #     debug_stats["zH"] = summarize_hist(zH_hist)
        #     debug_stats["zL"] = summarize_hist(zL_hist)

        return new_carry, time_logits, reg_offsets, (q_logits[..., 0], q_logits[...,1]), debug_stats

class Video_TRM_ACT(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TRMLocalizerConfig(**config_dict)
        self.inner = TRMLocalizerSync(self.config)


    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["video_emb"].shape[0]
        device = batch["video_emb"].device
        return VideoTRM_Carry(
            inner_carry=self.inner.empty_carry(batch_size, device=device),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
            prev_loss=torch.full((batch_size,), float('inf'), dtype=torch.float32, device = device),
            prev_iou=torch.full((batch_size,), float(0), dtype=torch.float32, device = device)
        )
        
    def forward(self, carry: VideoTRM_Carry, batch: Dict[str, torch.Tensor]) -> Tuple[VideoTRM_Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        if self.training:
            halted_early = (new_steps == 0) & (carry.steps > 0) & (carry.steps < self.config.halt_max_steps-1)
            if halted_early.any():
                print('Halted early! After steps', carry.steps[halted_early].tolist())

        new_prev_loss = torch.where(
            carry.halted,
            # torch.full_like(carry.prev_loss, float('inf')),
            torch.zeros_like(carry.prev_loss),
            carry.prev_loss,
        )

        new_prev_iou = torch.where(
            carry.halted,
            # torch.full_like(carry.prev_loss, float('inf')),
            torch.zeros_like(carry.prev_iou),
            carry.prev_iou,
        )

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, extra_logits, (q_halt_logits, q_continue_logits), debug_stats = self.inner(new_inner_carry, new_current_data, new_steps)

        outputs = {
            "logits": logits,
            "extra_logits": extra_logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "debug_stats": debug_stats
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return VideoTRM_Carry(new_inner_carry, new_steps, halted, new_current_data, new_prev_loss, new_prev_iou), outputs
