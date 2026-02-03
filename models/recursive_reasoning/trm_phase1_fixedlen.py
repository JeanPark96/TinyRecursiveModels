import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
# your existing utilities
from models.layers import SwiGLU, Attention, CastedLinear, CastedEmbedding, rms_norm
from models.common import trunc_normal_init_
import numpy as np
from models.head import TRM_DenseHead

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


# ---------------------------
# Config
# ---------------------------
@dataclass
class TRMLocalizerConfig:
    hidden_size: int = 768
    num_heads: int = 4
    expansion: float = 4.0
    norm_eps: float = 1e-5

    H_cycles: int = 3
    L_cycles: int = 4
    L_layers: int = 2
    max_frames: int = 256
    max_text_len: int = 77

    num_y_tokens: int = 4
    num_z_tokens: int = 4

    halt_max_steps: int = 8
    halt_exploration_prob: float = 0.0
    no_ACT_continue: bool = True

    forward_dtype: str = "bfloat32"

    video_in_dim: int = 512
    query_in_dim: int = 512

    batch_size: int = 16

    # choose localization head
    # "token_mixture" | "dot_product" | "attn_pool"
    loc_head: str = "dense_head"

    include_query_in_context: bool = True   # usually helpful for stability
    add_temporal_pos: bool = True           # set False if your encoder already has temporal pos



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


# ---------------------------
# TRM block
# ---------------------------
class VideoTRMBlock(nn.Module):
    def __init__(self, config: TRMLocalizerConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_heads
        self.self_attn = Attention(config.hidden_size, self.head_dim, config.num_heads, config.num_heads, causal=False)
        self.cross_attn = CrossAttention(config.hidden_size, config.num_heads, causal=False)
        self.mlp = SwiGLU(config.hidden_size, config.expansion)
        self.norm_eps = config.norm_eps

    def forward(self, hidden_states, context_tokens, cos_sin=None, attention_mask=None):
        h = self.self_attn(cos_sin, hidden_states)
        hidden_states = rms_norm(hidden_states + h, self.norm_eps)

        h = self.cross_attn(hidden_states, context_tokens, attention_mask=attention_mask)
        hidden_states = rms_norm(hidden_states + h, self.norm_eps)

        h = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + h, self.norm_eps)
        return hidden_states

#----------------------------
# Reasoning Module
#----------------------------
class VideoTRM_ReasoningModule(nn.Module):
    def __init__(self, layers: List[VideoTRMBlock]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, context_tokens=input_injection, **kwargs)
        return hidden_states




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

        self.modality_emb = nn.Embedding(2, cfg.hidden_size)
        # Initialize small to avoid disrupting pre-trained feature distribution
        nn.init.normal_(self.modality_emb.weight, std=0.02)

        self.text_pos_emb = CastedEmbedding(cfg.max_text_len, cfg.hidden_size, init_std=0.02, cast_to=torch.float32)
        self.video_pos_emb = CastedEmbedding(cfg.max_frames, cfg.hidden_size, init_std=0.02, cast_to=torch.float32)

        # Reasoning Layer
        self.L_level = VideoTRM_ReasoningModule(layers = [VideoTRMBlock(self.cfg) for _ in range(cfg.L_layers)])

        self.forward_dtype = getattr(torch, self.cfg.forward_dtype)

        self.H_init = nn.Buffer(torch.empty(cfg.num_y_tokens, cfg.hidden_size))
        self.L_init = nn.Buffer(torch.empty(cfg.num_z_tokens, cfg.hidden_size))
        trunc_normal_init_(self.H_init, std=0.02)
        trunc_normal_init_(self.L_init, std=0.02)

        if cfg.loc_head == "dense_head" or cfg.loc_head == "soft_nms":
            self.loc_head = TRM_DenseHead(cfg.hidden_size)
        else:
            raise ValueError(f"Unknown loc_head: {cfg.loc_head}")

        self.q_head = CastedLinear(cfg.hidden_size, 2, bias=True)
        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

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

        video_type_emb = self.modality_emb(torch.tensor(0, device=v.device))
        text_type_emb = self.modality_emb(torch.tensor(1, device=q.device))
        v = v + video_type_emb
        q = q + text_type_emb

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
        v_ids = torch.arange(T, device=v.device)
        
        # Add embeddings (casted automatically by CastedEmbedding)
        q = q + self.text_pos_emb(t_ids)
        v = v + self.video_pos_emb(v_ids)

        # 4. Concatenate
        # Result: [Batch, T_len + V_len, Hidden]
        if self.cfg.include_query_in_context:
            context = torch.cat([q, v], dim=1)          # (B,Q+T,D)
            total_len = T + T_text
            mask = torch.cat([query_mask, frame_mask], dim=1)
        else:
            context = v                                  # (B,T,D)
            total_len = T
            mask = frame_mask

        return v, q, context, total_len, mask

    def empty_carry(self, batch_size:int, device: torch.device)-> VideoTRM_InnerCarry:
        return VideoTRM_InnerCarry(
            z_H = torch.empty(batch_size, self.cfg.num_y_tokens, self.cfg.hidden_size, dtype=self.forward_dtype, device=device),
            z_L = torch.empty(batch_size, self.cfg.num_z_tokens, self.cfg.hidden_size, dtype=self.forward_dtype, device=device))
        
    
    def reset_carry(self, reset_flag: torch.Tensor, carry: VideoTRM_InnerCarry):
        return VideoTRM_InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: VideoTRM_InnerCarry, batch: Dict[str, torch.Tensor]):
        # Unpack batch
        video_emb = batch["video_emb"]
        text_emb = batch["text_emb"]
        frame_mask = batch["video_mask"]
        query_mask = batch["query_mask"]

        v, q, context, T, mask = self._prep_inputs(
            video_emb, text_emb, frame_mask, query_mask
        )

        # --- MASK PREPARATION ---
        # mask is (B, T_total). 1=Keep, 0=Mask.
        # SDPA requires shape broadcasting to (B, NumHeads, Q_Len, KV_Len).
        # We transform to (B, 1, 1, KV_Len) which broadcasts correctly.
        attention_mask = None
        if mask is not None:
            # Expand dims: (B, 1, 1, T)
            attention_mask = mask[:, None, None, :].bool()

        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
            attention_mask=attention_mask # Pack mask here
        )
        # Final H update (Cross attn to z_L)
        # NOTE: When z_H reads z_L, the 'context' is z_L. 
        # z_L is fixed size (no padding), so we should theoretically DISABLE the mask here 
        # or pass a full-ones mask.
        seq_info_for_h = seq_info.copy()
        seq_info_for_h['attention_mask'] = None # No mask needed for z_L
        z_H, z_L = carry.z_H, carry.z_L

        # with torch.no_grad():
        for _h in range(self.cfg.H_cycles - 1):
            for _ in range(self.cfg.L_cycles):
                z_L_input = z_L + z_H
                z_L = self.L_level(z_L_input, context, **seq_info)
            z_H = self.L_level(z_H, z_L, **seq_info_for_h)

        # last cycle with grad
        for _ in range(self.cfg.L_cycles):
            z_L_input = z_L + z_H
            z_L = self.L_level(z_L_input, context, **seq_info)
        
        z_H = self.L_level(z_H, z_L, **seq_info_for_h)

        new_carry = VideoTRM_InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        # ---- Localization Prediction ----
        # We will pack outputs into a dictionary 'preds'
        reg_offsets = None
        
        if isinstance(self.loc_head, TRM_DenseHead):
            # Input: z_H (32 tokens), v (256 tokens)
            #print("TRM_DenseHead")
            time_logits, reg_offsets = self.loc_head(z_H, v, frame_mask)
        else:
            time_logits = self.loc_head(z_H, T)              # (B,T)

            raise ValueError(f"Unknown loc_head: {cfg.loc_head}")
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, time_logits, reg_offsets, (q_logits[..., 0], q_logits[...,1])

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
            prev_loss=torch.full((batch_size,), float('inf'), dtype=torch.float32, device = device)
        )
        
    def forward(self, carry: VideoTRM_Carry, batch: Dict[str, torch.Tensor]) -> Tuple[VideoTRM_Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_prev_loss = torch.where(
            carry.halted,
            torch.full_like(carry.prev_loss, float('inf')),
            carry.prev_loss,
        )
        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, extra_logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "extra_logits": extra_logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
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

        return VideoTRM_Carry(new_inner_carry, new_steps, halted, new_current_data, new_prev_loss), outputs
