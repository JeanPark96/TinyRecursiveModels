from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin,
    CastedEmbedding, CastedLinear
)


# =========================
# Carries
# =========================

@dataclass
class TRM_ACT_NuScenes_InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TRM_ACT_NuScenes_Carry:
    inner_carry: TRM_ACT_NuScenes_InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


# =========================
# Config
# =========================

class TRM_ACT_NuScenes_Config(BaseModel):
    batch_size: int

    n_history: int              # T_obs
    max_obstacles: int          # A
    n_horizon: int              # T_pred

    in_dim: int = 7
    out_dim: int = 2            # default: predict (x,y). set 7 to predict all 7 dims
    out_slice: int = 2          # how many dims to supervise from targets (2 -> x,y). If out_dim=7, set out_slice=7

    predict_delta: bool = True  # predict deltas for first out_slice dims

    # global token (kept)
    global_len: int = 1         # keep as 1
    
    seq_len: int = 360            # computed: A * history_len

    # recursion
    H_cycles: int
    L_cycles: int
    H_layers: int  # ignored
    L_layers: int

    # transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str          # "rope" | "learned" | "none"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # halting (ACT)
    halt_max_steps: int
    halt_exploration_prob: float = 0.0
    no_ACT_continue: bool = True

    forward_dtype: str = "bfloat16"
    mlp_t: bool = False


# =========================
# Block / Reasoning Module
# =========================

class TRM_ACT_NuScenes_Block(nn.Module):
    def __init__(self, config: TRM_ACT_NuScenes_Config) -> None:
        super().__init__()
        self.config = config
        self.norm_eps = config.rms_norm_eps

        if self.config.mlp_t:
            self.mlp_t = SwiGLU(hidden_size=self.config.seq_len, expansion=config.expansion)
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )

        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)

    def forward(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,  # [B, L_total] boolean mask
    ) -> torch.Tensor:
        if self.config.mlp_t:
            hs = hidden_states.transpose(1, 2)
            out = self.mlp_t(hs)
            hs = rms_norm(hs + out, variance_epsilon=self.norm_eps)
            hidden_states = hs.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps,
            )

        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)

        # pragmatic padding containment if Attention has no mask support
        if token_mask is not None:
            hidden_states = hidden_states * token_mask[..., None].to(hidden_states.dtype)

        return hidden_states


class TRM_ACT_NuScenes_ReasoningModule(nn.Module):
    def __init__(self, layers: List[TRM_ACT_NuScenes_Block]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


# =========================
# Inner: obs_pose/obs_mask -> future trajectory
# =========================

class TRM_ACT_NuScenes_Inner(nn.Module):
    def __init__(self, config: TRM_ACT_NuScenes_Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # tokens = A * T_obs (flattened agent-by-time)
        self.seq_len = self.config.max_obstacles * self.config.n_history
        self.config.seq_len = self.seq_len  # for parity
        self.global_len = self.config.global_len

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Continuous input projection
        self.embed_inputs = CastedLinear(self.config.in_dim, self.config.hidden_size, bias=True)

        # Agent & time embeddings
        self.embed_agent = CastedEmbedding(
            self.config.max_obstacles, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )
        self.embed_time = CastedEmbedding(
            self.config.n_history, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )

        # Global token (learned)
        gt = trunc_normal_init_(torch.empty(1, self.global_len, self.config.hidden_size, dtype=self.forward_dtype), std=1.0)
        self.global_token = nn.Parameter(gt)

        # Output head: per-agent predictions over horizon
        self.pred_head = CastedLinear(self.config.hidden_size, self.config.n_horizon * self.config.out_dim, bias=True)

        # ACT Q-head (optional; uses global token at position 0)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # Pos encodings over (global + tokens)
        total_len = self.global_len + self.seq_len
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=total_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(total_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

        # Reasoning layers
        self.L_level = TRM_ACT_NuScenes_ReasoningModule(
            layers=[TRM_ACT_NuScenes_Block(self.config) for _ in range(self.config.L_layers)]
        )

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(
        self,
        obs_pose: torch.Tensor,   # [B, T_obs, A, 7]
        obs_mask: torch.Tensor,   # [B, T_obs, A] (1 valid, 0 pad)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, A, Fdim = obs_pose.shape
        assert T == self.config.n_history and A == self.config.max_obstacles and Fdim == self.config.in_dim

        # Project
        x = obs_pose.to(self.forward_dtype)
        emb = self.embed_inputs(x)  # [B, T, A, D]

        # Add time/agent embeddings (broadcast)
        time_ids = torch.arange(T, device=obs_pose.device)
        agent_ids = torch.arange(A, device=obs_pose.device)
        emb = emb + self.embed_time(time_ids)[None, :, None, :] + self.embed_agent(agent_ids)[None, None, :, :]

        # Flatten to [B, A*T, D] with token index = agent*T + t
        emb = emb.permute(0, 2, 1, 3).contiguous().view(B, A * T, self.config.hidden_size)

        token_mask = obs_mask.to(torch.bool).permute(0, 2, 1).contiguous().view(B, A * T)  # [B, A*T]
        emb = emb * token_mask[..., None].to(emb.dtype)

        # Prepend learned global token(s)
        global_tok = self.global_token.expand(B, -1, -1)  # [B, global_len, D]
        emb = torch.cat([global_tok, emb], dim=1)         # [B, global+tokens, D]

        # Full mask: global positions are always valid
        global_mask = torch.ones((B, self.global_len), device=token_mask.device, dtype=torch.bool)
        full_mask = torch.cat([global_mask, token_mask], dim=1)  # [B, global+tokens]

        # learned abs pos (optional)
        if self.config.pos_encodings == "learned":
            emb = 0.707106781 * (emb + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # last observed pose per agent for delta decoding
        last_obs = obs_pose[:, -1].to(self.forward_dtype)  # [B, A, 7]

        return self.embed_scale * emb, full_mask, last_obs

    def empty_carry(self, batch_size: int, device: torch.device) -> TRM_ACT_NuScenes_InnerCarry:
        total_len = self.global_len + self.seq_len
        return TRM_ACT_NuScenes_InnerCarry(
            z_H=torch.empty(batch_size, total_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, total_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TRM_ACT_NuScenes_InnerCarry) -> TRM_ACT_NuScenes_InnerCarry:
        return TRM_ACT_NuScenes_InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self,
        carry: TRM_ACT_NuScenes_InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TRM_ACT_NuScenes_InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None

        input_embeddings, full_mask, last_obs = self._input_embeddings(
            batch["obs_pose"], batch["obs_mask"]
        )

        z_H, z_L = carry.z_H, carry.z_L

        # H_cycles-1 without grad
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin, token_mask=full_mask)
                z_H = self.L_level(z_H, z_L, cos_sin=cos_sin, token_mask=full_mask)

        # 1 with grad
        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin, token_mask=full_mask)
        z_H = self.L_level(z_H, z_L, cos_sin=cos_sin, token_mask=full_mask)

        new_carry = TRM_ACT_NuScenes_InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())

        # global latent (kept)
        global_latent = z_H[:, 0]  # [B, D]

        # Q head on global token
        q_logits = self.q_head(global_latent).to(torch.float32)
        q_halt_logits, q_continue_logits = q_logits[..., 0], q_logits[..., 1]

        # Per-agent summaries: use each agent's last history token (t=T-1)
        seq_states = z_H[:, self.global_len:]  # [B, A*T, D]
        B = seq_states.shape[0]
        A = self.config.max_obstacles
        T = self.config.n_history
        D = seq_states.shape[-1]

        last_idx = (torch.arange(A, device=seq_states.device) * T + (T - 1))  # [A]
        idx = last_idx[None, :].expand(B, A)  # [B, A]
        agent_states = torch.gather(seq_states, 1, idx[..., None].expand(B, A, D))  # [B, A, D]

        pred = self.pred_head(agent_states).view(B, A, self.config.n_horizon, self.config.out_dim)

        # delta decoding for first out_slice dims
        if self.config.predict_delta and self.config.out_slice > 0:
            base = last_obs[:, :, : self.config.out_slice].to(pred.dtype)  # [B, A, out_slice]
            pred_slice = pred[..., : self.config.out_slice]
            pred = torch.cat([base[:, :, None, :] + pred_slice, pred[..., self.config.out_slice :]], dim=-1)

        # zero-out agents that don't exist at last obs step
        last_step_mask = batch["obs_mask"][:, -1].to(torch.bool)  # [B, A]
        pred = pred * last_step_mask[:, :, None, None].to(pred.dtype)

        return new_carry, pred, (q_halt_logits, q_continue_logits), global_latent


# =========================
# ACT Wrapper
# =========================

class TRM_ACT_NuScenes(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TRM_ACT_NuScenes_Config(**config_dict)
        self.inner = TRM_ACT_NuScenes_Inner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TRM_ACT_NuScenes_Carry:
        B = batch["obs_pose"].shape[0]
        device = batch["obs_pose"].device
        return TRM_ACT_NuScenes_Carry(
            inner_carry=self.inner.empty_carry(B, device=device),
            steps=torch.zeros((B,), dtype=torch.int32, device=device),
            halted=torch.ones((B,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: TRM_ACT_NuScenes_Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TRM_ACT_NuScenes_Carry, Dict[str, torch.Tensor]]:

        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)

        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
        }

        new_inner_carry, pred, (q_halt_logits, q_continue_logits), global_latent = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {
            "pred": pred,                        # [B, A, H, out_dim]
            "global_latent": global_latent,      # [B, D]
            "q_halt_logits": q_halt_logits,      # [B]
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                if self.config.halt_exploration_prob > 0:
                    explore = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    min_halt = torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                    min_halt = torch.where(explore, min_halt, torch.zeros_like(min_halt))
                    halted = halted & (new_steps >= min_halt)

        new_carry = TRM_ACT_NuScenes_Carry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=new_current_data,
        )
        return new_carry, outputs


# =========================
# Loss (matches your targets/targets_mask)
# =========================

def traj_loss_smooth_l1(
    pred: torch.Tensor,            # [B, A, H, out_dim]
    targets: torch.Tensor,         # [B, H, A, 7]
    targets_mask: torch.Tensor,    # [B, H, A]
    out_slice: int = 2,
) -> torch.Tensor:
    tgt = targets[..., :out_slice].permute(0, 2, 1, 3).contiguous()      # [B, A, H, out_slice]
    m = targets_mask.permute(0, 2, 1).to(torch.float32)                  # [B, A, H]
    pred_slice = pred[..., :out_slice]
    loss = F.smooth_l1_loss(pred_slice, tgt, reduction="none").sum(-1)   # [B,A,H]
    return (loss * m).sum() / (m.sum() + 1e-6)
