from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import torchvision.models as models

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm, SwiGLU, Attention_Mask, RotaryEmbedding, CosSin,
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
    prev_loss: torch.Tensor = None # added for Q_head training, default for back-compatability

    visual_context: Optional[torch.Tensor] = None

# =========================
# Config
# =========================

class TRM_ACT_NuScenes_Config(BaseModel):
    batch_size: int
    n_history: int
    max_obstacles: int
    n_horizon: int

    in_dim: int = 7
    out_dim: int = 2
    out_slice: int = 2
    predict_delta: bool = True
    global_len: int = 1
    seq_len: int = 360  # max_obstacles * n_history

    # Recursion
    H_cycles: int
    L_cycles: int
    H_layers: int = 0
    L_layers: int

    # Transformer
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting
    halt_max_steps: int
    halt_exploration_prob: float = 0.0
    no_ACT_continue: bool = True
    forward_dtype: str = "bfloat16"
    mlp_t: bool = False

    # --- Camera Config ---
    # --- Vision Config (No intrinsics needed!) ---
    use_camera: bool = True
    num_cameras: int = 6          # NEW
    cam_names: List[str] = ["F","FL","FR","B","BL","BR"]  # optional, for readability

    cam_in_channels: int = 3
    cam_feat_height: int = 9  # ResNet18 output size for 224x224 input
    cam_feat_width: int = 16


# =========================
# Block & Reasoning Module
# =========================

class TRM_ACT_NuScenes_Block(nn.Module):
    def __init__(self, config: TRM_ACT_NuScenes_Config) -> None:
        super().__init__()
        self.config = config
        self.norm_eps = config.rms_norm_eps

        if self.config.mlp_t:
            self.mlp_t = SwiGLU(hidden_size=self.config.seq_len, expansion=config.expansion)
        else:
            self.self_attn = Attention_Mask(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )
        # 2. Cross Attention (Agent-to-Image) - NEW
        if self.config.use_camera:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.config.hidden_size,
                num_heads=self.config.num_heads,
                batch_first=True,
                dropout=0.0 
            )
            # Gate for cross attention residual
            self.cross_gate = CastedLinear(self.config.hidden_size, self.config.hidden_size, bias=True)
            # --- CRITICAL FIX: Zero Init ---
            # This ensures the model starts by ignoring vision (safe start)
            with torch.no_grad():
                self.cross_gate.weight.zero_()
                if self.cross_gate.bias is not None:
                    self.cross_gate.bias.zero_()
        # 3. MLP
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, visual_context: Optional[torch.Tensor],  # [B, T*H*W+1, D]
                token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Prepare Attention Mask
        # token_mask is usually [B, Seq]. 
        # Attention modules typically expect [B, 1, 1, Seq] or similar for broadcasting.
        attn_mask = None
        if token_mask is not None:
            # Create additive mask: 0.0 for valid, -inf for invalid
            # Assuming token_mask is 1 for valid, 0 for invalid
            mask_float = token_mask.to(hidden_states.dtype)
            attn_mask = (1.0 - mask_float) * torch.finfo(hidden_states.dtype).min
            # Reshape for broadcasting: [B, 1, 1, SeqLen]
            attn_mask = attn_mask[:, None, None, :]

        if self.config.mlp_t:
            hs = hidden_states.transpose(1, 2)
            out = self.mlp_t(hs)
            hs = rms_norm(hs + out, variance_epsilon=self.norm_eps)
            hidden_states = hs.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states, attention_mask=attn_mask),
                variance_epsilon=self.norm_eps,
            )
        # --- B. Cross Attention Phase (Iterative Refinement) ---
        if self.config.use_camera and visual_context is not None:
            # Flatten hidden for attention: [B, Seq, D]
            # visual_context is [B, VisSeq, D]
            # We assume hidden_states has B integrated or is compatible. 
            # Note: visual_context assumes flattened B*T if batching, but here hidden_states is [B, A*T, D]
            
            # Standard MultiheadAttention expects [Batch, Seq, D]
            ca_out, _ = self.cross_attn(
                query=hidden_states,
                key=visual_context,
                value=visual_context
            )
            
            # Gated Residual
            # Post-Norm style: Norm(x + Gate(Attention(x)))
            hidden_states = rms_norm(hidden_states + self.cross_gate(ca_out), variance_epsilon=self.norm_eps)
        
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        if token_mask is not None:
            hidden_states = hidden_states * token_mask[..., None].to(hidden_states.dtype)
        return hidden_states

class TRM_ACT_NuScenes_ReasoningModule(nn.Module):
    def __init__(self, layers: List[TRM_ACT_NuScenes_Block]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, visual_context: Optional[torch.Tensor], token_mask: Optional[torch.Tensor], **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, visual_context=visual_context, token_mask=token_mask, **kwargs)
        return hidden_states


# =========================
# Inner Model
# =========================

class TRM_ACT_NuScenes_Inner(nn.Module):
    def __init__(self, config: TRM_ACT_NuScenes_Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.seq_len = self.config.max_obstacles * self.config.n_history
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # --- Vision Setup ---
        if self.config.use_camera:
            # resnet = models.resnet18(pretrained=True)
            # if self.config.cam_in_channels != 3:
            #     resnet.conv1 = nn.Conv2d(self.config.cam_in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # self.cam_backbone = nn.Sequential(*list(resnet.children())[:-2])
            
            self.vis_proj = CastedLinear(512, self.config.hidden_size, bias=True)
            self.visual_null_token = nn.Parameter(trunc_normal_init_(torch.empty(1, 1, self.config.hidden_size), std=0.02))
            self.vis_norm = nn.LayerNorm(self.config.hidden_size) # <--- ADD THIS
            # Spatial Embeddings (14x14)
            num_patches = self.config.cam_feat_height * self.config.cam_feat_width
            self.vis_pos_emb = nn.Parameter(trunc_normal_init_(torch.empty(1, num_patches, self.config.hidden_size), std=0.02))
            # Note: We will reuse self.embed_time for Temporal Visual Embeddings

        # --- Standard Embeddings ---
        self.embed_inputs = CastedLinear(self.config.in_dim, self.config.hidden_size, bias=True)
        self.embed_agent = CastedEmbedding(self.config.max_obstacles, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.embed_time = CastedEmbedding(self.config.n_history, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.cam_to_id = {name: i for i, name in enumerate(self.config.cam_names)}
        self.embed_cam = CastedEmbedding(len(self.config.cam_names), self.config.hidden_size,
                                        init_std=embed_init_std, cast_to=self.forward_dtype)
        
        gt = trunc_normal_init_(torch.empty(1, self.config.global_len, self.config.hidden_size, dtype=self.forward_dtype), std=1.0)
        self.global_token = nn.Parameter(gt)

        # Decoder & Heads
        self.future_queries = nn.Parameter(trunc_normal_init_(torch.empty(1, 1, self.config.n_horizon, self.config.hidden_size, dtype=self.forward_dtype), std=1.0))
        self.decoder_attn = nn.MultiheadAttention(embed_dim=self.config.hidden_size, num_heads=self.config.num_heads, batch_first=True, dtype=self.forward_dtype)
        self.output_proj = CastedLinear(self.config.hidden_size, self.config.out_dim, bias=True)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # Pos Encoding
        total_len = self.config.global_len + self.seq_len
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads, max_position_embeddings=total_len, base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(total_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

        # Layers
        self.L_level = TRM_ACT_NuScenes_ReasoningModule(layers=[TRM_ACT_NuScenes_Block(self.config) for _ in range(self.config.L_layers)])

        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _prepare_inputs(self, obs_pose, obs_mask, batch):
        """
        Processes Kinematics and Images ONCE before recursion.
        Returns: 
          - agent_embeddings: [B, A*T, D]
          - visual_context: [B, T*H*W+1, D]
          - full_mask
        """
        B, T, A, _ = obs_pose.shape
        
        # --- 1. Agent Embeddings ---
        emb = self.embed_inputs(obs_pose.to(self.forward_dtype)) # [B, T, A, D]
        
        time_ids = torch.arange(T, device=obs_pose.device)
        agent_ids = torch.arange(A, device=obs_pose.device)
        emb = emb + self.embed_time(time_ids)[None, :, None, :] + self.embed_agent(agent_ids)[None, None, :, :]
        
        emb = emb.permute(0, 2, 1, 3).contiguous().view(B, A * T, self.config.hidden_size)
        token_mask = obs_mask.to(torch.bool).permute(0, 2, 1).contiguous().view(B, A * T)
        emb = emb * token_mask[..., None].to(emb.dtype)

        global_tok = self.global_token.expand(B, -1, -1)
        emb = torch.cat([global_tok, emb], dim=1)
        
        global_mask = torch.ones((B, self.config.global_len), device=token_mask.device, dtype=torch.bool)
        full_mask = torch.cat([global_mask, token_mask], dim=1)

        if self.config.pos_encodings == "learned":
            emb = 0.707106781 * (emb + self.embed_pos.embedding_weight.to(self.forward_dtype))

        visual_context = None
        if batch is not None:
            #print("batch is not None")
            visual_context = self._prepare_visual_context_from_batch(batch, T=T, device=obs_pose.device)
        return self.embed_scale * emb, full_mask, visual_context
    
    def _prepare_visual_context_from_batch(self, batch: Dict[str, torch.Tensor], T: int, device) -> Optional[torch.Tensor]:
        # Collect selected cameras from batch
        #selected = getattr(self.config, "use_cameras", ["F"])
        #print(selected)
        cam_names = getattr(self.config, "cam_names", ["F","FL","FR","B","BL","BR"])
        #print(cam_names, batch.keys())
        vis_seqs = []
        for cam in cam_names:
            key = f"camera_{cam}_features"  # you already have these keys
            if key not in batch:
                #print("does not exist")
                continue

            feats = batch[key]  # expected [B, T, 512, 9, 16] (precomputed)
            B, T2, C, H, W = feats.shape
            assert T2 == T, f"{key} has T={T2}, expected {T}"

            feats = feats.to(self.forward_dtype)

            # [B*T, 512, H, W]
            feats = feats.view(B * T, C, H, W)

            # [B*T, H*W, 512]
            feats_flat = feats.flatten(2).transpose(1, 2)

            # [B*T, H*W, D]
            tokens = self.vis_proj(feats_flat)
            tokens = self.vis_norm(tokens)
            tokens = tokens * self.embed_scale

            # spatial emb: [1, HW, D]
            tokens = tokens + self.vis_pos_emb

            # reshape for time/cam embedding: [B, T, HW, D]
            HW = tokens.shape[1]
            tokens = tokens.view(B, T, HW, self.config.hidden_size)

            # add temporal embedding: [1, T, 1, D]
            time_ids = torch.arange(T, device=device)
            tokens = tokens + self.embed_time(time_ids).view(1, T, 1, self.config.hidden_size)

            # add camera-id embedding: [1, 1, 1, D]
            cam_id = torch.tensor([self.cam_to_id[cam]], device=device)
            tokens = tokens + self.embed_cam(cam_id).view(1, 1, 1, self.config.hidden_size)

            # flatten: [B, T*HW, D]
            tokens = tokens.view(B, T * HW, self.config.hidden_size)

            vis_seqs.append(tokens)

        if len(vis_seqs) == 0:
            #print("Return None")
            return None

        # concat cameras along sequence dimension: [B, sum_cam(T*HW), D]
        vis_tokens = torch.cat(vis_seqs, dim=1)
        expected_len = self.config.num_cameras * T * HW
        _, loc_seq_len, _ = vis_tokens.shape
        #print("length", expected_len)

        assert loc_seq_len == expected_len, (
        f"[Vision ERROR] visual token length mismatch: "
        f"got seq_len={loc_seq_len}, expected={expected_len} "
        f"(cams={len(vis_seqs)}, T={T}, HW={HW})."
        )

        # add null token at front
        B = vis_tokens.shape[0]
        null_tok = self.visual_null_token.expand(B, 1, -1)
        return torch.cat([null_tok, vis_tokens], dim=1)
    
    def empty_carry(self, batch_size: int, device: torch.device) -> TRM_ACT_NuScenes_InnerCarry:
        total_len = self.config.global_len + self.seq_len
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
        visual_context_cache: Optional[torch.Tensor]
    ) -> Tuple[TRM_ACT_NuScenes_InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:

        #cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        

        # Precompute Inputs (and Vision) only if cache is empty
        # In ACT, batch data stays same, so we reuse visual_context
        if visual_context_cache is None:
            #print("visual conext none")
            input_embeddings, full_mask, visual_context = self._prepare_inputs(
                batch["obs_pose"],
                batch["obs_mask"],
                batch
            )
            
        else:
            # Re-compute only agent embeddings if needed (but usually constant in ACT step)
            # For simplicity here we re-run prepare but should optimize in prod
            #print("visual context cached")
            input_embeddings, full_mask, _ = self._prepare_inputs(batch["obs_pose"], batch["obs_mask"], None)
            visual_context = visual_context_cache
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
            token_mask=full_mask # Pack mask here
        )
        z_H, z_L = carry.z_H, carry.z_L
        
        # Recursive Passes with Cross-Attention
        # The 'visual_context' is passed into L_level -> Block -> CrossAttn
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(
                        hidden_states=z_L, 
                        input_injection=z_H + input_embeddings, 
                        visual_context=visual_context, 
                        **seq_info
                    )
                z_H = self.L_level(
                    hidden_states=z_H, 
                    input_injection=z_L, 
                    visual_context=visual_context, 
                    **seq_info
                )
        
        # Grad Pass
        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, visual_context, **seq_info)
        z_H = self.L_level(z_H, z_L, visual_context, **seq_info)

        new_carry = TRM_ACT_NuScenes_InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        
        # Decoding
        global_latent = z_H[:, 0]
        q_logits = self.q_head(global_latent).to(torch.float32)
        
        B, A, T_obs, D = z_H.shape[0], self.config.max_obstacles, self.config.n_history, self.config.hidden_size
        agent_tokens = z_H[:, self.config.global_len:]
        history_kv = agent_tokens.contiguous().view(B, A, T_obs, D).view(B * A, T_obs, D)
        queries = self.future_queries.squeeze(0).expand(B * A, -1, -1)
        
        attn_out, _ = self.decoder_attn(query=queries, key=history_kv, value=history_kv)
        pred = self.output_proj(attn_out).view(B, A, self.config.n_horizon, self.config.out_dim)

        if self.config.predict_delta and self.config.out_slice > 0:
            last_obs = batch["obs_pose"][:, -1].to(self.forward_dtype)
            base = last_obs[:, :, : self.config.out_slice].to(pred.dtype)
            pred_slice = pred[..., : self.config.out_slice]
            pred = torch.cat([base[:, :, None, :] + pred_slice, pred[..., self.config.out_slice :]], dim=-1)

        last_step_mask = batch["obs_mask"][:, -1].to(torch.bool)
        pred = pred * last_step_mask[:, :, None, None].to(pred.dtype)
        visual_context = visual_context.detach() if visual_context is not None else None
        return new_carry, pred, (q_logits[..., 0], q_logits[..., 1]), global_latent, visual_context

# =========================
# ACT Wrapper (Standard)
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
            current_data={k: torch.empty_like(v) if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
            prev_loss = torch.zeros((B,), dtype=torch.float32, device=device),
            visual_context=None
        )

    def forward(self, carry: TRM_ACT_NuScenes_Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TRM_ACT_NuScenes_Carry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)
        new_prev_loss = torch.where(
            carry.halted,
            torch.full_like(carry.prev_loss, float('inf')),
            carry.prev_loss,
        )

        # new_current_data = {}
        # for k, v in carry.current_data.items():
        #     if k in batch and isinstance(batch[k], torch.Tensor) and isinstance(v, torch.Tensor):
        #         dims_needed = batch[k].ndim - 1
        #         mask_expanded = carry.halted.view((-1,) + (1,) * dims_needed)
        #         new_current_data[k] = torch.where(mask_expanded, batch[k], v)
        #     else:
        #         new_current_data[k] = batch[k]
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
        }

        new_inner_carry, pred, (q_halt, q_cont), global_latent, vis_ctx = self.inner(new_inner_carry, new_current_data, carry.visual_context)
        
        outputs = {"pred": pred, "global_latent": global_latent, "q_halt_logits": q_halt, "q_continue_logits": q_cont}
        
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last = new_steps >= self.config.halt_max_steps
            halted = is_last
            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt > 0)
                else:
                    halted = halted | (q_halt > q_cont)
                if self.config.halt_exploration_prob > 0:
                     explore = (torch.rand_like(q_halt) < self.config.halt_exploration_prob)
                     min_halt = torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                     halted = halted & (new_steps >= torch.where(explore, min_halt, torch.zeros_like(min_halt)))
        if vis_ctx is not None:
            vis_ctx = vis_ctx.detach()
        new_carry = TRM_ACT_NuScenes_Carry(new_inner_carry, new_steps, halted, new_current_data, new_prev_loss, vis_ctx)
        return new_carry, outputs