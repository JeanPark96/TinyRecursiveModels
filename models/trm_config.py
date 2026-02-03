
from dataclasses import dataclass

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
