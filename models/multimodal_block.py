import torch
import torch.nn as nn
from models.blocks import TransformerDecoder, TransformerEncoder
from models.trm_config import TRMLocalizerConfig

class TRMBlock(nn.Module):
    def __init__(self, config: TRMLocalizerConfig, n_heads=4, dropout=0.1):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_heads

        # 1. INSTRUCTION: z_L looks at ORIGINAL Query/Text
        # Q=z_L, K=Text_Original
        self.instruction_unit = TransformerDecoder(
            embd_dim=config.hidden_size, kv_dim=config.hidden_size, # Assuming text is projected to latent_dim
            n_heads=n_heads, attn_pdrop=dropout, proj_pdrop=dropout, 
            xattn_mode='adaln'
        )

        # 2. READ: z_L looks at ORIGINAL Video
        # Q=z_L, K=Video_Original (Crucial for preventing drift)
        self.read_unit = TransformerDecoder(
            embd_dim=config.hidden_size, kv_dim=config.hidden_size, 
            n_heads=n_heads, attn_pdrop=dropout, proj_pdrop=dropout, 
            xattn_mode='adaln'
        )

        # 3. WRITE: Current Video Representation updates itself from z_L
        # Q=Video_Current, K=z_L
        # Note: We update the "Current" video features, but z_L is grounded in "Original"
        self.write_unit = TransformerDecoder(
            embd_dim=config.hidden_size, kv_dim=config.hidden_size, 
            n_heads=n_heads, attn_pdrop=dropout, proj_pdrop=dropout, 
            xattn_mode='adaln'
        )
        
        # 4. REPORT: z_H updates state from z_L
        self.report_unit = TransformerDecoder(
            embd_dim=config.hidden_size, kv_dim=config.hidden_size, 
            n_heads=n_heads, attn_pdrop=dropout, proj_pdrop=dropout, 
            xattn_mode='adaln'
        )

    def forward(self, 
                vid_curr,       
                vid_orig,       
                vid_mask, 
                z_H, z_L, 
                q_orig,         
                q_mask,          # <--- ADDED MASK INPUT
                loop_embedding):

        # --- A. MANAGER UPDATE ---
        if loop_embedding != None:
            z_H = z_H + loop_embedding 

        #z_L, _ = self.report_unit(q=z_L, q_mask=None, kv=z_H, kv_mask=None)
        # --- B. INSTRUCTION PHASE (Anchored) ---
        # Added q_mask here so attention ignores padding text
        z_L, _ = self.instruction_unit(q=z_L, q_mask=None, kv=q_orig, kv_mask=q_mask)

        # --- C. EXECUTION PHASE (Anchored Read) ---
        z_L, _ = self.read_unit(q=z_L, q_mask=None, kv=vid_orig, kv_mask=vid_mask)
        
        # --- D. WRITE PHASE ---
        vid_curr, _ = self.write_unit(q=vid_curr, q_mask=vid_mask, kv=z_L, kv_mask=None)

        # --- E. REPORT PHASE ---
        z_H, _ = self.report_unit(q=z_H, q_mask=None, kv=z_L, kv_mask=None)

        return vid_curr, z_H, z_L

class LightTRMBlock(nn.Module):
    def __init__(self, vid_dim=256, latent_dim=256, n_heads=4, dropout=0.1):
        super().__init__()
        
        # LIGHTWEIGHT: Just Cross-Attention (No FFN, No massive parameter count)
        # Used for z_L <-> Text and z_H <-> z_L
        self.attn_instruction = nn.MultiheadAttention(latent_dim, n_heads, dropout=dropout, batch_first=True)
        self.attn_report      = nn.MultiheadAttention(latent_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_instr       = nn.LayerNorm(latent_dim)
        self.norm_report      = nn.LayerNorm(latent_dim)

        # HEAVYWEIGHT: Full Decoder needed for Video Interactions
        self.read_unit = TransformerDecoder(
            embd_dim=latent_dim, kv_dim=vid_dim, 
            n_heads=n_heads, attn_pdrop=dropout, proj_pdrop=dropout, 
            xattn_mode='adaln'
        )

        self.write_unit = TransformerDecoder(
            embd_dim=vid_dim, kv_dim=latent_dim, 
            n_heads=n_heads, attn_pdrop=dropout, proj_pdrop=dropout, 
            xattn_mode='adaln'
        )

    def forward(self, vid_curr, vid_orig, vid_mask, z_H, z_L, q_orig, q_mask, loop_embedding):
        
        # --- A. MANAGER UPDATE ---
        z_H = z_H + loop_embedding 

        # --- B. INSTRUCTION (Lightweight) ---
        # Standard Residual Cross-Attn
        # Note: q_mask needs to be inverted for nn.MultiheadAttention (True = Ignore)
        # We assume q_mask is 1 for keep, 0 for ignore.
        
        # Prepare key_padding_mask for PyTorch MHA (True where we should IGNORE)
        key_mask = ~q_mask.bool().squeeze(1) if q_mask is not None else None
        
        q_instr, _ = self.attn_instruction(query=z_L, key=q_orig, value=q_orig, key_padding_mask=key_mask)
        z_L = self.norm_instr(z_L + q_instr)

        # --- C. READ (Heavy) ---
        z_L, _ = self.read_unit(q=z_L, q_mask=None, kv=vid_orig, kv_mask=vid_mask)
        
        # --- D. WRITE (Heavy) ---
        vid_curr, _ = self.write_unit(q=vid_curr, q_mask=vid_mask, kv=z_L, kv_mask=None)

        # --- E. REPORT (Lightweight) ---
        # z_H attends to z_L
        q_report, _ = self.attn_report(query=z_H, key=z_L, value=z_L)
        z_H = self.norm_report(z_H + q_report)

        return vid_curr, z_H, z_L