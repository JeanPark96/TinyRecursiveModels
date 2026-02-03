import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.decafnet_generator import PyramidGenerator, MaskedConv1D
# --- 1. Helper Blocks (Required dependencies from blocks.py) ---

class LayerNorm(nn.Module):
    """ LayerNorm that supports input of size (bs, c, t) """
    def __init__(self, n_channels, affine=True, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.n_channels = n_channels
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(n_channels, 1))
            self.bias = nn.Parameter(torch.zeros(n_channels, 1))
        else:
            self.weight = self.bias = None

    def forward(self, x):
        # x: [Batch, Channel, Time]
        x = x - torch.mean(x, dim=1, keepdim=True)
        sigma = torch.mean(x ** 2, dim=1, keepdim=True)
        x = x / torch.sqrt(sigma + self.eps)
        if self.weight is not None:
            x = x * self.weight + self.bias
        return x

class Scale(nn.Module):
    """ Learnable scale parameter for regression range """
    def __init__(self, init=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.as_tensor(init, dtype=torch.float))

    def forward(self, x):
        return x * self.scale.to(x.dtype)


# --- 2. DeCafNet Heads (from head.py) ---

class ClsHead(nn.Module):
    """ 1D Conv head for event classification """
    def __init__(self, embd_dim, n_layers=2, prior_prob=0.01):
        super().__init__()
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        
        # Stack convolution layers
        for _ in range(n_layers):
            self.convs.append(MaskedConv1D(embd_dim, embd_dim, 3, 1, 1, bias=False))
            self.norms.append(LayerNorm(embd_dim))

        # Final projection to 1 channel (logit)
        self.cls_head = MaskedConv1D(embd_dim, 1, 3, 1, 1)

        # Bias initialization for stability (RetinaNet style)
        bias_init = 0
        if prior_prob > 0:
            bias_init = -np.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_head.conv.bias, bias_init)

    def forward(self, fpn, fpn_masks):
        # Expects tuples/lists as input (Standard DeCafNet/FPN behavior)
        out_logits, out_masks = tuple(), tuple()    
        for x, mask in zip(fpn, fpn_masks):
            for conv, norm in zip(self.convs, self.norms):
                x, _ = conv(x, mask)
                x = F.relu(norm(x), inplace=True)
            
            logits, _ = self.cls_head(x, mask)                 # (bs, 1, t)
            logits = logits.squeeze(1)                         # (bs, t)
            mask = mask.squeeze(1)                             # (bs, t)
            out_logits += (logits, )
            out_masks += (mask, )

        return out_logits, out_masks

class RegHead(nn.Module):
    """ 1D Conv head for offset regression """
    def __init__(self, embd_dim, num_fpn_levels=1, n_layers=2):
        super().__init__()
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        
        for _ in range(n_layers):
            self.convs.append(MaskedConv1D(embd_dim, embd_dim, 3, 1, 1, bias=False))
            self.norms.append(LayerNorm(embd_dim))

        # Final projection to 2 channels (Start Offset, End Offset)
        self.reg_head = MaskedConv1D(embd_dim, 2, 3, 1, 1)
        
        # Scale parameter per pyramid level
        self.scales = nn.ModuleList([Scale() for _ in range(num_fpn_levels)])

    def forward(self, fpn, fpn_masks):
        out_offsets, out_masks = tuple(), tuple()
        for i, (x, mask) in enumerate(zip(fpn, fpn_masks)):
            for conv, norm in zip(self.convs, self.norms):
                x, _ = conv(x, mask)
                x = F.relu(norm(x), inplace=True)
            
            offsets, _ = self.reg_head(x, mask)
            
            # Apply learnable scale and ReLU (offsets must be positive)
            offsets = F.relu(self.scales[i](offsets))          # (bs, 2, t)
            offsets = offsets.transpose(1, 2)                  # (bs, t, 2)
            mask = mask.squeeze(1)                             # (bs, t)
            out_offsets += (offsets, )
            out_masks += (mask, )
            
        return out_offsets, out_masks


# --- 3. Wrapper for Your Model ---

class BoundaryPredictor(nn.Module):
    """
    Wraps the DeCafNet heads to work with your single modified video tensor.
    """
    def __init__(self, embd_dim=256, n_layers=2):
        super().__init__()
        
        # We assume 1 FPN level since you are using a single refined video stream
        self.cls_head = ClsHead(embd_dim, n_layers=n_layers, prior_prob=0.01)
        self.reg_head = RegHead(embd_dim, num_fpn_levels=1, n_layers=n_layers)

    def forward(self, video_feats, video_mask):
        """
        Args:
            video_feats: [Batch, Channel, Time] - Output from your recurrent loop
            video_mask:  [Batch, 1, Time]
        """
        
        # DeCafNet heads expect a TUPLE of levels (Feature Pyramid style).
        # Since we have only one level, we wrap it in a tuple.
        fpn = (video_feats, )
        fpn_masks = (video_mask, )
        
        # 1. Classification
        # logits: tuple containing one tensor of shape [Batch, Time]
        logits_tuple, _ = self.cls_head(fpn, fpn_masks)
        
        # 2. Regression
        # offsets: tuple containing one tensor of shape [Batch, Time, 2]
        offsets_tuple, _ = self.reg_head(fpn, fpn_masks)
        
        # Unwrap the single level results
        return logits_tuple[0], offsets_tuple[0]

class PyramidBoundaryPredictor(nn.Module):
    def __init__(self, embd_dim=256, num_levels=3, n_layers=2):
        super().__init__()
        
        # 1. Feature Pyramid Generator
        self.pyramid_gen = PyramidGenerator(embd_dim, num_levels)
        
        # 2. DeCafNet Heads
        # Note: RegHead needs to know num_fpn_levels to create correct number of Scales
        self.cls_head = ClsHead(embd_dim, n_layers=n_layers, prior_prob=0.01)
        self.reg_head = RegHead(embd_dim, num_fpn_levels=num_levels, n_layers=n_layers)

    def forward(self, video_feats, video_mask):
        """
        video_feats: [B, C, T] (Your refined single-level feature)
        """
        
        # A. Build the Pyramid
        # fpn: (Lvl0, Lvl1, Lvl2)
        fpn, fpn_masks = self.pyramid_gen(video_feats, video_mask)
        
        # B. Predict on all levels
        # cls_logits: tuple of (B, T_i)
        cls_logits, _ = self.cls_head(fpn, fpn_masks)
        
        # reg_offsets: tuple of (B, T_i, 2)
        reg_offsets, _ = self.reg_head(fpn, fpn_masks)
        
        # Return tuples (standard for loss computation in these models)
        return cls_logits, reg_offsets