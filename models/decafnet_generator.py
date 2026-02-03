import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv1D(nn.Module):
    """ Masked 1D convolution """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super(MaskedConv1D, self).__init__()
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias
        )
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x, mask):
        # x: (bs, c, t), mask: (bs, 1, t)
        if mask is None:
            mask = torch.ones_like(x[:, :1], dtype=torch.bool)
        
        mask_float = mask.to(x.dtype)
        x = self.conv(x * mask_float)
        
        if self.stride > 1:
            mask_float = F.interpolate(mask_float, size=x.size(-1), mode='nearest')
            mask = mask_float.bool()
        return x, mask


class PyramidGenerator(nn.Module):
    """
    Takes a single feature stream and builds a hierarchy of downsampled features.
    Similar to the 'branch' layers in DeCafNet's VideoTransformer.
    """
    def __init__(self, embd_dim, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        self.layers = nn.ModuleList()
        
        # Create downsampling layers for levels 1 to N-1
        # Level 0 is the input itself (no layer needed)
        for _ in range(num_levels - 1):
            self.layers.append(
                MaskedConv1D(
                    embd_dim, embd_dim,
                    kernel_size=3, stride=2, padding=1, bias=False
                )
            )

    def forward(self, x, mask):
        """
        Args:
            x: [Batch, Channel, Time] (The refined output from your loop)
            mask: [Batch, 1, Time]
        Returns:
            fpn: tuple of tensors (Level0, Level1, ...)
            fpn_masks: tuple of masks
        """
        fpn = [x]
        fpn_masks = [mask]
        
        current_x = x
        current_mask = mask
        
        for layer in self.layers:
            # Apply strided convolution to downsample
            current_x, current_mask = layer(current_x, current_mask)
            # You might want to add Non-linearity/Norm here if not included in the block
            # But MaskedConv1D in DeCafNet is just Conv. 
            # Usually a ReLU is helpful between pyramid levels:
            current_x = F.relu(current_x) 
            
            fpn.append(current_x)
            fpn_masks.append(current_mask)
            
        return tuple(fpn), tuple(fpn_masks)