import torch
from torch import nn


def register_compat_buffer(
    module: nn.Module,
    name: str,
    tensor: torch.Tensor,
    persistent: bool = True,
):
    """
    Register a tensor as a buffer in a way that is compatible across
    PyTorch versions.

    - Uses nn.Buffer if available (newer torch)
    - Falls back to register_buffer otherwise
    """
    Buffer = getattr(nn, "Buffer", None)

    if Buffer is not None:
        # Newer PyTorch (Buffer subclass exists)
        setattr(module, name, Buffer(tensor, persistent=persistent))
    else:
        # Old PyTorch
        module.register_buffer(name, tensor, persistent=persistent)
