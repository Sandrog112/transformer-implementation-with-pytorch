import torch
from torch import nn


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.

    Applies rotary positional encoding to input tensors by rotating
    query and key vectors in attention mechanisms.

    Args:
        d (int): Dimension of the embedding (should be even)
        base (int): Base for frequency computation (default: 10000)
    """

    def __init__(self, d: int, base: int = 10000):
        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def build_cache(self, x: torch.Tensor):
        """
        Build and cache cos/sin rotation matrices for the sequence length.

        Args:
            x (torch.Tensor): Input tensor with shape [seq_len, batch, heads, d_model]
        """
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        seq_len = x.shape[0]

        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(
            x.device
        )

        seq_idx = torch.arange(seq_len, device=x.device).float()

        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)

        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def neg_half(self, x: torch.Tensor):
        """
        Apply rotation by swapping and negating half of the dimensions.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Rotated tensor for complex multiplication
        """
        d_2 = self.d // 2

        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        """
        Apply RoPE to input tensor.

        Args:
            x (torch.Tensor): Input tensor [seq_len, batch, heads, d_model]

        Returns:
            torch.Tensor: Tensor with rotary positional encoding applied
        """
        self.build_cache(x)

        seq_len = x.shape[0]

        x_rope, x_pass = x[:, :, :, : self.d], x[:, :, :, self.d :]

        neg_half_x = self.neg_half(x_rope)

        x_rope = (x_rope * self.cos_cached[:seq_len]) + (
            neg_half_x * self.sin_cached[:seq_len]
        )

        return torch.cat([x_rope, x_pass], dim=-1)
