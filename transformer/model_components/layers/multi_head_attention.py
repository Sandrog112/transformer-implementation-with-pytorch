from torch import nn
import torch
import math
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism that allows the model to attend to information
    from different representation subspaces.

    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads

    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(
        self, Q: Tensor, K: Tensor, V: Tensor, mask: Tensor = None
    ) -> Tensor:
        """
        Computes scaled dot-product attention.

        Shape:
            - Q, K, V: (batch_size, num_heads, seq_len, d_k)
            - mask: (batch_size, 1, seq_len, seq_len)
            - Output: (batch_size, num_heads, seq_len, d_k)
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x: Tensor) -> Tensor:
        """Split tensor into multiple attention heads.

        Shape:
            - Input: (batch_size, seq_len, d_model)
            - Output: (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: Tensor) -> Tensor:
        """
        Combines multiple attention heads back into original shape.

        Shape:
            - Input: (batch_size, num_heads, seq_len, d_k)
            - Output: (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Tensor = None) -> Tensor:
        """
        Forward pass for multi-head attention.

        Shape:
            - Q, K, V: (batch_size, seq_len, d_model)
            - mask: (batch_size, 1, seq_len, seq_len)
            - Output: (batch_size, seq_len, d_model)
        """
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
