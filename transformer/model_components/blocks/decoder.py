import sys

sys.path.append(
    "c:\\Users\\AleksandreKurtishvil\\Desktop\\mastering-nlp-through-paper-implementation"
)
from torch import nn
import torch
from model_components.layers.multi_head_attention import MultiHeadAttention
from model_components.layers.feed_forward_nn import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    """
    Implements a single decoder layer in transformer architecture.

    Consists of self-attention, cross-attention with encoder output,
    and feed-forward network with residual connections and layer normalization.

    Args:
        d_model (int): Model's dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward hidden layer dimension
        dropout (float): Dropout probability

    Shape:
        - Input x: (batch_size, tgt_seq_len, d_model)
        - Input enc_output: (batch_size, src_seq_len, d_model)
        - Output: (batch_size, tgt_seq_len, d_model)
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        """
        Initialize decoder layer components.

        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass through decoder layer.

        Args:
            x (Tensor): Input sequence
            enc_output (Tensor): Output from encoder
            src_mask (Tensor, optional): Mask for encoder attention
            tgt_mask (Tensor, optional): Mask for decoder self-attention

        Returns:
            Tensor: Processed sequence through self-attention, cross-attention, and feed-forward layers
        """
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
