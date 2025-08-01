from torch import nn
import torch   
from model_components.layers.multi_head_attention import MultiHeadAttention
from model_components.layers.feed_forward_nn import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    """
    Implements encoder block in transformer architecture.
    
    Consists of multi-head attention, position-wise feed-forward network,   
    and layer normalization.

    Args:
        d_model (int): Model's dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward hidden layer dimension
        dropout (float): Dropout probability
        
    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        """
        Initializes encoder layer components.
        
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through encoder layer.
        
        Args:
            x (Tensor): Input sequence
            mask (Tensor, optional): Attention mask
            
        Returns:
            Tensor: Processed sequence through self-attention and feed-forward layers
        """
        attn_output = self.self_attn(Q=x, K=x, V=x, mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x