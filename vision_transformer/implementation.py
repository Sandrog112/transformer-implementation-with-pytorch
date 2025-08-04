from torch import nn
import torch
import math
from torch import Tensor

# Before implementing the preprocessing steps of Vision Transformer, let's first define the encoder block and its components.

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
        
    def scaled_dot_product_attention(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        """
        Computes scaled dot-product attention.

        Shape:
            - Q, K, V: (batch_size, num_heads, seq_len, d_k)
            - Output: (batch_size, num_heads, seq_len, d_k)
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
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
        
    def forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        """
        Forward pass for multi-head attention.

        Shape:
            - Q, K, V: (batch_size, seq_len, d_model)
            - Output: (batch_size, seq_len, d_model)
        """
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed forward neural network layer.

    Args:
        d_model (int): Input and output dimension
        d_ff (int): Hidden layer dimension
        
    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    """
    
    def __init__(self, d_model: int, d_ff: int):
        """
        Initializes the feed forward network layers.
        
        """
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed forward network.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Processed tensor through two linear layers with GELU activation
        """
        return self.fc2(self.gelu(self.fc1(x)))

    
class TransformerEncoderBlock(nn.Module):
    """
    A single Transformer encoder block used in Vision Transformer (ViT).
    
    Args:
        d_model (int): Dimension of input embeddings
        num_heads (int): Number of attention heads
        d_ff (int): Dimension of feedforward network
        dropout (float): Dropout rate
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        norm_x = self.norm1(x)
        attn_output = self.mha(norm_x, norm_x, norm_x)
        x = x + self.dropout1(attn_output)
        
        norm_x = self.norm2(x)
        ffn_output = self.ffn(norm_x)
        x = x + self.dropout2(ffn_output)

        return x


class VisionTransformerEncoder(nn.Module):
    """
    Complete Vision Transformer Encoder - stack of transformer encoder blocks.
    
    Args:
        num_layers (int): Number of encoder blocks
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward dimension 
        dropout (float): Dropout rate
        
    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    """
    
    def __init__(self, num_layers: int, d_model: int, num_heads: int, 
                 d_ff: int, dropout: float):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through all encoder layers.
        
        Args:
            x (Tensor): Input embeddings (patches + positional + [CLS] token)
            
        Returns:
            Tensor: Final encoded representations
        """
        for layer in self.layers:
            x = layer(x)
            
        return self.norm(x)


# Transformer encoder for our ViT is ready, we can now implement the preprocessing steps to convert images into patch embeddings.
