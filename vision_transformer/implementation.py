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


class PatchEmbeddings(nn.Module):
    """
    Convert images into patches and project them into embedding space.
    
    This layer splits an image into non-overlapping patches and linearly projects
    each patch into a vector of size hidden_size.
    
    Args:
        image_size (int): Size of input image (assumes square images)
        patch_size (int): Size of each patch (assumes square patches)
        num_channels (int): Number of input channels (3 for RGB)
        hidden_size (int): Dimension of the embedding space
        
    Shape:
        - Input: (batch_size, num_channels, image_size, image_size)
        - Output: (batch_size, num_patches, hidden_size)
    """

    def __init__(self, image_size: int, patch_size: int, 
                 num_channels: int, hidden_size: int):
        super().__init__()
        
        assert image_size % patch_size == 0, f"Image size {image_size} must be divisible by patch size {patch_size}"
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            num_channels, hidden_size, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass to convert image to patch embeddings.
        
        Args:
            x (Tensor): Input images of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor: Patch embeddings of shape (batch_size, num_patches, hidden_size)
        """
        x = self.projection(x)
        
        x = x.flatten(2).transpose(1, 2)
        
        return x

class VisionTransformerEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings for Vision Transformer.
    
    This combines patch embeddings with learnable position embeddings and
    adds a special [CLS] token for classification.
    
    Args:
        image_size (int): Size of input image
        patch_size (int): Size of each patch
        num_channels (int): Number of input channels
        hidden_size (int): Dimension of embeddings
        dropout_prob (float): Dropout probability
        
    Shape:
        - Input: (batch_size, num_channels, image_size, image_size)
        - Output: (batch_size, num_patches + 1, hidden_size)  # +1 for CLS token
    """
        
    def __init__(self, image_size: int, patch_size: int, 
                 num_channels: int, hidden_size: int, 
                 dropout_prob: float):
        super().__init__()
        
        self.patch_embeddings = PatchEmbeddings(
            image_size=image_size,
            patch_size=patch_size, 
            num_channels=num_channels,
            hidden_size=hidden_size
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, hidden_size)
        )
        
        self.dropout = nn.Dropout(dropout_prob)
        
        self._init_parameters()
        
    def _init_parameters(self):
        """
        Initialize parameters following ViT paper.
        
        """
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass to create complete embeddings.
        
        Args:
            x (Tensor): Input images
            
        Returns:
            Tensor: Complete embeddings with CLS token and positional encoding
        """
        batch_size = x.shape[0]
        
        patch_embeddings = self.patch_embeddings(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        embeddings = torch.cat((cls_tokens, patch_embeddings), dim=1)
        
        embeddings = embeddings + self.position_embeddings
        
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
# Let’s now implement the complete Vision Transformer model that combines all these components.

class VisionTransformer(nn.Module):
    """
    Vision Transformer model.
    
    Args:
        image_size (int): Size of input image
        patch_size (int): Size of each patch
        num_channels (int): Number of input channels
        hidden_size (int): Dimension of embeddings
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        mlp_dim (int): Dimension of MLP layers
        dropout_prob (float): Dropout probability
        num_classes (int): Number of output classes
    """
    def __init__(self, image_size: int, patch_size: int, num_channels: int,
                 hidden_size: int, num_heads: int, num_layers: int,
                 mlp_dim: int, dropout_prob: float, num_classes: int):
        super().__init__()
        
        assert image_size % patch_size == 0, f"Image size {image_size} must be divisible by patch size {patch_size}"
        assert hidden_size % num_heads == 0, f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"

        self.embeddings = VisionTransformerEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
            dropout_prob=dropout_prob
        )

        self.transformer = VisionTransformerEncoder(
            num_layers=num_layers,
            d_model=hidden_size,
            num_heads=num_heads,
            d_ff=mlp_dim,
            dropout=dropout_prob
        )

        self.head = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Vision Transformer.
        
        Args:
            x (Tensor): Input images
            
        Returns:
            Tensor: Output logits
        """
        x = self.embeddings(x)

        x = self.transformer(x)

        x = self.head(x[:, 0])

        return x