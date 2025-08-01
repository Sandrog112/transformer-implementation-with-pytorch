import torch
from torch import nn
import math

class TokenEmbedding(nn.Embedding):
    """
    Token embedding with torch.nn

    """

    def __init__(self, vocab_size, d_model):
        """
        Initializes the TokenEmbedding layer.

        This layer creates embeddings for input tokens by inheriting from nn.Embedding.
        It maps token indices to dense vectors of fixed size.

        Args:
            vocab_size (int): Size of the vocabulary
            d_model (int): Dimension of the embedding vector for each token

        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class PositionalEncoding(nn.Module):
    """Positional Encoding layer for Transformer models.

    This layer adds positional information to the input embeddings using
    sine and cosine functions of different frequencies. 

    Args:
        d_model (int): The dimension of the embeddings
        max_seq_length (int): Maximum length of input sequences

    """
    def __init__(self, d_model, max_seq_length):
        """Initializes the PositionalEncoding layer.

        Args:
            d_model (int): Dimension of the model's embeddings
            max_seq_length (int): Maximum sequence length to pre-compute encodings for
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """Adds positional encodings to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Input combined with positional encodings
        """
        return x + self.pe[:, :x.size(1)]


class TransformerEmbedding(nn.Module):
    """
    Combines token embeddings with positional encodings.

    """
    def __init__(self, vocab_size, d_model, max_len, drop_prob):
        """
        Initialize embedding layers and dropout.
        
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)  
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """Combines token embeddings with positional encodings.
        
        Args:
            x (Tensor): Input tensor of token ids
            
        Returns:
            Tensor: Combined embeddings with dropout applied
        """
        tok_emb = self.tok_emb(x)  
        return self.drop_out(self.pos_emb(tok_emb))  