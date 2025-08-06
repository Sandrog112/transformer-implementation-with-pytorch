import sys

sys.path.append(
    "c:\\Users\\AleksandreKurtishvil\\Desktop\\mastering-nlp-through-paper-implementation"
)
import torch
from torch import nn
from model_components.blocks.encoder import EncoderLayer
from model_components.blocks.decoder import DecoderLayer
from model_components.embedding.input_embedding import TransformerEmbedding


class Transformer(nn.Module):
    """
    Complete Transformer model architecture.

    Args:
        vocab_size (int): Size of vocabulary
        d_model (int): Model's dimension
        num_heads (int): Number of attention heads
        num_layers (int): Number of encoder/decoder layers
        d_ff (int): Feed-forward hidden dimension
        max_seq_length (int): Maximum sequence length
        dropout (float): Dropout probability

    Shape:
        - src: (batch_size, src_seq_len)
        - tgt: (batch_size, tgt_seq_len)
        - output: (batch_size, tgt_seq_len, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder_embedding = TransformerEmbedding(
            vocab_size, d_model, max_seq_length, dropout
        )
        self.decoder_embedding = TransformerEmbedding(
            vocab_size, d_model, max_seq_length, dropout
        )

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.final_layer = nn.Linear(d_model, vocab_size)

    def create_masks(self, src: torch.Tensor, tgt: torch.Tensor) -> tuple:
        """
        Creates source and target masks for transformer.

        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        seq_len = tgt.size(1)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        subsequent_mask = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).eq(
            0
        )
        tgt_mask = tgt_mask & subsequent_mask

        return src_mask, tgt_mask

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Encodes source sequence through encoder stack.

        """
        x = self.encoder_embedding(src)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes target sequence through decoder stack.

        """
        x = self.decoder_embedding(tgt)
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer.

        """
        src_mask, tgt_mask = self.create_masks(src, tgt)

        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)

        output = self.final_layer(dec_output)
        return torch.softmax(output, dim=-1)
