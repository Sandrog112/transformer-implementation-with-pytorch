import torch
from torch import nn
import math
from roformer.rope import RoPE
from transformer.model_components.layers.feed_forward_nn import PositionWiseFeedForward


class TokenEmbedding2(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding2, self).__init__(vocab_size, d_model, padding_idx=1)


class RoformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob):
        super(RoformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding2(vocab_size, d_model)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        return self.drop_out(tok_emb)


class MultiHeadAttention2(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention2, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.rope = RoPE(self.d_k)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, V)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        Q = Q.transpose(1, 2).transpose(0, 1)
        K = K.transpose(1, 2).transpose(0, 1)

        Q = self.rope(Q)
        K = self.rope(K)

        Q = Q.transpose(0, 1).transpose(1, 2)
        K = K.transpose(0, 1).transpose(1, 2)

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention2(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_output = self.self_attn(Q=x, K=x, V=x, mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention2(d_model, num_heads)
        self.cross_attn = MultiHeadAttention2(d_model, num_heads)
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
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
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

        self.encoder_embedding = RoformerEmbedding(
            vocab_size, d_model, max_seq_length, dropout
        )
        self.decoder_embedding = RoformerEmbedding(
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
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        seq_len = tgt.size(1)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        subsequent_mask = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).eq(
            0
        )
        tgt_mask = tgt_mask & subsequent_mask

        return src_mask, tgt_mask

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
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
        x = self.decoder_embedding(tgt)
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask, tgt_mask = self.create_masks(src, tgt)

        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)

        output = self.final_layer(dec_output)
        return torch.softmax(output, dim=-1)
