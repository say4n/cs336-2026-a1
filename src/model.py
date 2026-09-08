from src.utils import softmax
from src.layers import Linear
from src.layers import SwiGLU
from src.embedding import Embedding, RotaryPositionalEmbedding
from src.regularization import RMSNorm
import torch
from torch import nn

from src.layers import CausalMultiHeadedSelfAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        self.rope = rope

        self.attn = CausalMultiHeadedSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
        )

        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_intermediate = x + self.attn(
            self.ln1(x),
            self.rope,
        )

        y = y_intermediate + self.ffn(self.ln2(y_intermediate))

        return y


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)

        self.layers = nn.Sequential(*[
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                rope,
                device,
                dtype
            ) for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )

        self.lm_head = Linear(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.token_embeddings(x)
        y = self.layers(y)
        y = self.ln_final(y)
        y = self.lm_head(y)

        return y
