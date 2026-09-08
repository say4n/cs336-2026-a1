from src.layers import SwiGLU
from src.embedding import RotaryPositionalEmbedding
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
