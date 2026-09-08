from src.utils import scaled_dot_product_attention
from src.embedding import RotaryPositionalEmbedding
from einops import einsum, rearrange
import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """Construct a linear transformation module. This function should accept the following parameters:

        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        w = torch.empty(self.in_features, self.out_features, device=device, dtype=dtype)
        std = 2.0/(self.out_features + self.in_features)
        nn.init.trunc_normal_(
            w, mean=0.0, std=std, a=-3 * std**0.5, b=3 * std**0.5
        )

        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input"""

        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.d_ff =  d_ff

        self.w1 = Linear(d_ff, d_model, device, dtype)
        self.w2 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: d_model
        w1: d_ff x d_model
        w2: d_model x d_ff
        w3: d_ff x d_model

        w1_x: ... d_ff
        silu: ... d_ff
        w3_x: ... d_ff
        """
        a1 = self.w1(x)

        silu = a1 * torch.sigmoid(a1)

        swiglu = self.w2(silu * self.w3(x))

        return swiglu


class CausalMultiHeadedSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k, device, dtype)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k, device, dtype)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v, device, dtype)

        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model, device, dtype)

    def forward(
        self, 
        x: torch.Tensor,
        rope: RotaryPositionalEmbedding = None,
        token_positions: torch.Tensor = None, 
    ) -> torch.Tensor:
        batch, seq, _ = x.shape

        w_q_x = self.q_proj(x)
        q = rearrange(w_q_x, "batch seq (heads d_k) -> batch heads seq d_k", heads = self.num_heads)

        w_k_x = self.k_proj(x)
        k = rearrange(w_k_x, "batch seq (heads d_k) -> batch heads seq d_k", heads = self.num_heads)

        w_v_x = self.v_proj(x)
        v = rearrange(w_v_x, "batch seq (heads d_v) -> batch heads seq d_v", heads = self.num_heads)

        if rope:
            if token_positions is None:
                token_positions = torch.arange(seq, dtype=torch.long, device=x.device)
            q = rope(q, token_positions)
            k = rope(k, token_positions)

        mask = ~torch.triu(torch.ones((seq, seq), device=x.device, dtype=torch.bool), diagonal=1)

        y = scaled_dot_product_attention(q, k, v, mask)
        y_rearranged = rearrange(y, "batch heads seq d_v -> batch seq (heads d_v)")

        return self.output_proj(y_rearranged)
