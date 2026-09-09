from einops import einsum
import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)

    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    dim = len(Q.shape) - 1

    qk_norm = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / d_k ** 0.5
    qk_masked = torch.where(mask, qk_norm, float("-inf"))
    qk_softmax = softmax(qk_masked, dim)

    attention = einsum(qk_softmax, V, "... n m, ... m d_v -> ... n d_v")

    return attention


def cross_entropy(
    predicted_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    batch, n_vocab = predicted_logits.shape
    stable_logits = predicted_logits - predicted_logits.max(dim=-1, keepdim=True).values

    # log(a/b) = log(a) - log(b)
    log_softmax = stable_logits - stable_logits.exp().sum(dim=-1, keepdim=True).log()
    selected_probs = log_softmax[torch.arange(batch), targets]

    return -selected_probs.mean(dim=-1)

def perplexity(
    losses: torch.Tensor
) -> torch.Tensor:
    m = losses.size(dim=0)
    return (losses.sum() / m).exp()
