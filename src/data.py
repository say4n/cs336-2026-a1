import numpy as np
import torch


def get_batch(
    x: np.array, batch_size: int, context_length: int, device: torch.device
) -> tuple[torch.LongTensor, torch.LongTensor]:
    min_idx = 0
    max_idx = len(x) - context_length - 1

    if max_idx < 0:
        raise ValueError(f"Can't generate data with `{len(x)}` samples using {context_length = }")

    start_indices = np.random.randint(
        min_idx, 
        max_idx + 1,
        batch_size,
    )

    batch = np.array([x[start : start + context_length] for start in start_indices])
    batch_tensor = torch.tensor(batch, dtype=torch.long, device=device)

    target = np.array([x[start + 1 : start + context_length + 1] for start in start_indices])
    target_tensor = torch.tensor(target, dtype=torch.long, device=device)

    return batch_tensor, target_tensor
