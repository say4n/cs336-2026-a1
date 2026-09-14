from traitlets import default
from collections.abc import Callable
import torch
import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay rate: {lr}")

        defaults = {
            "lr": lr,
            "lambda": weight_decay,
            "betas": betas,
            "eps": eps,
        }

        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["lambda"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                
                t = state.get("t", 1)  # Get iteration number from the state, or 0.
                m = state.get("m", 0)  # First order moment estimate.
                v = state.get("v", 0)  # Second order moment estimate.
                grad = p.grad.data     # Get the gradient of loss with respect to p.

                adjusted_lr = lr * math.sqrt(1 - beta_2 ** t)/(1 - beta_1 ** t)
                
                p.data = p.data * ( 1 - lr * weight_decay)

                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * grad ** 2

                p.data = p.data - adjusted_lr * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v


        return loss


def experiment(optim: torch.optim.Optimizer):
    num_runs = 10
    steps = 100
    learning_rates = [1e1, 1e2, 1e3]

    records = []

    for lr in tqdm(learning_rates):
        for run in range(num_runs):
            weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
            opt = optim([weights], lr=lr)

            for step in range(steps):
                opt.zero_grad()

                loss = (weights**2).mean()
                records.append(
                    {"lr": str(lr), "run": run, "step": step, "loss": loss.item()}
                )

                loss.backward()
                opt.step()

    df = pd.DataFrame(records)

    # Plotting
    plt.figure(figsize=(8, 5))

    sns.lineplot(
        data=df,
        x="step",
        y="loss",
        hue="lr",
        errorbar=(
            "ci",
            95,
        ),  # Use errorbar=('ci', 95) if you want 95% confidence intervals instead, else "sd"
    )

    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Loss (log scale)")
    plt.title("Mean Loss with (95% CI)")
    plt.show()


if __name__ == "__main__":
    experiment(SGD)
    # experiment(AdamW)
