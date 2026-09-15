import numpy as np


def learning_rate_schedule(t, alpha_min, alpha_max, t_w, t_c):
    if t < t_w:
        return t * alpha_max / t_w
    elif t_w <= t <= t_c:
        return alpha_min + 0.5 * (1 + np.cos((t - t_w) * np.pi / (t_c - t_w))) * (
            alpha_max - alpha_min
        )
    else:
        return alpha_min
