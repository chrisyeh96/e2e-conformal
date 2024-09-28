import numpy as np
import torch
from torch import Tensor


def calc_q(scores: Tensor, alpha: float) -> Tensor:
    """
    Calculates the quantile values from the scores.

    Args:
        scores: shape [n], predicted scores
        alpha: quantile level

    Returns:
        q: scalar Tensor, same dtype as scores
    """
    n = len(scores)
    j = int(np.ceil((n+1) * (1-alpha)))
    if j > n:
        return torch.tensor(torch.inf)
    sorted_inds = torch.argsort(scores)
    q = scores[sorted_inds[j-1]]
    return q
