import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def split_loader(loader: DataLoader) -> tuple[TensorDataset, TensorDataset]:
    """Splits DataLoader into two."""
    ds: TensorDataset = loader.dataset
    n = len(ds)
    n1 = n // 2

    rng = np.random.default_rng(seed=567)
    indices = rng.permutation(n).tolist()

    d1 = TensorDataset(*[t[indices[:n1]] for t in ds.tensors])
    d2 = TensorDataset(*[t[indices[n1:]] for t in ds.tensors])
    return d1, d2
