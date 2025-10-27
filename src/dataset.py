import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


classes = ["entailment", "neutral", "contradiction"]

def split_indices(N, seed, n_calib = 500, n_test = 500):
    """Randomly partitions dataset indices into training, calibration and test sets."""
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)

    calib = idx[:n_calib]
    test = idx[n_calib:(n_calib + n_test)]
    train = idx[(n_calib + n_test):]

    return train, calib, test

def loaders(X, Y_true, Y_noisy, idx_train, idx_calib, idx_test, batch = 32):
    """Builds loaders for one (train/calib/test)."""
    def mk(idx, shuffle):
        ds = TensorDataset(X[idx], Y_true[idx], Y_noisy[idx])

        return DataLoader(ds, batch_size = batch, shuffle = shuffle)

    return mk(idx_train, False), mk(idx_calib, True), mk(idx_test, True)

def load_cat(m_pt, m_noisy_pt, s_pt, s_noisy_pt):
    """Loads and concatenates PyTorch Tensor packs."""
    ps = torch.load(m_pt, map_location = "cpu")
    pm = torch.load(s_pt, map_location = "cpu")
    ps_noisy = torch.load(m_noisy_pt, map_location = "cpu")
    pm_noisy = torch.load(s_noisy_pt, map_location = "cpu")

    X = torch.cat([ps["embeddings"],pm["embeddings"]], dim = 0).float()
    Y_true = torch.cat([ps["distributions"], pm["distributions"]], dim = 0).float()
    Y_noisy = torch.cat([ps_noisy["distributions_noisy"], pm_noisy["distributions_noisy"]], dim = 0).float()

    return X, Y_true, Y_noisy
