import numpy as np
import torch
from scipy.spatial import ConvexHull
from typing import Literal

from .plot import corners


tri_area = np.sqrt(3) / 4

def hull_area(points_bary):
    """Calculates the 2D area of the convex hull of credal points."""
    points = np.asarray(points_bary)

    if points.ndim != 2 or points.shape[0] < 3:
        return 0

    xy = points @ corners

    return ConvexHull(xy).volume

def evaluate_fo(confpred, loader, err_rate = 0.1, num_samples = 10000, nc: Literal["KL", "CHI2"] = "KL"):
    """Evaluates a first-order conformal credal predictor."""
    confpred.calibrate_fo(loader, err_rate, nc)

    cov, eff = [], []

    with torch.no_grad():
        for X, Y_true, Y_noisy in loader:
            for x, y_true, y_noisy in zip(X, Y_true, Y_noisy):
                cred, pred = confpred.pred_set_sampling_fo(x, num_samples = num_samples, nc = nc)
                s = confpred.score_fo(x, y_true, nc)

                cov.append(1 if s < confpred.q else 0)
                eff.append(hull_area(cred) / tri_area)

    return np.mean(cov), np.mean(eff)

def evaluate_so(confpred, loader, err_rate = 0.1, num_samples = 10000, eps = 1e-9):
    """Evaluates a second-order conformal credal predictor."""
    confpred.calibrate_so(loader, err_rate)

    cov, eff = [], []

    with torch.no_grad():
        for X, Y_true, Y_noisy in loader:
            for x, y_true, y_noisy in zip(X, Y_true, Y_noisy):
                cred, weights = confpred.pred_set_sampling_so(x, num_samples = num_samples)
                alpha = confpred.model(x.unsqueeze(0), order = "second").squeeze(0).cpu().numpy()
                s = np.asarray(confpred.rl(y_true.numpy()[None, :], alpha, eps)).reshape(())

                cov.append(1 if s > confpred.q else 0)
                eff.append(hull_area(cred) / tri_area)

    return np.mean(cov), np.mean(eff)
