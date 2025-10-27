import numpy as np
import torch
from typing import Literal


def kl(p, q, eps = 1e-9):
    """Computes the Kullback-Leibler divergence between two probability distributions."""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)

    return p * (np.log(p) - np.log(q))

def chi2(p, q, eps = 1e-9):
    """Computes the Chi-squared divergence between two probability distributions."""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)

    return ((p - q) ** 2) / q

class ConformalCredalPredictor:
    def __init__(self, model):
        self.model = model
        self.q = None

    def score_fo(self, x, y, nc):
        """Calculates distance scores for calibration."""
        with torch.no_grad():
            pred = self.model(x.unsqueeze(0), order = "first").squeeze(0).numpy()

        if nc == "KL":
            return np.sum(kl(y.numpy(), pred))
        else:
            return np.sum(chi2(y.numpy(), pred))

    def calibrate_fo(self, calib_loader, err_rate = 0.1, nc: Literal["KL", "CHI2"] = "KL", show_q = True):
        """Calculates a quantile threshold on a calibration set."""
        scores = []

        for X, Y_true, Y_noisy in calib_loader:
            for x, y_true, y_noisy in zip(X, Y_true, Y_noisy):
                scores.append(self.score_fo(x, y_noisy, nc))

        scores = np.array(scores)

        if len(scores) == 0:
            raise RuntimeError("No scores found for calibration set.")

        k = int(np.ceil((len(scores) + 1) * (1 - err_rate))) - 1
        self.q = np.sort(scores)[min(k, len(scores) - 1)]

        if show_q: print(f"Calibrated threshold: {self.q:.4f} (first-order, {nc})")

    def pred_set_sampling_fo(self,  x: torch.Tensor, num_samples = 10000, nc: Literal["KL", "CHI2"] = "KL"):
        """Builds a credal set for a new input by sampling in a simplex, using a distance-based NC function."""
        if self.q is None:
            raise RuntimeError("Model not calibrated yet.")

        with torch.no_grad():
            pred = self.model(x.unsqueeze(0), order = "first").squeeze(0).cpu().numpy()

        cands = np.random.dirichlet(np.ones(pred.shape[0]), size = num_samples)

        if nc == "KL":
            mask = (np.sum(kl(cands, pred), axis = 1) < self.q)
        else:
            mask = (np.sum(chi2(cands, pred), axis = 1) < self.q)

        return cands[mask], pred

    def rl(self, lambdas, alpha, eps = 1e-9):
        """Calculates relative-likelihood scores for calibration."""
        l = np.clip(lambdas, eps, 1)
        a = np.maximum(np.asarray(alpha), 1 + eps)
        m = np.clip((a - 1.0) / max(a.sum() - a.shape[0], eps), eps, 1.0) # mode

        logl = ((a[None, :] - 1.0) * np.log(l)).sum(axis = 1)
        logl_max = ((a - 1.0) * np.log(m)).sum()

        return np.exp(logl - logl_max)

    def calibrate_so(self, calib_loader, err_rate = 0.1, eps = 1e-9, show_q = True):
        """Calculates a quantile threshold on a calibration set."""
        scores = []

        for X, Y_true, Y_noisy in calib_loader:
            for x, y_true, y_noisy in zip(X, Y_true, Y_noisy):
                with torch.no_grad():
                    alpha = self.model(x.unsqueeze(0), order = "second").squeeze(0).cpu().numpy() # prediction for so

                s = self.rl(y_noisy.numpy(), alpha, eps)
                scores.append(float(s))

        scores = np.asarray(scores, float)

        if len(scores) == 0:
            raise RuntimeError("No scores found for calibration set.")

        self.q = float(np.quantile(scores, err_rate))

        if show_q: print(f"Calibrated threshold: {self.q:.4f} (second-order)")

    def pred_set_sampling_so(self, x: torch.Tensor, num_samples = 10000, eps = 1e-9):
        """Builds a credal set for a new input by sampling in a simplex, using a likelihood-based NC function."""
        q = self.q

        if q is None:
            raise RuntimeError("Model not calibrated yet.")

        with torch.no_grad():
            alpha = self.model(x.unsqueeze(0), order = "second").squeeze(0).cpu().numpy()

        cands = np.random.dirichlet(np.ones(alpha.shape[0]), size = num_samples)
        scores = self.rl(cands, alpha, eps)
        mask = (scores > q)

        weights = scores[mask]
        if weights.size > 0: weights /= weights.sum()

        return cands[mask], weights
