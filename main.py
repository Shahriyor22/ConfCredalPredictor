import torch
import torch.nn.functional as F
import numpy as np

from src.dataset import split_indices, loaders, load_cat, classes
from src.model import CredalPredictor, dirichlet_nll
from src.cp import ConformalCredalPredictor
from src.plot import plot_simplex
from src.evaluation import evaluate_fo, evaluate_so


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop
def train_once(model, train_loader, epochs = 20, lr = 1e-3):
    """Trains the dual-head predictor for one run."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(epochs):
        model.train()
        loss_tot, loss_fo_tot, loss_so_tot = 0, 0, 0

        for x, y_true, y_noisy in train_loader:
            x, y_true, y_noisy = x.to(DEVICE), y_true.to(DEVICE), y_noisy.to(DEVICE)
            optimizer.zero_grad()

            # FO
            p = model(x, order = "first")
            logp = torch.log(p.clamp_min(1e-9))
            loss_fo = F.kl_div(logp, y_noisy, reduction = "batchmean")

            # SO
            alpha = model(x, order = "second")
            loss_so = dirichlet_nll(y_noisy, alpha)

            loss = loss_fo + 0.3 * loss_so
            loss.backward()
            optimizer.step()

            loss_tot += loss.item()
            loss_fo_tot += loss_fo.item()
            loss_so_tot += loss_so.item()

        print(f"Epoch {epoch:02d}: Total Loss = {loss_tot/len(train_loader):.4f} "
              f"FO Loss = {loss_fo_tot/len(train_loader):.4f}, SO Loss = {loss_so_tot/len(train_loader):.4f}")

# Dataset loading
X, Y_true, Y_noisy = load_cat("data/chaosNLI_mnli_m_embeddings.pt",
                              "data/chaosNLI_mnli_m_embeddings_noisy.pt",
                              "data/chaosNLI_snli_embeddings.pt",
                              "data/chaosNLI_snli_embeddings_noisy.pt")

n = X.shape[0]
seeds = 10
cov_kl, cov_chi2, cov_so = [], [], []
eff_kl, eff_chi2, eff_so = [], [], []

# Model training
for seed in range(seeds):
    print(f"----- SEED {seed + 1} -----")
    idx_tr, idx_calib, idx_test = split_indices(n, seed, 500, 500)
    train_loader, calib_loader, test_loader = loaders(X, Y_true, Y_noisy, idx_tr, idx_calib, idx_test, 32)

    model = CredalPredictor()
    train_once(model, train_loader, 10, 1e-4)

    confpred = ConformalCredalPredictor(model)

    c_kl, e_kl = evaluate_fo(confpred, calib_loader, 0.1, 10000, "KL")
    cov_kl.append(c_kl)
    eff_kl.append(e_kl)

    c_chi2, e_chi2 = evaluate_fo(confpred, calib_loader, 0.1, 10000, "CHI2")
    cov_chi2.append(c_chi2)
    eff_chi2.append(e_chi2)

    c_so, e_so = evaluate_so(confpred, calib_loader, 0.1, 10000)
    cov_so.append(c_so)
    eff_so.append(e_so)

    # Test
    if seed == seeds - 1:
        print(f"FO - KL: ({np.mean(cov_kl):.4f}, {np.mean(eff_kl):.4f}), "
              f"FO - CHI2: ({np.mean(cov_chi2):.4f}, {np.mean(eff_chi2):.4f}), "
              f"SO: ({np.mean(cov_so):.4f}, {np.mean(eff_so):.4f})")

        x_ex, y_true_ex, y_noisy_ex = X[idx_test][0], Y_true[idx_test][0], Y_noisy[idx_test][0]

        # Calibration
        confpred.calibrate_fo(calib_loader, 0.1, "KL", False)
        credal_set_kl, pred = confpred.pred_set_sampling_fo(x_ex, nc = "KL")

        confpred.calibrate_fo(calib_loader, 0.1, "CHI2", False)
        credal_set_chi2, _ = confpred.pred_set_sampling_fo(x_ex, nc = "CHI2")

        confpred.calibrate_so(calib_loader, 0.1, show_q = False)
        credal_set_so, weights = confpred.pred_set_sampling_so(x_ex)

        # Simplex plot
        plot_simplex(pred, y_true_ex.numpy(), y_noisy_ex.numpy(), weights, classes,
                     credal_set_kl, credal_set_chi2, credal_set_so)
