import numpy as np
import torch


def noise_injection(pt_path, save_path, m = 20, seed = 42):
    """Loads distributions and creates noisy versions by sampling m times."""
    rng = np.random.default_rng(seed) # for reproducibility

    pack = torch.load(pt_path, map_location = "cpu")
    dists = pack["distributions"].numpy()
    N, K = dists.shape

    noisy = np.zeros_like(dists, dtype = np.float32)

    for i in range(N):
        p = dists[i].astype(np.float64)
        p = np.clip(p / (p.sum() + 1e-9), 0.0, 1.0)
        counts = rng.multinomial(m, p)
        noisy[i] = counts / float(m)

    pack["distributions_noisy"] = torch.from_numpy(noisy)
    torch.save(pack, save_path)

if __name__ == "__main__":
    noise_injection("data/chaosNLI_mnli_m_embeddings.pt", "data/chaosNLI_mnli_m_embeddings_noisy.pt")
    noise_injection("data/chaosNLI_snli_embeddings.pt", "data/chaosNLI_snli_embeddings_noisy.pt")

    print("Noisy distributions are saved.")
