import torch
import torch.nn as nn
import torch.nn.functional as F


def dirichlet_nll(lambdas, alpha, eps = 1e-9):
    """Computes the mean negative log-likelihood of target distributions under a Dirichlet parameterization."""
    l = torch.clamp(lambdas, eps, 1)
    a = torch.clamp(alpha, eps, None)

    logB = torch.lgamma(a).sum(dim = 1) - torch.lgamma(a.sum(dim = 1))
    logp = -logB + ((a - 1) * torch.log(l)).sum(dim = 1)

    return (-logp).mean()

class CredalPredictor(nn.Module):
    def __init__(self, input = 768, hidden = [256, 64, 16], num_classes = 3, do = 0.3):
        super().__init__()

        self.shared = nn.Sequential(
            nn.LayerNorm(input),
            nn.Linear(input, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Dropout(do)
        )

        self.head_fo = nn.Linear(hidden[-1], num_classes)
        self.head_so = nn.Linear(hidden[-1], num_classes)

    def forward(self, x, order = "first"):
        h = self.shared(x)

        if order == "first":
            return torch.softmax(self.head_fo(h), dim = 1)
        else:
            return F.relu(self.head_so(h)) + 1.0
