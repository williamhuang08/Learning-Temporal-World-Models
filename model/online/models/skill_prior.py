import torch
import torch.nn as nn
from torch.distributions import Normal, Independent


class AbstractSkillPrior(nn.Module):
    """
    Unimodal Gaussian skill prior conditioned on an abstract state.

    Input:  abstract state s_t
    Output: (mean, std) over skill z
    """

    def __init__(self, s_dim: int, z_dim: int, h_dim: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(s_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
        )
        self.sig_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus(),
        )

    def forward(self, s):
        """
        Args:
            s: [B, s_dim]
        Returns:
            mean: [B, z_dim],  std: [B, z_dim]
        """
        h = self.layers(s)
        return self.mean_head(h), self.sig_head(h)
