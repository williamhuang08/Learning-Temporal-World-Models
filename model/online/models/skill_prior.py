import torch
import torch.nn as nn


class AbstractSkillPrior(nn.Module):
    """
    Unimodal Gaussian skill prior conditioned on an abstract state.

    Input:  abstract state s_t
    Output: (mean, std) over skill z
    """

    def __init__(self, s_dim: int, z_dim: int, h_dim: int = 256,
                 min_std: float = 0.1, max_std: float = 2.0):
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std
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
        )

    def forward(self, s):
        """
        Args:
            s: [B, s_dim]
        Returns:
            mean: [B, z_dim],  std: [B, z_dim]
        """
        h = self.layers(s)
        mean = self.mean_head(h)
        std = self.max_std * torch.sigmoid(self.sig_head(h)) + self.min_std
        return mean, std
