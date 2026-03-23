import torch
import torch.nn as nn


class RewardModel(nn.Module):
    """
    Gaussian reward predictor for cumulative reward over an H-step window.

    Input:  abstract state s_t and skill z
    Output: (mean [B], std [B]) parameterising N(reward | mean, std)
    """

    def __init__(self, s_dim: int, z_dim: int, h_dim: int = 256,
                 min_std: float = 0.1, max_std: float = 2.0):
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std
        self.trunk = nn.Sequential(
            nn.Linear(s_dim + z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(h_dim, 1)
        self.std_head = nn.Linear(h_dim, 1)

    def forward(self, s, z):
        """
        Args:
            s: [B, s_dim]
            z: [B, z_dim]
        Returns:
            mean: [B]
            std:  [B]
        """
        h = self.trunk(torch.cat([s, z], dim=-1))
        mean = self.mean_head(h).squeeze(-1)
        std = self.max_std * torch.sigmoid(self.std_head(h).squeeze(-1)) + self.min_std
        return mean, std
