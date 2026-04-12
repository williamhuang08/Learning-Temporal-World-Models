"""
Per-timestep reward decoder: p(r_t | s_t) where s_t is an RSSM feature.
Diagonal Gaussian over scalar reward.
"""

import torch
import torch.nn as nn


class RewardDecoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        h_dim: int = 256,
        min_std: float = 0.1,
        max_std: float = 2.0,
    ):
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std

        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )
        self.std_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )

    def forward(self, features):
        """
        Args:
            features: [..., feature_dim]
        Returns:
            mean: [...],  std: [...]   (scalar reward per timestep)
        """
        h = self.trunk(features)
        mean = self.mean_head(h).squeeze(-1)
        std = (self.max_std * torch.sigmoid(self.std_head(h)) + self.min_std).squeeze(-1)
        return mean, std
