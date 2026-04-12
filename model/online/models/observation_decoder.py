"""
Per-timestep observation decoder: p(o_t | s_t) where s_t is an RSSM feature.
Diagonal Gaussian over obs_dim, applied identically at every timestep.
"""

import torch
import torch.nn as nn


class ObservationDecoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        feature_dim: int,
        h_dim: int = 256,
        min_std: float = 0.01,
        max_std: float = 2.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
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
            nn.Linear(h_dim, obs_dim),
        )
        self.std_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, obs_dim),
        )

    def forward(self, features):
        """
        Args:
            features: [..., feature_dim]  (any leading batch/time dims)
        Returns:
            mean: [..., obs_dim],  std: [..., obs_dim]
        """
        h = self.trunk(features)
        mean = self.mean_head(h)
        std = self.max_std * torch.sigmoid(self.std_head(h)) + self.min_std
        return mean, std
