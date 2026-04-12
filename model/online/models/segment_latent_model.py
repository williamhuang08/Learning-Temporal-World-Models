"""
Terminal state predictor: p_psi(s_T | s_0, z).

Predicts the RSSM feature at the end of a skill segment given the start
feature and the skill latent.  Used in the M-step ELBO and in CEM planning.
"""

import torch
import torch.nn as nn


class TerminalStatePredictor(nn.Module):
    """p_psi(s_T | s_0, z) — Gaussian over the RSSM feature at segment end."""

    def __init__(
        self,
        feature_dim: int,
        z_dim: int,
        h_dim: int = 256,
        min_std: float = 0.1,
        max_std: float = 2.0,
    ):
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std

        self.net = nn.Sequential(
            nn.Linear(feature_dim + z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, feature_dim),
        )
        self.std_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, feature_dim),
        )

    def forward(self, s0_feat, z):
        """
        Args:
            s0_feat: [B, feature_dim]  RSSM feature at t=0
            z:       [B, z_dim]        skill latent
        Returns:
            mean: [B, feature_dim],  std: [B, feature_dim]
        """
        x = torch.cat([s0_feat, z], dim=-1)
        h = self.net(x)
        mean = self.mean_head(h)
        std = self.max_std * torch.sigmoid(self.std_head(h)) + self.min_std
        return mean, std
