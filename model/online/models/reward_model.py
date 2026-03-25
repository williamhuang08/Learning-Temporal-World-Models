import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    """
    Goal-conditioned predictor for minimum distance-to-goal over the H-step window.

    Input:  abstract state s0, skill z, raw goal g (e.g. desired xy)
    Output: (mean [B], std [B]) for a Gaussian over the minimum Euclidean distance
            from achieved goal xy to g achieved at any timestep t in {0..H}.
    """

    def __init__(
        self,
        s_dim: int,
        z_dim: int,
        goal_dim: int = 2,
        h_dim: int = 256,
        min_std: float = 0.1,
        max_std: float = 2.0,
    ):
        super().__init__()
        self.goal_dim = goal_dim
        self.min_std = min_std
        self.max_std = max_std
        self.trunk = nn.Sequential(
            nn.Linear(s_dim + z_dim + goal_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(h_dim, 1)
        self.std_head = nn.Linear(h_dim, 1)

    def forward(self, s, z, g):
        """
        Args:
            s: [B, s_dim]
            z: [B, z_dim]
            g: [B, goal_dim]  raw goal (not observation-normalized)
        Returns:
            mean: [B]
            std:  [B]
        """
        h = self.trunk(torch.cat([s, z, g], dim=-1))
        mean = F.softplus(self.mean_head(h).squeeze(-1)) + 1e-4
        std = self.max_std * torch.sigmoid(self.std_head(h).squeeze(-1)) + self.min_std
        return mean, std
