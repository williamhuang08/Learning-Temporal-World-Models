import torch
import torch.nn as nn


class RewardModel(nn.Module):
    """
    Predict cumulative reward over an H-step skill execution window.

    Input:  abstract state s_t and skill z
    Output: scalar predicted cumulative reward
    """

    def __init__(self, s_dim: int, z_dim: int, h_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )

    def forward(self, s, z):
        """
        Args:
            s: [B, s_dim]
            z: [B, z_dim]
        Returns:
            reward: [B, 1]
        """
        return self.net(torch.cat([s, z], dim=-1))
