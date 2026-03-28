"""
Dreamer-style Gaussian observation decoder: predicts endpoints o_0 and o_H from
latent states s_0 and s_1 (diagonal Normal, NLL training).
"""

import torch
import torch.nn as nn


class SegmentObservationDecoder(nn.Module):
    """
    p_dec(o_0 | s_0) and p_dec(o_H | s_1), each a diagonal Gaussian over obs_dim.
    Separate heads after a shared trunk on s_dim.
    """

    def __init__(
        self,
        obs_dim: int,
        s_dim: int,
        h_dim: int = 256,
        min_std: float = 0.01,
        max_std: float = 2.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.min_std = min_std
        self.max_std = max_std

        self.trunk = nn.Sequential(
            nn.Linear(s_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.o0_mean = nn.Linear(h_dim, obs_dim)
        self.o0_std = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, obs_dim),
        )
        self.oH_mean = nn.Linear(h_dim, obs_dim)
        self.oH_std = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, obs_dim),
        )

    def _split_gaussian(self, mean_lin, std_mlp, h):
        mu = mean_lin(h)
        std = self.max_std * torch.sigmoid(std_mlp(h)) + self.min_std
        return mu, std

    def forward_o0(self, s0: torch.Tensor):
        """s0: [B, s_dim] -> mu, std each [B, obs_dim]"""
        h = self.trunk(s0)
        return self._split_gaussian(self.o0_mean, self.o0_std, h)

    def forward_oH(self, s1: torch.Tensor):
        """s1: [B, s_dim] -> mu, std each [B, obs_dim]"""
        h = self.trunk(s1)
        return self._split_gaussian(self.oH_mean, self.oH_std, h)
