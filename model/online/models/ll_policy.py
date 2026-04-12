import torch
import torch.nn as nn


class SkillPolicy(nn.Module):
    """Low-level skill-conditioned policy: pi_theta(a_t | s_t, z).

    state_dim should be the RSSM feature dimension (h_dim + stoch_dim).
    """

    def __init__(self, state_dim, action_dim, h_dim=256, z_dim=256,
                 min_std=0.1, max_std=2.0, fixed_sig=None):
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std
        self.fixed_sig = fixed_sig

        self.layers = nn.Sequential(
            nn.Linear(state_dim + z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, action_dim),
        )
        self.sig_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, action_dim),
        )

    def forward(self, state, z):
        x = torch.cat([state, z], dim=-1)
        feats = self.layers(x)
        mean = self.mean_head(feats)
        if self.fixed_sig is not None:
            sig = self.fixed_sig * torch.ones_like(mean)
        else:
            sig = self.max_std * torch.sigmoid(self.sig_head(feats)) + self.min_std
        return mean, sig
