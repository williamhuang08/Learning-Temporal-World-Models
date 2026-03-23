import torch
import torch.nn as nn

# Low-Level Skill-Conditioned Policy, pi_theta
class SkillPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, h_dim=256, z_dim=256, a_dist='normal', max_sig=None, fixed_sig=None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.a_dist = a_dist
        self.max_sig = max_sig
        self.fixed_sig = fixed_sig

        self.layers = nn.Sequential(
            nn.Linear(state_dim + z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, action_dim)
        )
        self.sig_head  = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, action_dim)
        )

    def forward(self, state, z):
        # state: [B*T, state_dim], z: [B*T, Z_DIM]
        x = torch.cat([state, z], dim=-1)
        feats = self.layers(x)
        mean  = self.mean_head(feats)
        if self.max_sig is None:
            sig = F.softplus(self.sig_head(feats))
            # sig = F.softplus(self.sig_head(feats)) + 1e-4
            # sig = torch.clamp(sig, min=1e-4, max=10.0)
        else:
            sig = self.max_sig * torch.sigmoid(self.sig_head(feats))
        if self.fixed_sig is not None:
            sig = self.fixed_sig * torch.ones_like(sig)
        return mean, sig
