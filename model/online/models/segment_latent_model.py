"""
Segment-level latent model (no RSSM): deterministic s0 from o0, transformer q(s1|o,a),
and Markov prior p(s1|s0,z).
"""

import torch
import torch.nn as nn


class StartStateEncoder(nn.Module):
    """s_0 = f_start(o_0), deterministic."""

    def __init__(self, obs_dim: int, s_dim: int, h_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, s_dim),
        )

    def forward(self, o0: torch.Tensor) -> torch.Tensor:
        """o0: [B, obs_dim] -> [B, s_dim]"""
        return self.net(o0)


class StatePosteriorTransformer(nn.Module):
    """
    q(s_1 | o_{<=H}, a_{<H}): same transformer pattern as TransformerSkillEncoder.
    Tokens: (o_0,a_0), ..., (o_{H-1},a_{H-1}), (o_H, 0_a).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        s_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        min_std: float = 0.1,
        max_std: float = 2.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.min_std = min_std
        self.max_std = max_std

        self.oa_proj = nn.Linear(obs_dim + action_dim, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.mean_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, s_dim),
        )
        self.std_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, s_dim),
        )

    def forward(self, obs_seq: torch.Tensor, act_seq: torch.Tensor):
        """
        obs_seq: [B, H+1, obs_dim]
        act_seq: [B, H, action_dim]
        Returns:
            mean, std: [B, s_dim]
        """
        B, Hp1, _ = obs_seq.shape
        H = Hp1 - 1
        oa = torch.cat([obs_seq[:, :H, :], act_seq], dim=-1)
        a_pad = torch.zeros(B, self.action_dim, device=obs_seq.device, dtype=obs_seq.dtype)
        last = torch.cat([obs_seq[:, H, :], a_pad], dim=-1)
        tokens_in = torch.cat([oa, last.unsqueeze(1)], dim=1)

        tokens = self.oa_proj(tokens_in)
        positions = torch.arange(tokens.size(1), device=tokens.device)
        tokens = tokens + self.pos_emb(positions).unsqueeze(0)

        out = self.transformer(tokens)
        last_h = out[:, -1, :]

        mean = self.mean_head(last_h)
        std = self.max_std * torch.sigmoid(self.std_head(last_h)) + self.min_std
        return mean, std


class SegmentDynamics(nn.Module):
    """p_eta(s_1 | s_0, z): Markov Gaussian prior."""

    def __init__(
        self,
        s_dim: int,
        z_dim: int,
        h_dim: int = 256,
        min_std: float = 0.1,
        max_std: float = 2.0,
    ):
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std
        self.net = nn.Sequential(
            nn.Linear(s_dim + z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, s_dim),
        )
        self.std_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, s_dim),
        )

    def forward(self, s0: torch.Tensor, z: torch.Tensor):
        """Returns prior mean and std for s_1, each [B, s_dim]."""
        x = torch.cat([s0, z], dim=-1)
        h = self.net(x)
        mean = self.mean_head(h)
        std = self.max_std * torch.sigmoid(self.std_head(h)) + self.min_std
        return mean, std
