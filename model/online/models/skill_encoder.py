"""
Transformer skill encoder: q_chi(z | s_{0:T}, a_{0:T-1}).

Operates on RSSM feature sequences (not raw observations).
Token layout:  [(s_0,a_0), (s_1,a_1), ..., (s_{T-1},a_{T-1}), (s_T, 0_a)]
Readout: last token position.
"""

import torch
import torch.nn as nn


class TransformerSkillEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        z_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        min_std: float = 0.1,
        max_std: float = 2.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.min_std = min_std
        self.max_std = max_std

        self.sa_proj = nn.Linear(feature_dim + action_dim, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.mean_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, z_dim),
        )
        self.std_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, z_dim),
        )

    def forward(self, feat_seq, act_seq):
        """
        Args:
            feat_seq: [B, T+1, feature_dim]  RSSM features s_0 … s_T
            act_seq:  [B, T,   action_dim]   actions a_0 … a_{T-1}
        Returns:
            mean: [B, z_dim],  std: [B, z_dim]
        """
        B, Tp1, _ = feat_seq.shape
        T = Tp1 - 1

        sa = torch.cat([feat_seq[:, :T, :], act_seq], dim=-1)
        a_pad = torch.zeros(B, self.action_dim, device=feat_seq.device, dtype=feat_seq.dtype)
        last = torch.cat([feat_seq[:, T, :], a_pad], dim=-1)
        tokens_in = torch.cat([sa, last.unsqueeze(1)], dim=1)

        tokens = self.sa_proj(tokens_in)
        positions = torch.arange(tokens.size(1), device=tokens.device)
        tokens = tokens + self.pos_emb(positions).unsqueeze(0)

        out = self.transformer(tokens)
        last_h = out[:, -1, :]

        mean = self.mean_head(last_h)
        std = self.max_std * torch.sigmoid(self.std_head(last_h)) + self.min_std
        return mean, std
