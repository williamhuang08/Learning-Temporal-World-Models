import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerSkillEncoder(nn.Module):
    """
    Encode an H-step trajectory of (o, a) pairs into a skill z (mean, std).

    Token layout:  [(o_0,a_0) | (o_1,a_1) | ... | (o_{H-1},a_{H-1})]
    Readout: last token position.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        z_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 80,
    ):
        super().__init__()
        self.d_model = d_model
        self.z_dim = z_dim

        self.oa_proj = nn.Linear(obs_dim + action_dim, d_model)
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
            nn.Softplus(),
        )

    def forward(self, obs_seq, act_seq):
        """
        Args:
            obs_seq:  [B, H, obs_dim]    observations o_0 ... o_{H-1}
            act_seq:  [B, H, action_dim] actions      a_0 ... a_{H-1}
        Returns:
            mean: [B, z_dim],  std: [B, z_dim]
        """
        oa = torch.cat([obs_seq, act_seq], dim=-1)  # [B, H, obs+act]
        tokens = self.oa_proj(oa)  # [B, H, d]

        positions = torch.arange(tokens.size(1), device=tokens.device)
        tokens = tokens + self.pos_emb(positions).unsqueeze(0)

        out = self.transformer(tokens)  # [B, H, d]
        last = out[:, -1, :]  # readout from last position

        return self.mean_head(last), self.std_head(last)
