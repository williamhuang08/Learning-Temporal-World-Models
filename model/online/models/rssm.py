"""
GRU-based Recurrent State-Space Model (RSSM).

State s_t = (h_t, z_t) where h_t is deterministic (GRU) and z_t is stochastic
(diagonal Gaussian).  The full RSSM feature is concat(h_t, z_t).

Prior path  (imagination):  h_{t+1} = GRU(h_t, [z_t, a_t])
                             z_{t+1} ~ p(z | h_{t+1})

Posterior path (training):   h_{t+1} = GRU(h_t, [z_t, a_t])
                             z_{t+1} ~ q(z | h_{t+1}, o_{t+1})

Initial:                     h_0 = 0,  z_0 ~ q(z | o_0)
Initial prior:               p(s_0) = N(0, I)  (for KL at t=0)
"""

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence


class RSSM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        h_dim: int = 256,
        stoch_dim: int = 32,
        hidden_dim: int = 256,
        min_std: float = 0.1,
        max_std: float = 2.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.h_dim = h_dim
        self.stoch_dim = stoch_dim
        self.feature_dim = h_dim + stoch_dim
        self.min_std = min_std
        self.max_std = max_std

        # GRU input: previous stochastic z_t concatenated with action a_t
        self.gru_input_proj = nn.Linear(stoch_dim + action_dim, h_dim)
        self.gru_cell = nn.GRUCell(h_dim, h_dim)

        # Prior: p(z_{t+1} | h_{t+1})
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.prior_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim),
        )
        self.prior_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim),
        )

        # Posterior: q(z_{t+1} | h_{t+1}, o_{t+1})
        self.posterior_net = nn.Sequential(
            nn.Linear(h_dim + obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.posterior_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim),
        )
        self.posterior_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim),
        )

        # Initial posterior: q(z_0 | o_0) — h_0 is zero so we only condition on o_0
        self.init_posterior_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.init_posterior_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim),
        )
        self.init_posterior_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim),
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _clamp_std(self, raw):
        return self.max_std * torch.sigmoid(raw) + self.min_std

    def _prior(self, h):
        feat = self.prior_net(h)
        return self.prior_mean(feat), self._clamp_std(self.prior_std(feat))

    def _posterior(self, h, o):
        feat = self.posterior_net(torch.cat([h, o], dim=-1))
        return self.posterior_mean(feat), self._clamp_std(self.posterior_std(feat))

    def _init_posterior(self, o0):
        feat = self.init_posterior_net(o0)
        return self.init_posterior_mean(feat), self._clamp_std(self.init_posterior_std(feat))

    @staticmethod
    def _sample(mean, std):
        return mean + std * torch.randn_like(mean)

    def get_features(self, h, z):
        """Concatenate deterministic and stochastic parts -> full RSSM state."""
        return torch.cat([h, z], dim=-1)

    # ------------------------------------------------------------------
    # full posterior roll-out (training)
    # ------------------------------------------------------------------

    def observe(self, obs_seq, act_seq):
        """
        Run posterior inference over a trajectory chunk.

        Args:
            obs_seq:    [B, T+1, obs_dim]  observations o_0 … o_T
            act_seq:    [B, T,   action_dim] actions a_0 … a_{T-1}
        Returns:
            dict with keys:
                h_seq        [B, T+1, h_dim]
                z_seq        [B, T+1, stoch_dim]
                feat_seq     [B, T+1, feature_dim]
                prior_mean   [B, T+1, stoch_dim]   (t=0 entry is from N(0,1))
                prior_std    [B, T+1, stoch_dim]
                post_mean    [B, T+1, stoch_dim]
                post_std     [B, T+1, stoch_dim]
        """
        B, Tp1, _ = obs_seq.shape
        T = Tp1 - 1
        device = obs_seq.device

        h_list, z_list = [], []
        pr_mean_list, pr_std_list = [], []
        po_mean_list, po_std_list = [], []

        # t = 0: h_0 = 0, z_0 ~ q(z_0 | o_0)
        h = torch.zeros(B, self.h_dim, device=device)
        po_mean_0, po_std_0 = self._init_posterior(obs_seq[:, 0])
        z = self._sample(po_mean_0, po_std_0)

        # prior at t=0 is N(0, I)
        pr_mean_0 = torch.zeros_like(po_mean_0)
        pr_std_0 = torch.ones_like(po_std_0)

        h_list.append(h)
        z_list.append(z)
        pr_mean_list.append(pr_mean_0)
        pr_std_list.append(pr_std_0)
        po_mean_list.append(po_mean_0)
        po_std_list.append(po_std_0)

        for t in range(T):
            h, z, pr_m, pr_s, po_m, po_s = self._step_posterior(
                h, z, act_seq[:, t], obs_seq[:, t + 1]
            )
            h_list.append(h)
            z_list.append(z)
            pr_mean_list.append(pr_m)
            pr_std_list.append(pr_s)
            po_mean_list.append(po_m)
            po_std_list.append(po_s)

        h_seq = torch.stack(h_list, dim=1)
        z_seq = torch.stack(z_list, dim=1)

        return {
            "h_seq": h_seq,
            "z_seq": z_seq,
            "feat_seq": self.get_features(h_seq, z_seq),
            "prior_mean": torch.stack(pr_mean_list, dim=1),
            "prior_std": torch.stack(pr_std_list, dim=1),
            "post_mean": torch.stack(po_mean_list, dim=1),
            "post_std": torch.stack(po_std_list, dim=1),
        }

    def _step_posterior(self, h, z, a, o_next):
        """One posterior transition step."""
        gru_in = self.gru_input_proj(torch.cat([z, a], dim=-1))
        h_next = self.gru_cell(gru_in, h)
        pr_m, pr_s = self._prior(h_next)
        po_m, po_s = self._posterior(h_next, o_next)
        z_next = self._sample(po_m, po_s)
        return h_next, z_next, pr_m, pr_s, po_m, po_s

    # ------------------------------------------------------------------
    # single prior step (imagination / planning)
    # ------------------------------------------------------------------

    def imagine_step(self, h, z, a):
        """
        One prior transition step (no observation).

        Returns:
            h_next, z_next, prior_mean, prior_std
        """
        gru_in = self.gru_input_proj(torch.cat([z, a], dim=-1))
        h_next = self.gru_cell(gru_in, h)
        pr_m, pr_s = self._prior(h_next)
        z_next = self._sample(pr_m, pr_s)
        return h_next, z_next, pr_m, pr_s

    # ------------------------------------------------------------------
    # initialize from a single observation (for planning)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_state(self, o0):
        """
        Encode a single observation into the initial RSSM state.

        Args:
            o0: [B, obs_dim]
        Returns:
            h: [B, h_dim],  z: [B, stoch_dim]
        """
        h = torch.zeros(o0.shape[0], self.h_dim, device=o0.device)
        po_m, po_s = self._init_posterior(o0)
        z = self._sample(po_m, po_s)
        return h, z

    @torch.no_grad()
    def posterior_step(self, h, z, a, o_next):
        """
        One posterior update for online execution (planner uses this to
        maintain RSSM state as real observations come in).
        """
        h_next, z_next, _, _, _, _ = self._step_posterior(h, z, a, o_next)
        return h_next, z_next
