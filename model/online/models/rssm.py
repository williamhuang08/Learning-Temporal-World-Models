"""
Abstract RSSM (Recurrent State-Space Model) operating at the temporally-
abstracted level.  Each abstract timestep corresponds to one H-step skill
segment; the skill z plays the role of "action" at the abstract level.

Dreamer-style architecture:
    Recurrence (deterministic):  h_τ = GRUCell(h_{τ-1}, [s_{τ-1}, z_{τ-1}])
    Prior      (dynamics):       p(s_τ | h_τ)
    Posterior  (representation): q(s_τ | h_τ, context_τ)

The module exposes encoder_parameters() and dynamics_parameters() so the
EM trainer can assign them to separate optimisers.
"""

import torch
import torch.nn as nn


# ── Dynamics: GRUCell + prior heads (M-step parameters) ──────────────────

class _RSSMDynamics(nn.Module):
    def __init__(self, s_dim: int, z_dim: int, h_dim: int):
        super().__init__()
        self.gru_cell = nn.GRUCell(s_dim + z_dim, h_dim)
        self.prior_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, s_dim),
        )
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, s_dim),
            nn.Softplus(),
        )

    def forward(self, h_prev, s_prev, z_prev):
        """
        One deterministic recurrence step + prior readout.

        Args:
            h_prev: [B, h_dim]
            s_prev: [B, s_dim]
            z_prev: [B, z_dim]
        Returns:
            h_new:      [B, h_dim]
            prior_mean: [B, s_dim]
            prior_std:  [B, s_dim]
        """
        x = torch.cat([s_prev, z_prev], dim=-1)
        h_new = self.gru_cell(x, h_prev)
        return h_new, self.prior_mean(h_new), self.prior_std(h_new)


# ── Encoder: obs/traj encoder + posterior heads (E-step parameters) ──────

class _RSSMEncoder(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, s_dim: int, h_dim: int):
        super().__init__()
        # Encode a single observation (for s_0 posterior)
        self.obs_enc = nn.Sequential(
            nn.Linear(obs_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim),
        )
        # Encode an H-step trajectory (for s_1 posterior)
        self.traj_gru = nn.GRU(
            input_size=obs_dim + action_dim,
            hidden_size=h_dim,
            batch_first=True,
        )
        self.obs_proj = nn.Linear(obs_dim, obs_dim + action_dim)

        # Shared posterior heads: (h_τ, context) -> (mean, std)
        self.post_mean = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, s_dim),
        )
        self.post_std = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, s_dim),
            nn.Softplus(),
        )

    def posterior(self, h, context):
        """
        Args:
            h:       [B, h_dim]  deterministic hidden state
            context: [B, h_dim]  encoded observation or trajectory
        """
        x = torch.cat([h, context], dim=-1)
        return self.post_mean(x), self.post_std(x)

    def encode_obs(self, o):
        """Encode a single observation for s_0 posterior.  o: [B, obs_dim]"""
        return self.obs_enc(o)

    def encode_trajectory(self, obs_seq, act_seq):
        """
        Encode an H-step trajectory for s_1 posterior.

        Args:
            obs_seq: [B, H+1, obs_dim]
            act_seq: [B, H, action_dim]
        Returns:
            summary: [B, h_dim]
        """
        H = act_seq.size(1)
        oa = torch.cat([obs_seq[:, :H, :], act_seq], dim=-1)  # [B, H, obs+act]
        oH = self.obs_proj(obs_seq[:, -1:, :])                 # [B, 1, obs+act]
        tokens = torch.cat([oa, oH], dim=1)                    # [B, H+1, obs+act]
        _, h_final = self.traj_gru(tokens)                     # [1, B, h_dim]
        return h_final.squeeze(0)                               # [B, h_dim]


# ── AbstractRSSM: top-level module ───────────────────────────────────────

class AbstractRSSM(nn.Module):
    """
    Temporally-abstracted RSSM.

    Usage for a single independent segment::

        h0, s_zeros, z_zeros = rssm.initial_state(B, device)

        # step 0 — from zeros
        h0, s0_prior, s0_post = rssm.observe_step(
            h0, s_zeros, z_zeros, obs_context=rssm.encoder.encode_obs(o_0))

        # infer skill z_0 from skill encoder (external)
        ...

        # step 1 — after executing skill
        h1, s1_prior, s1_post = rssm.observe_step(
            h0, s0, z0, traj_context=rssm.encoder.encode_trajectory(obs_seq, act_seq))

    For imagination (no observations)::

        h_next, prior_mean, prior_std = rssm.imagine_step(h, s, z)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        s_dim: int,
        z_dim: int,
        h_dim: int = 256,
    ):
        super().__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.dynamics = _RSSMDynamics(s_dim, z_dim, h_dim)
        self.encoder = _RSSMEncoder(obs_dim, action_dim, s_dim, h_dim)

    # -- parameter groups for EM split --

    def dynamics_parameters(self):
        """GRUCell + prior heads — updated in the M-step."""
        return self.dynamics.parameters()

    def encoder_parameters(self):
        """Trajectory/obs encoder + posterior heads — updated in the E-step."""
        return self.encoder.parameters()

    # -- state initialisation --

    def initial_state(self, batch_size: int, device: torch.device):
        """
        Returns:
            h: [B, h_dim]  zero-initialised deterministic state
            s: [B, s_dim]  zero-initialised stochastic state
            z: [B, z_dim]  zero-initialised skill placeholder
        """
        h = torch.zeros(batch_size, self.h_dim, device=device)
        s = torch.zeros(batch_size, self.s_dim, device=device)
        z = torch.zeros(batch_size, self.z_dim, device=device)
        return h, s, z

    # -- single RSSM step with observations --

    def observe_step(self, h_prev, s_prev, z_prev, context):
        """
        Deterministic recurrence + prior + posterior for one abstract step.

        Args:
            h_prev:  [B, h_dim]
            s_prev:  [B, s_dim]
            z_prev:  [B, z_dim]
            context: [B, h_dim]  from encode_obs (for s_0) or
                                 encode_trajectory (for s_1)
        Returns:
            h:          [B, h_dim]
            prior:      (mean [B, s_dim], std [B, s_dim])
            posterior:  (mean [B, s_dim], std [B, s_dim])
        """
        h, pr_mean, pr_std = self.dynamics(h_prev, s_prev, z_prev)
        po_mean, po_std = self.encoder.posterior(h, context)
        return h, (pr_mean, pr_std), (po_mean, po_std)

    # -- imagination (no observations) --

    def imagine_step(self, h_prev, s_prev, z_prev):
        """
        One step using only the dynamics (prior).  Used for planning.

        Returns:
            h:          [B, h_dim]
            prior_mean: [B, s_dim]
            prior_std:  [B, s_dim]
        """
        return self.dynamics(h_prev, s_prev, z_prev)
