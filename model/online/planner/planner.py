"""
CEM planner that operates in whitened epsilon space over abstract states.

Plans a sequence of skills by optimizing epsilon values (standard-normal
deviations from the skill prior) using the RSSM dynamics for imagination
and the reward model as the cost function.
"""

import torch
import numpy as np

from planner.cem import cem


class DreamerCEMPlanner:
    """
    CEM-based planner for temporally-abstracted skill models with RSSM.

    At each replanning step:
      1. Encode the current raw observation into (h, s) via RSSM posterior.
      2. Run CEM over epsilon sequences, scoring candidates by imagined
         reward-model rollouts in abstract state space.
      3. Convert the best epsilon sequence to skills z for execution.
    """

    def __init__(
        self,
        rssm,
        skill_prior,
        reward_model,
        pi_theta,
        device="cpu",
        skill_seq_len=10,
        pop_size=100,
        n_iters=100,
        frac_keep=0.5,
        l2_pen=0.0,
        length_cost=0.0,
    ):
        self.rssm = rssm
        self.skill_prior = skill_prior
        self.reward_model = reward_model
        self.pi_theta = pi_theta
        self.device = device

        self.skill_seq_len = skill_seq_len
        self.pop_size = pop_size
        self.n_iters = n_iters
        self.frac_keep = frac_keep
        self.l2_pen = l2_pen
        self.length_cost = length_cost

        self.z_dim = skill_prior.mean_head[-1].out_features

    @torch.no_grad()
    def encode_observation(self, obs_vec):
        """
        Encode a single raw observation into RSSM abstract state.

        Args:
            obs_vec: np.ndarray [obs_dim] or torch.Tensor [obs_dim]
        Returns:
            h: [1, h_dim]  deterministic hidden state
            s: [1, s_dim]  sampled stochastic abstract state (posterior mean)
        """
        if isinstance(obs_vec, np.ndarray):
            obs_vec = torch.tensor(obs_vec, dtype=torch.float32)
        obs_t = obs_vec.to(self.device).unsqueeze(0)  # [1, obs_dim]

        h_init, s_init, z_init = self.rssm.initial_state(1, self.device)
        obs_context = self.rssm.encoder.encode_obs(obs_t)
        h, _, s_post = self.rssm.observe_step(h_init, s_init, z_init, obs_context)
        s = s_post[0]  # posterior mean (deterministic for planning)
        return h, s

    @torch.no_grad()
    def imagined_rollout_cost(self, h, s, eps_seq):
        """
        Roll out RSSM dynamics in imagination and score with reward model.

        Args:
            h:       [1, h_dim]  initial deterministic state
            s:       [1, s_dim]  initial stochastic state
            eps_seq: [B, L, Z]   epsilon sequences to evaluate
        Returns:
            costs: [B]  lower is better (negative cumulative predicted reward)
        """
        B, L, _ = eps_seq.shape

        h_b = h.expand(B, -1).clone()
        s_b = s.expand(B, -1).clone()

        total_reward = torch.zeros(B, device=self.device)

        for i in range(L):
            mu_z, sigma_z = self.skill_prior(s_b)
            z_i = mu_z + sigma_z * eps_seq[:, i, :]

            r_mean, _ = self.reward_model(s_b, z_i)
            total_reward += r_mean - i * self.length_cost

            h_b, s_prior_mean, _ = self.rssm.imagine_step(h_b, s_b, z_i)
            s_b = s_prior_mean

        return -total_reward

    @torch.no_grad()
    def plan(self, obs_vec, skill_seq_len=None):
        """
        Run CEM to find the best epsilon sequence from the current observation.

        Args:
            obs_vec: np.ndarray [obs_dim]
            skill_seq_len: override horizon length (default: self.skill_seq_len)
        Returns:
            eps_mean: [L, Z]  optimized epsilon mean
            eps_std:  [L, Z]  optimized epsilon std
            h:        [1, h_dim]  encoded hidden state (for convert_epsilon_to_z)
            s:        [1, s_dim]  encoded abstract state
        """
        L = skill_seq_len or self.skill_seq_len

        h, s = self.encode_observation(obs_vec)

        eps_mean = torch.zeros(L, self.z_dim, device=self.device)
        eps_std = torch.ones(L, self.z_dim, device=self.device)

        cost_fn = lambda eps_seq: self.imagined_rollout_cost(h, s, eps_seq)

        eps_mean, eps_std = cem(
            eps_mean, eps_std, cost_fn,
            pop_size=self.pop_size,
            frac_keep=self.frac_keep,
            n_iters=self.n_iters,
            l2_pen=self.l2_pen,
        )

        return eps_mean, eps_std, h, s

    @torch.no_grad()
    def convert_epsilon_to_z(self, eps_seq, h, s):
        """
        Convert an epsilon sequence to skills by rolling out the skill prior
        and RSSM dynamics in abstract state space.

        Args:
            eps_seq: [L, Z]   epsilon sequence (single, unbatched)
            h:       [1, h_dim]
            s:       [1, s_dim]
        Returns:
            z_seq: [L, Z]  skill sequence
        """
        L = eps_seq.shape[0]
        z_seq = []

        for i in range(L):
            mu_z, sigma_z = self.skill_prior(s)
            z_i = mu_z + sigma_z * eps_seq[i:i + 1, :]  # [1, Z]
            z_seq.append(z_i.squeeze(0))

            h, s_prior_mean, _ = self.rssm.imagine_step(h, s, z_i)
            s = s_prior_mean

        return torch.stack(z_seq, dim=0)  # [L, Z]

    @torch.no_grad()
    def policy_action(self, state_vec, z_vec, deterministic=False):
        """
        Query the low-level policy for a single-step action.

        Args:
            state_vec: np.ndarray [obs_dim]
            z_vec:     torch.Tensor [z_dim]
            deterministic: if True, return mean action
        Returns:
            action: np.ndarray [action_dim]
        """
        state = torch.tensor(
            state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        z = z_vec.unsqueeze(0).to(self.device)

        mu, std = self.pi_theta(state, z)

        if deterministic:
            action = mu
        else:
            action = mu + std * torch.randn_like(mu)

        return action.squeeze(0).cpu().numpy()
