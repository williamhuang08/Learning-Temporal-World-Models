import torch
import numpy as np
import random
from torch.utils.data import Dataset


class DreamerSubtrajDataset(Dataset):
    """
    Extends the offline SubtrajDataset to include:
      - obs_seq:  o_0 ... o_H   (H+1 observations, for the state encoder)
      - act_seq:  a_0 ... a_{H-1} (H actions)
      - cumulative_reward: sum of per-step rewards over the H-step window
      - s0 / sT kept for backward compatibility

    Minari episodes expose .rewards alongside .observations and .actions.
    """

    def __init__(self, minari_dataset, H, episode_ids, stride=1, max_step_size=0.5):
        self.H = H
        self.items = []
        self.removed_items = []
        self.stats = None

        episode_id_set = set(episode_ids)
        for ep_idx, ep in enumerate(minari_dataset.iterate_episodes()):
            if ep_idx not in episode_id_set:
                continue

            obs = ep.observations["observation"]
            ach = ep.observations["achieved_goal"]
            act = ep.actions
            rew = ep.rewards
            Ltot = len(obs)

            if Ltot < H + 1:
                continue

            state_ext = np.concatenate([obs, ach], axis=-1).astype(np.float32)

            for t in range(0, Ltot - H, stride):
                obs_window = state_ext[t : t + H + 1]     # [H+1, obs_dim]
                act_window = act[t : t + H].astype(np.float32)  # [H, action_dim]
                cum_reward = float(rew[t : t + H].sum())

                # Teleportation filter (same as offline dataloader)
                xy = obs_window[:, -2:]
                dxy = np.linalg.norm(xy[1:] - xy[:-1], axis=-1)
                if not np.all(dxy <= max_step_size):
                    self.removed_items.append((obs_window, act_window, cum_reward))
                    continue

                self.items.append((obs_window, act_window, cum_reward))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        obs_window, act_window, cum_reward = self.items[i]

        if self.stats is not None:
            S_mean, S_std = self.stats
            obs_window = (obs_window - S_mean) / S_std

        return {
            "s0": torch.as_tensor(obs_window[0], dtype=torch.float32),
            "sT": torch.as_tensor(obs_window[-1], dtype=torch.float32),
            "obs_seq": torch.as_tensor(obs_window, dtype=torch.float32),         # [H+1, obs_dim]
            "act_seq": torch.as_tensor(act_window, dtype=torch.float32),          # [H, action_dim]
            "cumulative_reward": torch.tensor(cum_reward, dtype=torch.float32),
        }


def dreamer_collate(batch):
    return {
        "s0": torch.stack([b["s0"] for b in batch], 0),
        "sT": torch.stack([b["sT"] for b in batch], 0),
        "obs_seq": torch.stack([b["obs_seq"] for b in batch], 0),
        "act_seq": torch.stack([b["act_seq"] for b in batch], 0),
        "cumulative_reward": torch.stack([b["cumulative_reward"] for b in batch], 0),
    }


def compute_stats(ds):
    """Per-feature mean/std from all observation timesteps in the dataset."""
    all_obs = []
    for obs_window, _, _ in ds.items:
        all_obs.append(obs_window.reshape(-1, obs_window.shape[-1]))
    S = np.concatenate(all_obs, axis=0)
    return S.mean(0), S.std(0) + 1e-6
