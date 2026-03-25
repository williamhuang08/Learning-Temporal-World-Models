import torch
import numpy as np
import random
from torch.utils.data import Dataset


def _min_distance_to_goal_xy(achieved_xy, goal_xy):
    """achieved_xy: [T, 2], goal_xy: [2] -> scalar min_t ||achieved_xy[t] - goal_xy||_2"""
    d = np.linalg.norm(achieved_xy - goal_xy[None, :], axis=-1)
    return float(d.min())


class DreamerSubtrajDataset(Dataset):
    """
    Subtrajectories for Dreamer-style training:
      - obs_seq:  o_0 ... o_H   (H+1 observations, for the state encoder)
      - act_seq:  a_0 ... a_{H-1} (H actions)
      - goal_xy:  desired_goal xy at window start (raw, not normalized), shape [2]
      - min_goal_dist: min Euclidean distance to goal_xy over achieved xy at t=0..H
      - s0 / sT kept for convenience

    Requires Minari observations to include 'desired_goal' (AntMaze / goal-conditioned).
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
            if "desired_goal" not in ep.observations:
                raise KeyError(
                    "Episode observations must include 'desired_goal' for goal-conditioned rewards."
                )
            des = ep.observations["desired_goal"]
            Ltot = len(obs)

            if Ltot < H + 1:
                continue

            state_ext = np.concatenate([obs, ach], axis=-1).astype(np.float32)

            for t in range(0, Ltot - H, stride):
                obs_window = state_ext[t : t + H + 1]     # [H+1, obs_dim]
                act_window = act[t : t + H].astype(np.float32)  # [H, action_dim]
                goal_xy = np.asarray(des[t], dtype=np.float32).ravel()[:2]

                ach_xy = obs_window[:, -2:]
                min_goal_dist = _min_distance_to_goal_xy(ach_xy, goal_xy)

                # Teleportation filter (same as offline dataloader)
                xy = obs_window[:, -2:]
                dxy = np.linalg.norm(xy[1:] - xy[:-1], axis=-1)
                if not np.all(dxy <= max_step_size):
                    self.removed_items.append((obs_window, act_window, goal_xy, min_goal_dist))
                    continue

                self.items.append((obs_window, act_window, goal_xy, min_goal_dist))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        obs_window, act_window, goal_xy, min_goal_dist = self.items[i]

        if self.stats is not None:
            S_mean, S_std = self.stats
            obs_window = (obs_window - S_mean) / S_std

        return {
            "s0": torch.as_tensor(obs_window[0], dtype=torch.float32),
            "sT": torch.as_tensor(obs_window[-1], dtype=torch.float32),
            "obs_seq": torch.as_tensor(obs_window, dtype=torch.float32),         # [H+1, obs_dim]
            "act_seq": torch.as_tensor(act_window, dtype=torch.float32),          # [H, action_dim]
            "goal_xy": torch.as_tensor(goal_xy, dtype=torch.float32),             # [2]
            "min_goal_dist": torch.tensor(min_goal_dist, dtype=torch.float32),
        }


def dreamer_collate(batch):
    return {
        "s0": torch.stack([b["s0"] for b in batch], 0),
        "sT": torch.stack([b["sT"] for b in batch], 0),
        "obs_seq": torch.stack([b["obs_seq"] for b in batch], 0),
        "act_seq": torch.stack([b["act_seq"] for b in batch], 0),
        "goal_xy": torch.stack([b["goal_xy"] for b in batch], 0),
        "min_goal_dist": torch.stack([b["min_goal_dist"] for b in batch], 0),
    }


def compute_stats(ds):
    """Per-feature mean/std from all observation timesteps in the dataset."""
    all_obs = []
    for obs_window, _, _, _ in ds.items:
        all_obs.append(obs_window.reshape(-1, obs_window.shape[-1]))
    S = np.concatenate(all_obs, axis=0)
    return S.mean(0), S.std(0) + 1e-6

def make_episode_splits(minari_dataset, train=0.8, val=0.1, test=0.1, seed=0):
    """Return three lists of episode indices (train_ids, val_ids, test_ids)."""
    # Materialize all episodes once so we know how many there are
    episodes = list(minari_dataset.iterate_episodes())
    n = len(episodes)
    idxs = list(range(n))
    # Shuffle the indices
    random.Random(seed).shuffle(idxs)
    n_train = int(round(train * n))
    n_val = int(round(val * n))
    train_ids = idxs[:n_train]
    val_ids = idxs[n_train:n_train+n_val]
    test_ids = idxs[n_train+n_val:]
    return train_ids, val_ids, test_ids