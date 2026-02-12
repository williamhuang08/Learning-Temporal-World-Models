""" Functions to Save GIFs of the Planned Trajectory (functions from prev utils in skill learning paper)"""

from PIL import Image
import matplotlib.pyplot as plt
import torch
import imageio
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random


def make_gif(frames, name):
    frames = [Image.fromarray(image) for image in frames]
    frame_one = frames[0]
    frame_one.save(name+'.gif', format="GIF", append_images=frames,save_all=True, duration=100)

def make_video(frames,name):
	writer = imageio.get_writer(name+'.mp4', fps=20)

	for im in frames:
		writer.append_data(im)
	writer.close()

def obs_to_state_vec(obs_dict):
    return np.concatenate([obs_dict["observation"], obs_dict["achieved_goal"]], axis=-1).astype(np.float32)

def xy_from_state(s):
    return s[..., -2:]

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

class SubtrajDataset(Dataset):
    """
    Loops over minari_dataset.iterate_episodes(), but keeps only episodes whose index is in episode_ids
    """
    def __init__(self, minari_dataset, T, episode_ids, stride=3):
        self.T = T
        self.items = [] 
        self.removed_items = [] 

        # Iterate all episodes but only process those whose global index is in episode_ids
        for ep_idx, ep in enumerate(minari_dataset.iterate_episodes()):
            if ep_idx not in set(episode_ids):
                continue
            obs = ep.observations["observation"]          
            ach = ep.observations["achieved_goal"]        
            act = ep.actions                               
            Ltot = len(obs)
            if Ltot < T + 1:
                continue

            state_ext = np.concatenate([obs, ach], axis=-1).astype(np.float32)
            for t in range(0, Ltot - T, stride):
                state_seq = state_ext[t:t+T]         
                s0 = state_seq[0]             
                action_seq = act[t:t+T].astype(np.float32)  
                sT = state_ext[t+T]           
                if np.linalg.norm(s0[-2:] - sT[-2:]) > 0.4:
                    self.items.append((s0, state_seq, action_seq, sT))
                else:
                    self.removed_items.append((s0, state_seq, action_seq, sT))

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, i):
        """standardize s0, state_sequence, and sT by (x - mean) / std"""
        
        s0, S, A, sT = self.items[i]
        if hasattr(self, "stats") and self.stats is not None:
            S_mean, S_std = self.stats
            S  = (S  - S_mean) / S_std
            s0 = (s0 - S_mean) / S_std
            sT = (sT - S_mean) / S_std
            A  = A
        return {
            "s0": torch.as_tensor(s0, dtype=torch.float32),
            "state_sequence": torch.as_tensor(S, dtype=torch.float32),
            "action_sequence": torch.as_tensor(A, dtype=torch.float32),
            "sT": torch.as_tensor(sT, dtype=torch.float32),
        }

def collate(batch):
    return {
        "s0": torch.stack([b["s0"] for b in batch], 0),
        "state_sequence": torch.stack([b["state_sequence"] for b in batch], 0),
        "action_sequence": torch.stack([b["action_sequence"] for b in batch], 0),
        "sT": torch.stack([b["sT"] for b in batch], 0),
    }