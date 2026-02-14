from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import random
import minari
import torch
import math
from skill_model import SkillPolicy, SkillPosterior, MoGSkillPrior, TAWM

ant_maze_dataset = minari.load_dataset('D4RL/antmaze/medium-diverse-v1')
filename = 'kl_balancing_MoG/beta_gamma/antmaze_diverse_detached_klbalance_mog_100_beta0.001_gamma1.pth'
PATH = '../checkpoints/' + filename
device = "cuda" if torch.cuda.is_available() else "cpu"
T, B = 40, 100
state_dim, action_dim = 29, 8

def load_checkpoint(path, q_phi, pi_theta, p_psi, p_omega, strict=True):
    ckpt = torch.load(path, map_location=torch.device('cpu'))
    q_phi.load_state_dict(ckpt["q_phi"], strict=strict)
    pi_theta.load_state_dict(ckpt["pi_theta"], strict=strict)
    p_psi.load_state_dict(ckpt["p_psi"], strict=strict)
    p_omega.load_state_dict(ckpt["p_omega"], strict=strict)
    stats = ckpt.get("S_stats", None)
    if stats is not None:
        global S_mean, S_std
        S_mean, S_std = stats["mean"], stats["std"]
    print(f"[checkpoint] loaded <- {path}")
    return ckpt

q_phi = SkillPosterior(state_dim=state_dim, action_dim=action_dim).to(device)
pi_theta = SkillPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
p_psi = TAWM(state_dim=state_dim).to(device)
p_omega = MoGSkillPrior(state_dim=state_dim).to(device)

_ = load_checkpoint(PATH, q_phi, pi_theta, p_psi, p_omega)


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


# Pick indices for train/test/split
train_ids, val_ids, test_ids = make_episode_splits(ant_maze_dataset, train=0.8, val=0.0, test=0.2, seed=0)
print(f"train:{len(train_ids)}  val:{len(val_ids)}  test:{len(test_ids)}")

# Datasets from episode subsets
train_ds = SubtrajDataset(ant_maze_dataset, T=T, episode_ids=train_ids, stride=3)
val_ds = SubtrajDataset(ant_maze_dataset, T=T, episode_ids=val_ids,   stride=3)
test_ds = SubtrajDataset(ant_maze_dataset, T=T, episode_ids=test_ids,  stride=3)  

print(f"train:{len(train_ds)}  val:{len(val_ds)}  test:{len(test_ds)}")

all_xy = []
for ep in ant_maze_dataset.iterate_episodes():
    xy = ep.observations["achieved_goal"][:, :2]  
    all_xy.append(xy)

all_xy = np.concatenate(all_xy, axis=0)

# find per-feature mean and std from all state_sequence timesteps in train_ds
def compute_stats(ds):
    Ss = []
    for item in ds.items:
        Ss.append(item[1])  # state_sequence [T,29]
    S = np.concatenate([x.reshape(-1, x.shape[-1]) for x in Ss], axis=0)
    S_mean, S_std = S.mean(0), S.std(0) + 1e-6
    return (S_mean, S_std)

S_mean, S_std = 0, 1

# pass stats into datasets
train_ds.stats = (S_mean, S_std)
val_ds.stats = (S_mean, S_std)

train_loader = DataLoader(train_ds, batch_size=B, shuffle=True,  collate_fn=collate, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=B, shuffle=False, collate_fn=collate, drop_last=False)

test_ds.stats = (S_mean, S_std)
test_loader = DataLoader(test_ds, batch_size=B, shuffle=False, collate_fn=collate, drop_last=False)

def _get_start_xy_from_item(item):
    s0, S, A, sT = item
    return s0[-2:].astype(np.float32), s0.astype(np.float32), S.astype(np.float32)

def build_xy_cache(minari_dataset):
    episodes_xy = []
    starts = []
    all_xy = []
    start_s0 = []

    # scan all subtrajectories
    for i, item in enumerate(train_ds.items):
        xy, s0, states = _get_start_xy_from_item(item)
        episodes_xy.append(states[:, -2:])
        starts.append(xy)
        start_s0.append(s0)
        all_xy.append(states[:, -2:])

    return episodes_xy, np.stack(starts, axis=0), np.stack(start_s0, axis=0), np.concatenate(all_xy, axis=0)

train_episodes_xy, train_episodes_start_xy, train_episodes_start, train_all_xy = build_xy_cache(train_ds)

def pick_nearby_ep(episodes_start_xy, current_xy):
    # want to find distances betwqeen all elements inthe batch with all elements in the dataset
    # goal shape # [B, 100]
    # [B, 1, 2] and [1, 100, 2] 
    n_neighbors = 10
    d2 = torch.sum((episodes_start_xy[:, None, :] - current_xy[None, :, :])**2, dim=2)
    idx = torch.empty(current_xy.shape[0], n_neighbors).to(torch.int64)
    # rng = np.random.default_rng()
    for i in range(current_xy.shape[0]):
        idx[i] = torch.argsort(d2[i])[:n_neighbors]
    print(idx)
    return idx

LOG_2PI = math.log(2.0 * math.pi)

def log_normal_diag(z, mu, std):
    var = std * std + 1e-8
    return -0.5 * (((z - mu) ** 2) / var + torch.log(var) + LOG_2PI).sum(dim=-1)

def log_mog_diag(z, logits, mu, std):
    log_pi = F.log_softmax(logits, dim=-1)                
    z_  = z[:, :, None, :]                                 
    mu_ = mu[:, None, :, :]                                
    st_ = std[:, None, :, :]                               
    log_n = log_normal_diag(z_, mu_, st_)                  
    return torch.logsumexp(log_pi[:, None, :] + log_n, dim=-1)  


def mc_kl_neighbors(curr_logits, curr_mu, curr_std, neigh_logits, neigh_mu, neigh_std):
    num_samples = 5
    B, n_neighbors, K = neigh_logits.shape
    z_dim = curr_mu.shape[-1]

    cat = torch.distributions.Categorical(logits=curr_logits)
    k_idx = cat.sample((num_samples,)).transpose(0, 1)               

    mu_sel = curr_mu.gather(1, k_idx[..., None].expand(B, num_samples, z_dim))   
    std_sel = curr_std.gather(1, k_idx[..., None].expand(B, num_samples, z_dim))  

    eps = torch.randn_like(mu_sel)
    z = mu_sel + std_sel * eps

    logp = log_mog_diag(z, curr_logits, curr_mu, curr_std)


    neigh_logits = neigh_logits.reshape(B * n_neighbors, -1)
    neigh_mu = neigh_mu.reshape(B * n_neighbors, K, -1)
    neigh_std = neigh_std.reshape(B * n_neighbors, K, -1)

    z_resp = z[:, None, :, :].expand(B, n_neighbors, num_samples, z_dim).reshape(B * n_neighbors, num_samples, z_dim)
    logq = log_mog_diag(z_resp, neigh_logits, neigh_mu, neigh_std)

    logq = logq.reshape(B, n_neighbors, num_samples)
    kl = (logp[:, None, :] - logq).mean(dim=-1)
    return kl.mean()

for batch in train_loader:
        # Rebuilds dictionary but moves tensors to the device
        batch = {k: v.to(device) for k, v in batch.items()}
        s0, S, A, sT = batch["s0"], batch["state_sequence"], batch["action_sequence"], batch["sT"]
        print(s0.shape)
        s0_xy = s0[:, -2:]
        indices = pick_nearby_ep(torch.tensor(train_episodes_start_xy), s0_xy)
        neighbor_s0 = train_episodes_start[indices]

        mu_q, std_q = q_phi(S, A)                      
        z = mu_q + std_q * torch.randn_like(mu_q)

        curr_logits, curr_mu_pr, curr_std_pr = p_omega(s0)
        B, n_neighbors, state_dim = neighbor_s0.shape
        neigh_logits, neigh_mu_pr, neigh_std_pr = p_omega(torch.tensor(neighbor_s0.reshape(B * n_neighbors, 29)))

        neigh_logits = neigh_logits.reshape(B, n_neighbors, -1)
        neigh_mu_pr = neigh_mu_pr.reshape(B, n_neighbors, -1)
        neigh_std_pr = neigh_std_pr.reshape(B, n_neighbors, -1)

        knn_kll = mc_kl_neighbors(curr_logits, curr_mu_pr, curr_std_pr, neigh_logits, neigh_mu_pr, neigh_std_pr)
        print(knn_kll.sum()/(B * 10))

        # now compute the kl divergence between current MoG and all 20 MoG
        # print(log_density.shape)
        # print(curr_log_density.shape)
        break