
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl_divergence
from torch.distributions.transforms import TanhTransform
import gymnasium as gym39
import mujoco
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import minari
from torch.utils.data import Dataset, DataLoader
import wandb
import os
import argparse

from skill_model import SkillPolicy, SkillPosterior, SkillPrior, TAWM, MoGSkillPrior

device = "cuda" if torch.cuda.is_available() else "cpu"

# parser = argparse.ArgumentParser()
# parser.add_argument("--beta", type=float, required=True)
# parser.add_argument("--gamma", type=float, required=True)
# args = parser.parse_args()

# beta = args.beta
# gamma = args.gamma

# beta = 1 
# gamma = 0.01
# betas = np.geomspace(0.001, 1, num=4) 
# gammas = np.geomspace(0.001, 1, num=4)
betas = np.array([0.001])
gammas = np.array([1])

# According to the paper, each layer contains 256 neurons
NUM_NEURONS = 256

Z_DIM = 256

# Loads the AntMaze dataset in Minari format
ant_maze_dataset = minari.load_dataset('D4RL/antmaze/medium-diverse-v1')

print(ant_maze_dataset[0].actions.shape)
print(ant_maze_dataset[0].observations.keys())
print(ant_maze_dataset[0].observations["observation"].shape)
print(ant_maze_dataset[0].observations["achieved_goal"].shape)

# B, the number of subtrajectories per batch (from paper)
B = 100

# T, the length of each subtrajectory (from paper)
T = 40

# AntMaze state and action dims (from Minari)
state_dim = 29
action_dim = 8

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
    def __init__(self, minari_dataset, T, episode_ids, stride=1):
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
                sT = state_seq[-1]

                state_seq_xy = state_seq[:, -2:]
                dxy = state_seq_xy[1:] - state_seq_xy[:-1] # rows 1 -> T-1 - rows 0 -> T - 2
                step_size = np.linalg.norm(dxy, axis=-1)
                fine = np.all(step_size <= 0.5)
                if not fine:
                    self.removed_items.append((s0, state_seq, action_seq, sT))
                else:
                    self.items.append((s0, state_seq, action_seq, sT))

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
train_ids, val_ids, test_ids = make_episode_splits(ant_maze_dataset, train=0.8, val=0.1, test=0.1, seed=0)
print(f"train:{len(train_ids)}  val:{len(val_ids)}  test:{len(test_ids)}")

# Datasets from episode subsets
train_ds = SubtrajDataset(ant_maze_dataset, T=T, episode_ids=train_ids, stride=1)
val_ds = SubtrajDataset(ant_maze_dataset, T=T, episode_ids=val_ids,   stride=1)
test_ds = SubtrajDataset(ant_maze_dataset, T=T, episode_ids=test_ids,  stride=1)  

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

alpha = 1.0

def compute_loss(batch):
    s0, S, A, sT = batch["s0"], batch["state_sequence"], batch["action_sequence"], batch["sT"]
    B, T, _  = S.shape
    denom = B * T

    # State encoder
    mu_q, std_q = q_phi(S, A)                      
    z = mu_q + std_q * torch.randn_like(mu_q)

    # Low-level policy pi_theta(a|s,z)
    z_bt = z.unsqueeze(1).expand(B, T, -1)         
    mu_pi, std_pi = pi_theta(S.reshape(B*T, -1), z_bt.reshape(B*T, -1))
    mu_pi, std_pi = mu_pi.view(B, T, -1), std_pi.view(B, T, -1)
    a_dist  = Independent(Normal(mu_pi, std_pi), 1)        

    # Compute policy loss
    a_loss  = -a_dist.log_prob(A).sum() / denom
    
    mu_pr, std_pr = p_omega(s0)                              
    prior_dist = Independent(Normal(mu_pr, std_pr), 1)
    log_prior = prior_dist.log_prob(z).sum() / denom
    post_dist = Independent(Normal(mu_q,  std_q),  1)
    log_post  = post_dist.log_prob(z).sum() / denom

    # w/0 KL balancing
    kl_loss = - log_prior + log_post

    # Detach gradient
    z_detached = z.detach()

    # TAWM over terminal state
    mu_T, std_T = p_psi(s0, z_detached)                            
    sT_dist = Independent(Normal(mu_T, std_T), 1)

    # State decoder loss
    sT_loss = -sT_dist.log_prob(sT).sum() / denom

    # Overall loss (naive VI loss from paper)
    loss = alpha * sT_loss + a_loss + beta * kl_loss
    return {
        "loss": loss,
        "policy_loss": a_loss,
        "kl_loss": kl_loss,
        "state_decoder_loss": sT_loss
    }


def compute_loss_klbalancing(batch, beta, gamma):
    s0, S, A, sT = batch["s0"], batch["state_sequence"], batch["action_sequence"], batch["sT"]
    B, T, _  = S.shape
    denom = B * T

    # State encoder
    mu_q, std_q = q_phi(S, A)                      
    z = mu_q + std_q * torch.randn_like(mu_q)

    # Low-level policy pi_theta(a|s,z)
    z_bt = z.unsqueeze(1).expand(B, T, -1)         
    mu_pi, std_pi = pi_theta(S.reshape(B*T, -1), z_bt.reshape(B*T, -1))
    mu_pi, std_pi = mu_pi.view(B, T, -1), std_pi.view(B, T, -1)
    a_dist  = Independent(Normal(mu_pi, std_pi), 1)        

    # Compute policy loss
    a_loss  = -a_dist.log_prob(A).sum() / denom

    mu_pr, std_pr = p_omega(s0)                              
    prior_dist = Independent(Normal(mu_pr, std_pr), 1)
    log_prior = prior_dist.log_prob(z).sum() / denom
    post_dist = Independent(Normal(mu_q,  std_q),  1)
    log_post  = post_dist.log_prob(z).sum() / denom

    kl_post = (log_post - log_prior.detach()) # post to match prior

    # Detach gradient
    z_detached = z.detach()

    # w/ KL balancing
    log_prior_det = prior_dist.log_prob(z_detached).sum() / denom
    kl_prior = (log_post.detach() - log_prior_det) # prior to match post
    kl_loss = beta * kl_post + gamma * kl_prior

    # TAWM over terminal state
    mu_T, std_T = p_psi(s0, z_detached)                            
    sT_dist = Independent(Normal(mu_T, std_T), 1)

    # State decoder loss
    sT_loss = -sT_dist.log_prob(sT).sum() / denom

    # Overall loss (naive VI loss from paper)
    loss = alpha * sT_loss + a_loss +  kl_loss
    return {
        "loss": loss,
        "policy_loss": a_loss,
        "kl_loss": kl_loss,
        "state_decoder_loss": sT_loss
    }


def compute_loss_klbalancing_mog(batch, beta, gamma):
    s0, S, A, sT = batch["s0"], batch["state_sequence"], batch["action_sequence"], batch["sT"]
    B, T, _  = S.shape
    denom = B * T

    # State encoder
    mu_q, std_q = q_phi(S, A)                      
    z = mu_q + std_q * torch.randn_like(mu_q)

    # Low-level policy pi_theta(a|s,z)
    z_bt = z.unsqueeze(1).expand(B, T, -1)         
    mu_pi, std_pi = pi_theta(S.reshape(B*T, -1), z_bt.reshape(B*T, -1))
    mu_pi, std_pi = mu_pi.view(B, T, -1), std_pi.view(B, T, -1)
    a_dist  = Independent(Normal(mu_pi, std_pi), 1)        

    # Compute policy loss
    a_loss  = -a_dist.log_prob(A).sum() / denom

    log_prior = p_omega.log_prob(z, s0).sum() / denom
    post_dist = Independent(Normal(mu_q,  std_q),  1)
    log_post  = post_dist.log_prob(z).sum() / denom

    kl_post = (log_post - log_prior.detach()) # post to match prior

    # Detach gradient
    z_detached = z.detach()

    # w/ KL balancing
    log_prior_det = p_omega.log_prob(z_detached, s0).sum() / denom
    kl_prior = (log_post.detach() - log_prior_det) # prior to match post
    kl_loss = beta * kl_post + gamma * kl_prior

    # TAWM over terminal state
    mu_T, std_T = p_psi(s0, z_detached)                            
    sT_dist = Independent(Normal(mu_T, std_T), 1)

    # State decoder loss
    sT_loss = -sT_dist.log_prob(sT).sum() / denom

    # Overall loss (naive VI loss from paper)
    loss = alpha * sT_loss + a_loss +  kl_loss
    return {
        "loss": loss,
        "policy_loss": a_loss,
        "kl_loss": kl_loss,
        "state_decoder_loss": sT_loss
    }

@torch.no_grad()
def eval_epoch(val_loader, q_phi, pi_theta, p_psi, p_omega, beta, gamma, device):
    """Compute validation loss"""
    q_phi.eval()
    pi_theta.eval()
    p_psi.eval()
    p_omega.eval()
    loss_sum,policy_loss_sum, kl_loss_sum, state_decoder_loss_sum, n = 0.0, 0.0, 0.0, 0.0, 0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        terms = compute_loss_klbalancing_mog(batch, beta, gamma)
        loss = terms["loss"]
        policy_loss = terms["policy_loss"]
        kl_loss = terms["kl_loss"]
        state_decoder_loss = terms["state_decoder_loss"]
        loss_sum += float(loss.item())
        policy_loss_sum += float(policy_loss.item())
        kl_loss_sum += float(kl_loss.item())
        state_decoder_loss_sum += float(state_decoder_loss.item())

        n += 1
    if n == 0: 
        return None, None, None, None
    return loss_sum / n, policy_loss_sum / n, kl_loss_sum / n, state_decoder_loss_sum / n

def skill_model_training_with_val(
    train_loader, val_loader,
    q_phi, pi_theta, p_psi, p_omega, beta, gamma,
    lr=5e-5,
    epochs=50, steps=1, grad_clip=1.0
):
    q_phi.to(device)
    pi_theta.to(device)
    p_psi.to(device)
    p_omega.to(device)

    opt = torch.optim.Adam(list(q_phi.parameters()) + list(pi_theta.parameters()) + list(p_psi.parameters()) + list(p_omega.parameters()), lr=lr)

    tr, va = [], []

    for epoch in range(1, epochs+1):
        q_phi.train()
        pi_theta.train()
        p_psi.train()
        p_omega.train()
        loss_run, policy_loss_run, kl_loss_run, state_decoder_loss_run = 0.0, 0.0, 0.0, 0.0 # Running loss in current epoch

        nb = 0

        for batch in train_loader:
            # Rebuilds dictionary but moves tensors to the device
            batch = {k: v.to(device) for k, v in batch.items()}
            nb += 1

            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                terms = compute_loss_klbalancing_mog(batch, beta, gamma)
                loss = terms["loss"]
                policy_loss = terms["policy_loss"]
                kl_loss = terms["kl_loss"]
                state_decoder_loss = terms["state_decoder_loss"]
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(list(q_phi.parameters()) + list(pi_theta.parameters()) + list(p_psi.parameters()) + list(p_omega.parameters()), grad_clip)
                opt.step()
            loss_run += float(loss.item())
            policy_loss_run += float(policy_loss.item())
            kl_loss_run += float(kl_loss.item())
            state_decoder_loss_run += float(state_decoder_loss.item())


        # Calculate the average losses over all the batches in the epoch
        loss_epoch = loss_run / max(1, nb)
        policy_loss_epoch = policy_loss_run / max(1, nb)
        kl_loss_epoch = kl_loss_run / max(1, nb)
        state_decoder_loss_epoch = state_decoder_loss_run / max(1, nb)

        tr.append(loss_epoch)

        # validation
        v_loss, v_policy_loss, v_kl_loss, v_state_decoder_loss = eval_epoch(val_loader, q_phi, pi_theta, p_psi, p_omega, beta, gamma, device)
        va.append(v_loss)

        print(f"[Epoch {epoch:03d}/{epochs}] "
              f"train loss:{loss_epoch:.4f} "
              f"| val loss:{v_loss:.4f}")

        wandb.log({
            "train/loss": loss_epoch,
            "train/policy_loss": policy_loss_epoch,
            "train/kl_loss": kl_loss_epoch,
            "train/state_decoder_loss": state_decoder_loss_epoch,
            "val/loss": v_loss,
            "val/policy_loss": v_policy_loss,
            "val/kl_loss": v_kl_loss,
            "val/state_decoder_loss": v_state_decoder_loss,
            "epoch": epoch
        }, step=epoch)

    plt.figure(figsize=(7.5,4.5))
    plt.plot(tr, label="Train loss")
    if all(v is not None for v in va):
        plt.plot(va, label="Val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("EM training: train vs. val losses")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    fig = plt.gcf()
    wandb.log({"plots/loss_curves": wandb.Image(fig)}, step=epoch)
    plt.close(fig)

    return {"train_loss": tr, "val_E": va}

# Load and save the model to a .pth file
def save_checkpoint(path, q_phi, pi_theta, p_psi, p_omega):
    ckpt = {
        "q_phi": q_phi.state_dict(),
        "pi_theta": pi_theta.state_dict(),
        "p_psi": p_psi.state_dict(),
        "p_omega": p_omega.state_dict(),
        "S_stats": {"mean": S_mean, "std": S_std},
        "config": dict(B=B, T=T, Z_DIM=Z_DIM, NUM_NEURONS=NUM_NEURONS,device=str(device))
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)
    print(f"checkpoint saved -> {path}")

def load_checkpoint(path, q_phi, pi_theta, p_psi, p_omega, strict=True):
    ckpt = torch.load(path)
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

epochs = 50
lr=5e-5

for beta in betas:
    for gamma in gammas:
        # Initialize the models
        q_phi = SkillPosterior(state_dim=state_dim, action_dim=action_dim).to(device)
        pi_theta = SkillPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
        p_psi = TAWM(state_dim=state_dim).to(device)
        p_omega = MoGSkillPrior(state_dim=state_dim).to(device)
        # p_omega = SkillPrior(state_dim=state_dim).to(device)

        wandb.init(
            project="tawm-skill-learning",
            name=f"antmaze-medium-detached-klbalance-mog-epoch{epochs}-beta{beta}-gamma{gamma}",
            config=dict(
                B=B, T=T, Z_DIM=Z_DIM, NUM_NEURONS=NUM_NEURONS,
                e_lr=5e-5, m_lr=5e-5, e_steps=1, m_steps=1,
                dataset="D4RL/antmaze/medium-diverse-v1",
                device=device
            )
        )

        wandb.watch([q_phi, pi_theta, p_psi, p_omega], log="gradients", log_freq=200)

        curves = skill_model_training_with_val(train_loader, val_loader, q_phi, pi_theta, p_psi, p_omega, beta, gamma, epochs=epochs, lr=lr, steps=1)

        wandb.finish()

        save_checkpoint(f"checkpoints/antmaze_diverse_detached_klbalance_mog_50_beta{beta}_gamma{gamma}.pth", q_phi, pi_theta, p_psi, p_omega)




