
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

from skill_model import SkillPolicy, SkillPosterior, SkillPrior, TAWM
from dataloader import SubtrajDataset, collate, make_episode_splits
from utils import save_checkpoint, load_checkpoint

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
betas = np.array([1]) # increased value of beta term may be beneficial when prior is more expressive or else posterior begins running away from prior
# gammas = np.array([0.1])

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

S_mean, S_std = 0, 1

# pass stats into datasets
train_ds.stats = (S_mean, S_std)
val_ds.stats = (S_mean, S_std)

train_loader = DataLoader(train_ds, batch_size=B, shuffle=True,  collate_fn=collate, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=B, shuffle=False, collate_fn=collate, drop_last=False)

test_ds.stats = (S_mean, S_std)
test_loader = DataLoader(test_ds, batch_size=B, shuffle=False, collate_fn=collate, drop_last=False)

alpha = 1.0

def reparameterize(mean, std):
    eps = torch.randn_like(mean)
    return mean + std * eps

def get_E_loss(batch, beta):
    s0, S, A, sT = batch["s0"], batch["state_sequence"], batch["action_sequence"], batch["sT"]
    B, T, _ = S.shape
    denom = B * T

    # posterior
    mu_q, std_q = q_phi(S, A)
    z = reparameterize(mu_q, std_q)             

    # prior
    mu_pr, std_pr = p_omega(s0)                  

    # low-level policy likelihood
    z_bt = z.unsqueeze(1).expand(B, T, -1)
    mu_pi, std_pi = pi_theta(
        S.reshape(B * T, -1),
        z_bt.reshape(B * T, -1)
    )
    mu_pi = mu_pi.view(B, T, -1)
    std_pi = std_pi.view(B, T, -1)

    post_dist = Normal(mu_q, std_q)
    prior_dist = Normal(mu_pr, std_pr)
    a_dist = Normal(mu_pi, std_pi)

    log_pi = torch.sum(a_dist.log_prob(A)) / denom
    log_prior = torch.sum(prior_dist.log_prob(z)) / denom
    log_post = torch.sum(post_dist.log_prob(z)) / denom

    E_loss = -log_pi - beta * log_prior + beta * log_post
    return E_loss



def get_M_loss(batch, beta, alpha):
    s0, S, A, sT = batch["s0"], batch["state_sequence"], batch["action_sequence"], batch["sT"]
    B, T, _ = S.shape
    denom = B * T

    # posterior sample
    mu_q, std_q = q_phi(S, A)
    z = reparameterize(mu_q, std_q)

    # prior
    mu_pr, std_pr = p_omega(s0)

    # policy likelihood
    z_bt = z.unsqueeze(1).expand(B, T, -1)
    mu_pi, std_pi = pi_theta(
        S.reshape(B * T, -1),
        z_bt.reshape(B * T, -1)
    )
    mu_pi = mu_pi.view(B, T, -1)
    std_pi = std_pi.view(B, T, -1)

    # TAWM
    mu_T, std_T = p_psi(s0, z)

    sT_dist = Normal(mu_T, std_T)
    a_dist = Normal(mu_pi, std_pi)
    prior_dist = Normal(mu_pr, std_pr)

    sT_loss = -torch.sum(sT_dist.log_prob(sT)) / denom
    a_loss = -torch.sum(a_dist.log_prob(A)) / denom
    prior_loss = -torch.sum(prior_dist.log_prob(z)) / denom

    M_loss = alpha * sT_loss + a_loss + beta * prior_loss
    return M_loss


def get_eval_losses(batch, beta, alpha):
    s0, S, A, sT = batch["s0"], batch["state_sequence"], batch["action_sequence"], batch["sT"]
    B, T, _ = S.shape

    mu_q, std_q = q_phi(S, A)
    z = reparameterize(mu_q, std_q)

    z_bt = z.unsqueeze(1).expand(B, T, -1)
    mu_pi, std_pi = pi_theta(
        S.reshape(B * T, -1),
        z_bt.reshape(B * T, -1)
    )
    mu_pi = mu_pi.view(B, T, -1)
    std_pi = std_pi.view(B, T, -1)

    mu_pr, std_pr = p_omega(s0)
    mu_T, std_T = p_psi(s0, z)

    a_dist = Normal(mu_pi, std_pi)
    sT_dist = Normal(mu_T, std_T)
    post_dist = Normal(mu_q, std_q)
    prior_dist = Normal(mu_pr, std_pr)

    sT_loss = -torch.mean(torch.sum(sT_dist.log_prob(sT), dim=-1)) / T
    a_loss  = -torch.mean(torch.sum(a_dist.log_prob(A), dim=-1))
    kl_loss = torch.mean(torch.sum(kl_divergence(post_dist, prior_dist), dim=-1)) / T

    total = alpha * sT_loss + a_loss + beta * kl_loss
    return total, sT_loss, a_loss, kl_loss


@torch.no_grad()
def eval_epoch(val_loader, q_phi, pi_theta, p_psi, p_omega, alpha, beta, device):
    """Compute validation loss"""
    q_phi.eval()
    pi_theta.eval()
    p_psi.eval()
    p_omega.eval()
    loss_sum,policy_loss_sum, kl_loss_sum, state_decoder_loss_sum, n = 0.0, 0.0, 0.0, 0.0, 0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, state_decoder_loss, policy_loss, kl_loss =  get_eval_losses(batch, beta, alpha)
        loss_sum += float(loss.item())
        policy_loss_sum += float(policy_loss.item())
        kl_loss_sum += float(kl_loss.item())
        state_decoder_loss_sum += float(state_decoder_loss.item())

        n += 1
    if n == 0: 
        return None, None, None, None
    return loss_sum / n, policy_loss_sum / n, kl_loss_sum / n, state_decoder_loss_sum / n


def skill_model_training_with_val(
    save_path, # checkpoints/beta_gamma/
    train_loader, val_loader, 
    q_phi, pi_theta, p_psi, p_omega, 
    beta,alpha,
    lr=5e-5,
    e_steps=1, m_steps=1,
    epochs=50, steps=1, grad_clip=1.0, 
):
    q_phi.to(device)
    pi_theta.to(device)
    p_psi.to(device)
    p_omega.to(device)

    E_optimizer = torch.optim.Adam(q_phi.parameters(), lr=lr)
    M_optimizer = torch.optim.Adam(
        list(pi_theta.parameters()) + list(p_psi.parameters()) + list(p_omega.parameters()),
        lr=lr
    )

    tr_e, tr_m, va = [], [], []


    best_val_loss = float("inf")

    for epoch in range(1, epochs+1):
        q_phi.train()
        pi_theta.train()
        p_psi.train()
        p_omega.train()
        e_run = m_run = 0.0 # Running e_loss, m_loss, in current epoch

        nb = 0

        for batch in train_loader:
            # Rebuilds dictionary but moves tensors to the device
            batch = {k: v.to(device) for k, v in batch.items()}
            nb += 1

            for _ in range(steps):
                # E step: update q_phi
                # train the posterior while freezing other parameters
                for _ in range(e_steps):
                    E_optimizer.zero_grad(set_to_none=True)
                    e_loss = get_E_loss(batch, beta)
                    e_loss.backward()
                    E_optimizer.step()
                e_run += float(e_loss.item())


                # M step: update theta, psi, omega
                # Freeze posterior weights, update all other weights
                for _ in range(m_steps):
                    # Reset gradients
                    M_optimizer.zero_grad(set_to_none=True)
                    m_loss = get_M_loss(batch, beta, alpha)
                    m_loss.backward()
                    M_optimizer.step()
                m_run += float(m_loss.item())

        # Calculate the average losses over all the batches in the epoch
        e_epoch = e_run / max(1, nb)
        m_epoch = m_run / max(1, nb)
        tr_e.append(e_epoch)
        tr_m.append(m_epoch)

        if epoch % 50 == 0:
            save_checkpoint(f"{save_path}/epochs/em_epoch{epoch}_beta{beta}.pth", q_phi, pi_theta, p_psi, p_omega, B, T, Z_DIM, NUM_NEURONS, device)

        # validation
        v_loss, v_policy_loss, v_kl_loss, v_state_decoder_loss = eval_epoch(val_loader, q_phi, pi_theta, p_psi, p_omega, alpha, beta, device)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_checkpoint(f"{save_path}/em_epoch{epoch}_beta{beta}_best.pth", q_phi, pi_theta, p_psi, p_omega, B, T, Z_DIM, NUM_NEURONS, device)
        va.append(v_loss)


        print(f"[Epoch {epoch:03d}/{epochs}] "
              f"train E:{e_epoch:.4f}  M:{m_epoch:.4f} "
              f"| val loss:{v_loss:.4f}")

        wandb.log({
            "train/E_loss": e_epoch,
            "train/M_loss": m_epoch,
            "val/loss": v_loss,
            "val/policy_loss": v_policy_loss,
            "val/kl_loss": v_kl_loss,
            "val/state_decoder_loss": v_state_decoder_loss,
            "epoch": epoch
        }, step=epoch)

    plt.figure(figsize=(7.5,4.5))
    plt.plot(tr_e, label="Train E loss")
    plt.plot(tr_m, label="Train M loss")
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

    return {"train_E": tr_e, "train_M": tr_m, "val_E": va}

epochs = 1000
lr=5e-5

for beta in betas:
    # Initialize the models
    q_phi = SkillPosterior(state_dim=state_dim, action_dim=action_dim).to(device)
    pi_theta = SkillPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
    p_psi = TAWM(state_dim=state_dim).to(device)
    p_omega = SkillPrior(state_dim=state_dim).to(device)

    wandb.init(
        project="tawm-skill-learning",
        name=f"antmaze-medium-detached-em-epoch{epochs}-beta{beta}",
        config=dict(
            B=B, T=T, Z_DIM=Z_DIM, NUM_NEURONS=NUM_NEURONS,
            e_lr=lr, m_lr=lr, e_steps=1, m_steps=1,
            dataset="D4RL/antmaze/medium-diverse-v1",
            device=device
        )
    )

    wandb.watch([q_phi, pi_theta, p_psi, p_omega], log="gradients", log_freq=200)
    save_path = f"checkpoints/em_{beta}"
    os.makedirs(os.path.join(save_path, "epochs"), exist_ok=True)
    curves = skill_model_training_with_val(save_path, train_loader, val_loader, q_phi, pi_theta, p_psi, p_omega, beta, alpha, epochs=epochs, lr=lr, steps=1)

    wandb.finish()




