
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
from dataloader import SubtrajDataset, collate, make_episode_splits
from utils import save_checkpoint, load_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"

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
beta  = 1.0

def e_terms(batch):
    s0, S, A = batch["s0"], batch["state_sequence"], batch["action_sequence"]
    B, T, _  = S.shape
    denom = B * T

    # Posterior q_phi(z|tau)
    mu_q, std_q = q_phi(S, A) # [B, Z_DIM]
    z = mu_q + std_q * torch.randn_like(mu_q) # [B, Z_DIM]

    # Low-level policy pi_theta(a|s,z)
    z_bt = z.unsqueeze(1).expand(B, T, -1) # [B, T, Z_DIM]
    mu_pi, std_pi = pi_theta(
        S.reshape(B*T, -1),
        z_bt.reshape(B*T, -1)
    )
    mu_pi, std_pi = mu_pi.view(B, T, -1), std_pi.view(B, T, -1)

    # Use plain Normal for skill prior
    a_dist = Independent(Normal(mu_pi, std_pi), 1) # sum over action dims
    post_dist = Independent(Normal(mu_q,  std_q),  1)

    # log_pi summed over t & batch
    log_pi = a_dist.log_prob(A).sum() / denom # [B,T] 
    log_prior = p_omega.log_prob(z, s0).sum() / denom
    log_post = post_dist.log_prob(z).sum() / denom

    E_loss = -log_pi - beta * log_prior + beta * log_post
    return {
        "e_loss": E_loss,
        "log_pi": log_pi,
        "log_prior": log_prior,
        "log_post": log_post
    }


def m_terms(batch):
    s0, S, A, sT = batch["s0"], batch["state_sequence"], batch["action_sequence"], batch["sT"]
    B, T, _  = S.shape
    denom = B * T

    # Posterior and sampled z
    mu_q, std_q = q_phi(S, A)
    z = mu_q + std_q * torch.randn_like(mu_q)

    z_bt = z.unsqueeze(1).expand(B, T, -1)
    mu_pi, std_pi = pi_theta(
        S.reshape(B*T, -1),
        z_bt.reshape(B*T, -1)
    )
    mu_pi, std_pi = mu_pi.view(B, T, -1), std_pi.view(B, T, -1)

    a_dist = Independent(Normal(mu_pi, std_pi), 1)

    # TAWM over terminal state
    mu_T, std_T = p_psi(s0, z) # [B, state_dim]
    sT_dist = Independent(Normal(mu_T, std_T), 1)

    # Prior over z given s0
    mu_pr, std_pr = p_omega(s0)
    prior_dist = Independent(Normal(mu_pr, std_pr), 1)

    sT_loss = -sT_dist.log_prob(sT).sum() / denom
    a_loss = -a_dist.log_prob(A).sum() / denom
    prior_loss = -p_omega.log_prob(z, s0).sum() / denom

    M_loss = alpha * sT_loss + a_loss + beta * prior_loss
    return {
        "m_loss": M_loss,
        "sT_loss": sT_loss,
        "a_loss": a_loss,
        "prior_loss": prior_loss
    }

def get_losses(self,states,actions):
    '''
    Computes various components of the loss:
    L = E_q [log P(s_T|s_0,z)] 
        + E_q [sum_t=0^T P(a_t|s_t,z)] 
        - D_kl(q(z|s_0,...,s_T,a_0,...,a_T)||P(z_0|s_0))
    Distributions we need:
    '''


    s_T_mean, s_T_sig, a_means, a_sigs, z_post_means, z_post_sigs  = self.forward(states,actions)

    s_T_dist = Normal.Normal(s_T_mean, s_T_sig )
    if self.decoder.ll_policy.a_dist == 'normal' or 'autoregressive':
        a_dist = Normal.Normal(a_means, a_sigs)
    elif self.decoder.ll_policy.a_dist == 'tanh_normal':
        base_dist = Normal.Normal(a_means, a_sigs)
        transform = torch.distributions.transforms.TanhTransform()
        a_dist = TransformedDistribution(base_dist, [transform])
    else:
        assert False
    z_post_dist = Normal.Normal(z_post_means, z_post_sigs)
    # z_prior_means = torch.zeros_like(z_post_means)
    # z_prior_sigs = torch.ones_like(z_post_sigs)
    z_prior_means,z_prior_sigs = self.prior(states[:,0:1,:]) 
    z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs) 

    # loss terms corresponding to -logP(s_T|s_0,z) and -logP(a_t|s_t,z)
    T = states.shape[1]
    s_T = states[:,-1:,:]  
    s_T_loss = -torch.mean(torch.sum(s_T_dist.log_prob(s_T),   dim=-1))/T # divide by T because all other losses we take mean over T dimension, effectively dividing by T
    a_loss   = -torch.mean(torch.sum(a_dist.log_prob(actions), dim=-1)) 
    s_T_ent  = torch.mean(torch.sum(s_T_dist.entropy(),       dim=-1))/T
    # print('a_sigs: ', a_sigs)
    # print('a_dist.log_prob(actions)[0,:,:]: ',a_dist.log_prob(actions)[0,:,:])
    # loss term correpsonding ot kl loss between posterior and prior
    # kl_loss = torch.mean(torch.sum(F.kl_div(z_post_dist, z_prior_dist),dim=-1))
    kl_loss = torch.mean(torch.sum(KL.kl_divergence(z_post_dist, z_prior_dist), dim=-1))/T # divide by T because all other losses we take mean over T dimension, effectively dividing by T

    loss_tot = self.alpha*s_T_loss + a_loss + self.beta*kl_loss + self.ent_pen*s_T_ent
    # loss_tot = s_T_loss + kl_loss

    return  loss_tot, s_T_loss, a_loss, kl_loss, s_T_ent


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
    save_path, # checkpoints/beta_gamma/
    train_loader, val_loader, 
    q_phi, pi_theta, p_psi, p_omega, beta, gamma,
    e_lr=5e-5, m_lr=5e-5,
    e_steps=1, m_steps=1,
    epochs=50, steps=1, grad_clip=1.0, 
):
    q_phi.to(device)
    pi_theta.to(device)
    p_psi.to(device)
    p_omega.to(device)

    e_opt = torch.optim.Adam(q_phi.parameters(), lr=e_lr)
    m_opt = torch.optim.Adam(list(pi_theta.parameters()) + list(p_psi.parameters()) + list(p_omega.parameters()), lr=m_lr)

    tr_e, tr_m, va_e, va_m = [], [], [], []

    best_val_loss = float("inf")

    for epoch in range(1, epochs+1):
        q_phi.train()
        pi_theta.train()
        p_psi.train()
        p_omega.train()
        e_run, m_run, policy_loss_run, kl_loss_run, state_decoder_loss_run = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 # Running loss in current epoch

        nb = 0

        for batch in train_loader:
            # Rebuilds dictionary but moves tensors to the device
            batch = {k: v.to(device) for k, v in batch.items()}
            nb += 1

            # E step: update q_phi
            # train the posterior while freezing other parameters
            q_phi.train()
            pi_theta.eval()
            p_psi.eval()
            p_omega.eval()
            for p in q_phi.parameters(): 
                p.requires_grad_(True)
            for m in (pi_theta, p_psi, p_omega):
                for p in m.parameters(): 
                    p.requires_grad_(False)

            for _ in range(e_steps):
                e_opt.zero_grad(set_to_none=True)
                terms = e_terms(batch)
                e_loss = terms["e_loss"]
                e_loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(q_phi.parameters(), grad_clip)
                e_opt.step()
            e_run += float(e_loss.item())

            # M step: update theta, psi, omega
            # Freeze posterior weights, update all other weights

            q_phi.eval()
            pi_theta.train()
            p_psi.train()
            p_omega.train()
            for p in q_phi.parameters(): 
                p.requires_grad_(False)
            for m in (pi_theta, p_psi, p_omega):
                for p in m.parameters(): 
                    p.requires_grad_(True)

            for _ in range(m_steps):
                # Reset gradients
                m_opt.zero_grad(set_to_none=True)
                terms = m_terms(batch)
                m_loss = terms["m_loss"]
                m_loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(list(pi_theta.parameters()) + list(p_psi.parameters()) + list(p_omega.parameters()),grad_clip)
                m_opt.step()
            m_run += float(m_loss.item())

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
        if epoch % 50 == 0:
            save_checkpoint(f"{save_path}/epochs/mog_epoch{epoch}_beta{beta}_gamma{gamma}.pth", q_phi, pi_theta, p_psi, p_omega, B, T, Z_DIM, NUM_NEURONS, device)

        # validation
        v_loss, v_policy_loss, v_kl_loss, v_state_decoder_loss = eval_epoch(val_loader, q_phi, pi_theta, p_psi, p_omega, beta, gamma, device)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_checkpoint(f"{save_path}/mog_epoch{epoch}_beta{beta}_gamma{gamma}_best.pth", q_phi, pi_theta, p_psi, p_omega, B, T, Z_DIM, NUM_NEURONS, device)

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

epochs = 50000
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
        save_path = f"checkpoints/{beta}_{gamma}"
        os.makedirs(os.path.join(save_path, "epoch"), exist_ok=True)
        curves = skill_model_training_with_val(save_path, train_loader, val_loader, q_phi, pi_theta, p_psi, p_omega, beta, gamma, epochs=epochs, lr=lr, steps=1)

        wandb.finish()




