"""
EM-style Dreamer training for temporally-abstracted skill models
with an RSSM operating at the abstract temporal level.

E-step: update RSSM encoder (posterior/trajectory/obs encoder) + skill encoder
M-step: update RSSM dynamics (GRUCell/prior heads) + skill prior + policy + reward model
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import matplotlib.pyplot as plt
import wandb

from model.online.training.config import ModelConfig, TrainConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reparameterize(mean, std):
    return mean + std * torch.randn_like(mean)


def _action_nll(pi_theta, obs_seq, z, act_seq):
    """
    Average negative log-likelihood of actions under the low-level policy.

    Args:
        pi_theta:  SkillPolicy  (o_t, z) -> (mean, std) over a_t
        obs_seq:   [B, H+1, obs_dim]  (first H observations used)
        z:         [B, z_dim]
        act_seq:   [B, H, action_dim]
    """
    B, Hp1, _ = obs_seq.shape
    H = Hp1 - 1

    z_bt = z.unsqueeze(1).expand(B, H, -1)
    mu_pi, std_pi = pi_theta(
        obs_seq[:, :H, :].reshape(B * H, -1),
        z_bt.reshape(B * H, -1),
    )
    mu_pi = mu_pi.view(B, H, -1)
    std_pi = std_pi.view(B, H, -1)
    a_dist = Normal(mu_pi, std_pi)
    return -torch.sum(a_dist.log_prob(act_seq)) / (B * H)


def _rssm_two_step(rssm, skill_enc, obs_seq, act_seq):
    """
    Run the two-step RSSM forward pass on a single independent segment.

    Step 0: h_0 = GRUCell(0, [0,0])  →  prior(s_0|h_0), posterior(s_0|h_0,o_0)
    Step 1: h_1 = GRUCell(h_0,[s_0,z])  →  prior(s_1|h_1), posterior(s_1|h_1,traj)

    Returns dict with all quantities needed for loss computation.
    """
    B = obs_seq.size(0)
    device = obs_seq.device

    h_init, s_init, z_init = rssm.initial_state(B, device)

    # --- step 0: initial abstract state ---
    obs_context = rssm.encoder.encode_obs(obs_seq[:, 0, :])
    h0, s0_prior, s0_post = rssm.observe_step(h_init, s_init, z_init, obs_context)
    s0 = reparameterize(*s0_post)

    # --- skill inference ---
    z_mean, z_std = skill_enc(obs_seq[:, :-1, :], act_seq)
    z = reparameterize(z_mean, z_std)

    # --- step 1: next abstract state after executing skill ---
    traj_context = rssm.encoder.encode_trajectory(obs_seq, act_seq)
    h1, s1_prior, s1_post = rssm.observe_step(h0, s0, z, traj_context)

    return {
        "s0": s0, "z_mean": z_mean, "z_std": z_std, "z": z,
        "s0_prior": s0_prior, "s0_post": s0_post,
        "s1_prior": s1_prior, "s1_post": s1_post,
        "h0": h0, "h1": h1,
    }


def _state_kl(post, prior):
    """KL divergence averaged over batch and s_dim (DreamerV2 convention)."""
    # return kl_divergence(Normal(*post), Normal(*prior)).mean()
    kl = kl_divergence(Normal(*post), Normal(*prior)) # [B, s_dim]
    kl = kl.mean(dim=-1)   # B Kl per sample
    kl = torch.clamp(kl, min=1.0) # free nats
    return kl.mean() # avg KL for batch

# ---------------------------------------------------------------------------
# E-step loss (update RSSM encoder + skill encoder)
# ---------------------------------------------------------------------------

def get_E_loss(
    batch,
    rssm, skill_enc, pi_theta, skill_prior, reward_model,
    beta, alpha_s, reward_weight,
    kl_balance=False, kl_balance_alpha=0.8,
):
    obs_seq = batch["obs_seq"]
    act_seq = batch["act_seq"]
    goal_xy = batch["goal_xy"]
    min_goal_dist = batch["min_goal_dist"]

    info = _rssm_two_step(rssm, skill_enc, obs_seq, act_seq)

    a_loss = _action_nll(pi_theta, obs_seq, info["z"], act_seq)

    # skill KL — posterior side (prior detached)
    with torch.no_grad():
        z_pr_mean, z_pr_std = skill_prior(info["s0"])
    skill_kl = kl_divergence(
        Normal(info["z_mean"], info["z_std"]),
        Normal(z_pr_mean, z_pr_std),
    ).mean()

    # state KL — posterior side (RSSM dynamics/prior detached)
    with torch.no_grad():
        s0_pr = (info["s0_prior"][0].detach(), info["s0_prior"][1].detach())
        s1_pr = (info["s1_prior"][0].detach(), info["s1_prior"][1].detach())
    state_kl = (_state_kl(info["s0_post"], s0_pr) + _state_kl(info["s1_post"], s1_pr)) / 2

    # Goal-conditioned min-distance loss — encoders frozen w.r.t. reward head params
    r_mean, r_std = reward_model(info["s0"], info["z"], goal_xy)
    reward_loss = -Normal(r_mean, r_std).log_prob(min_goal_dist).mean()

    kl_weight = (1 - kl_balance_alpha) if kl_balance else 1.0
    E_loss = (
        a_loss
        + kl_weight * beta * skill_kl
        + kl_weight * alpha_s * state_kl
        + reward_weight * reward_loss
    )
    return E_loss, {
        "E/a_loss": a_loss.item(),
        "E/skill_kl": skill_kl.item(),
        "E/state_kl": state_kl.item(),
        "E/reward_loss": reward_loss.item(),
    }


# ---------------------------------------------------------------------------
# M-step loss (update RSSM dynamics + skill prior + policy + reward model)
# ---------------------------------------------------------------------------

def get_M_loss(
    batch,
    rssm, skill_enc, pi_theta, skill_prior, reward_model,
    beta, alpha_s, reward_weight,
    kl_balance=False, kl_balance_alpha=0.8,
):
    obs_seq = batch["obs_seq"]
    act_seq = batch["act_seq"]
    goal_xy = batch["goal_xy"]
    min_goal_dist = batch["min_goal_dist"]

    with torch.no_grad():
        info = _rssm_two_step(rssm, skill_enc, obs_seq, act_seq)

    # Re-run dynamics with grad so GRUCell + prior heads get gradients,
    # while encoder outputs (s0, z, posteriors) stay detached.
    h_init, s_init, z_init = rssm.initial_state(obs_seq.size(0), obs_seq.device)

    h0_dyn, s0_pr_mean, s0_pr_std = rssm.dynamics(h_init, s_init, z_init)
    s0_prior_g = (s0_pr_mean, s0_pr_std)

    h1_dyn, s1_pr_mean, s1_pr_std = rssm.dynamics(h0_dyn, info["s0"], info["z"])
    s1_prior_g = (s1_pr_mean, s1_pr_std)

    a_loss = _action_nll(pi_theta, obs_seq, info["z"], act_seq)

    # skill KL — prior side (encoder detached)
    z_pr_mean, z_pr_std = skill_prior(info["s0"])
    skill_kl = kl_divergence(
        Normal(info["z_mean"], info["z_std"]),
        Normal(z_pr_mean, z_pr_std),
    ).mean()

    # state KL — prior side (encoder posteriors detached)
    s0_post_det = (info["s0_post"][0].detach(), info["s0_post"][1].detach())
    s1_post_det = (info["s1_post"][0].detach(), info["s1_post"][1].detach())
    state_kl = (_state_kl(s0_post_det, s0_prior_g) + _state_kl(s1_post_det, s1_prior_g)) / 2

    r_mean, r_std = reward_model(info["s0"], info["z"], goal_xy)
    reward_loss = -Normal(r_mean, r_std).log_prob(min_goal_dist).mean()

    kl_weight = kl_balance_alpha if kl_balance else 1.0
    M_loss = (
        a_loss
        + kl_weight * beta * skill_kl
        + kl_weight * alpha_s * state_kl
        + reward_weight * reward_loss
    )
    return M_loss, {
        "M/a_loss": a_loss.item(),
        "M/skill_kl": skill_kl.item(),
        "M/state_kl": state_kl.item(),
        "M/reward_loss": reward_loss.item(),
    }


# ---------------------------------------------------------------------------
# Evaluation (full KL, no EM detaching)
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_eval_losses(
    batch,
    rssm, skill_enc, pi_theta, skill_prior, reward_model,
    beta, alpha_s, reward_weight,
):
    obs_seq = batch["obs_seq"]
    act_seq = batch["act_seq"]
    goal_xy = batch["goal_xy"]
    min_goal_dist = batch["min_goal_dist"]

    info = _rssm_two_step(rssm, skill_enc, obs_seq, act_seq)

    a_loss = _action_nll(pi_theta, obs_seq, info["z"], act_seq)

    z_pr_mean, z_pr_std = skill_prior(info["s0"])
    skill_kl = kl_divergence(
        Normal(info["z_mean"], info["z_std"]),
        Normal(z_pr_mean, z_pr_std),
    ).mean()

    state_kl = (_state_kl(info["s0_post"], info["s0_prior"]) +
                _state_kl(info["s1_post"], info["s1_prior"])) / 2

    r_mean, r_std = reward_model(info["s0"], info["z"], goal_xy)
    reward_loss = -Normal(r_mean, r_std).log_prob(min_goal_dist).mean()

    total = a_loss + beta * skill_kl + alpha_s * state_kl + reward_weight * reward_loss
    return total, a_loss, skill_kl, state_kl, reward_loss


@torch.no_grad()
def eval_epoch(
    val_loader,
    rssm, skill_enc, pi_theta, skill_prior, reward_model,
    beta, alpha_s, reward_weight, device,
):
    for m in (rssm, skill_enc, pi_theta, skill_prior, reward_model):
        m.eval()

    sums = dict(total=0.0, a=0.0, skill_kl=0.0, state_kl=0.0, rew=0.0)
    n = 0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        total, a_loss, skill_kl, state_kl, rew_loss = get_eval_losses(
            batch, rssm, skill_enc, pi_theta,
            skill_prior, reward_model, beta, alpha_s, reward_weight,
        )
        sums["total"] += total.item()
        sums["a"] += a_loss.item()
        sums["skill_kl"] += skill_kl.item()
        sums["state_kl"] += state_kl.item()
        sums["rew"] += rew_loss.item()
        n += 1

    if n == 0:
        return {k: None for k in sums}
    return {k: v / n for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_dreamer_checkpoint(
    path,
    rssm, skill_enc, pi_theta, skill_prior, reward_model,
    model_cfg, train_cfg,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "rssm": rssm.state_dict(),
            "skill_enc": skill_enc.state_dict(),
            "pi_theta": pi_theta.state_dict(),
            "skill_prior": skill_prior.state_dict(),
            "reward_model": reward_model.state_dict(),
            "model_cfg": model_cfg.__dict__,
            "train_cfg": train_cfg.__dict__,
        },
        path,
    )
    print(f"checkpoint saved -> {path}")


def load_dreamer_checkpoint(
    path,
    rssm, skill_enc, pi_theta, skill_prior, reward_model,
    strict=True,
):
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    rssm.load_state_dict(ckpt["rssm"], strict=strict)
    skill_enc.load_state_dict(ckpt["skill_enc"], strict=strict)
    pi_theta.load_state_dict(ckpt["pi_theta"], strict=strict)
    skill_prior.load_state_dict(ckpt["skill_prior"], strict=strict)
    reward_model.load_state_dict(ckpt["reward_model"], strict=strict)
    print(f"[checkpoint] loaded <- {path}")
    return ckpt


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def dreamer_training_with_val(
    save_path: str,
    train_loader,
    val_loader,
    # models
    rssm,
    skill_enc,
    pi_theta,
    skill_prior,
    reward_model,
    # configs
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: str = "cpu",
):
    all_models = [rssm, skill_enc, pi_theta, skill_prior, reward_model]
    for m in all_models:
        m.to(device)

    # E-optimizer: RSSM encoder (posterior + obs/traj encoder) + skill encoder
    E_params = (
        list(rssm.encoder_parameters())
        + list(skill_enc.parameters())
    )
    E_optimizer = torch.optim.Adam(E_params, lr=train_cfg.lr)

    # M-optimizer: RSSM dynamics (GRUCell + prior heads) + generative models
    M_params = (
        list(rssm.dynamics_parameters())
        + list(pi_theta.parameters())
        + list(skill_prior.parameters())
        + list(reward_model.parameters())
    )
    M_optimizer = torch.optim.Adam(M_params, lr=train_cfg.lr)

    tr_e, tr_m, va = [], [], []
    best_val_loss = float("inf")

    for epoch in range(1, train_cfg.epochs + 1):
        for m in all_models:
            m.train()
        e_run = m_run = 0.0
        e_info_sums = {}
        m_info_sums = {}
        nb = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            nb += 1

            # ---------- E-step ----------
            for _ in range(train_cfg.e_steps):
                E_optimizer.zero_grad(set_to_none=True)
                e_loss, e_info = get_E_loss(
                    batch,
                    rssm, skill_enc, pi_theta, skill_prior, reward_model,
                    train_cfg.beta, train_cfg.alpha_s,
                    train_cfg.reward_weight,
                    train_cfg.kl_balance, train_cfg.kl_balance_alpha,
                )
                e_loss.backward()
                if train_cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(E_params, train_cfg.grad_clip)
                E_optimizer.step()
            e_run += e_loss.item()
            for k, v in e_info.items():
                e_info_sums[k] = e_info_sums.get(k, 0.0) + v

            # ---------- M-step ----------
            for _ in range(train_cfg.m_steps):
                M_optimizer.zero_grad(set_to_none=True)
                m_loss, m_info = get_M_loss(
                    batch,
                    rssm, skill_enc, pi_theta, skill_prior, reward_model,
                    train_cfg.beta, train_cfg.alpha_s,
                    train_cfg.reward_weight,
                    train_cfg.kl_balance, train_cfg.kl_balance_alpha,
                )
                m_loss.backward()
                if train_cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(M_params, train_cfg.grad_clip)
                M_optimizer.step()
            m_run += m_loss.item()
            for k, v in m_info.items():
                m_info_sums[k] = m_info_sums.get(k, 0.0) + v

        e_epoch = e_run / max(1, nb)
        m_epoch = m_run / max(1, nb)
        e_info_avg = {k: v / max(1, nb) for k, v in e_info_sums.items()}
        m_info_avg = {k: v / max(1, nb) for k, v in m_info_sums.items()}
        tr_e.append(e_epoch)
        tr_m.append(m_epoch)

        # periodic checkpoint
        if epoch % train_cfg.checkpoint_every == 0:
            save_dreamer_checkpoint(
                f"{save_path}/epochs/dreamer_epoch{epoch}.pth",
                rssm, skill_enc, pi_theta, skill_prior, reward_model,
                model_cfg, train_cfg,
            )

        # validation
        val_metrics = eval_epoch(
            val_loader,
            rssm, skill_enc, pi_theta, skill_prior, reward_model,
            train_cfg.beta, train_cfg.alpha_s, train_cfg.reward_weight, device,
        )
        v_loss = val_metrics["total"]
        if v_loss is not None and v_loss < best_val_loss:
            best_val_loss = v_loss
            save_dreamer_checkpoint(
                f"{save_path}/dreamer_best.pth",
                rssm, skill_enc, pi_theta, skill_prior, reward_model,
                model_cfg, train_cfg,
            )
        va.append(v_loss)

        print(
            f"[Epoch {epoch:03d}/{train_cfg.epochs}] "
            f"train E:{e_epoch:.4f}  M:{m_epoch:.4f} "
            f"| val loss:{v_loss:.4f}"
        )

        log_dict = {
            "train/E_loss": e_epoch,
            "train/M_loss": m_epoch,
            "val/total": val_metrics["total"],
            "val/a_loss": val_metrics["a"],
            "val/skill_kl": val_metrics["skill_kl"],
            "val/state_kl": val_metrics["state_kl"],
            "val/reward_loss": val_metrics["rew"],
            "epoch": epoch,
        }
        for k, v in e_info_avg.items():
            log_dict[f"train/{k}"] = v
        for k, v in m_info_avg.items():
            log_dict[f"train/{k}"] = v
        wandb.log(log_dict, step=epoch)

    # ---- final plot ----
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(tr_e, label="Train E loss")
    plt.plot(tr_m, label="Train M loss")
    if all(v is not None for v in va):
        plt.plot(va, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Dreamer EM training: train vs. val losses")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    fig = plt.gcf()
    wandb.log({"plots/loss_curves": wandb.Image(fig)}, step=epoch)
    plt.close(fig)

    return {"train_E": tr_e, "train_M": tr_m, "val": va}
