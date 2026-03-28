"""
EM-style Dreamer training for temporally-abstracted skill models with a segment
latent model: start encoder, transformer q(s1|o,a), Markov p(s1|s0,z).

E-step: encoders + observation NLL (grad through decoder into latents; decoder weights updated in M-step)
M-step: segment dynamics + skill prior + policy + reward model + observation decoder (NLL on detached latents)
"""

import os
import torch
import torch.nn as nn
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


def _segment_latent_forward(start_enc, s1_post_enc, skill_enc, obs_seq, act_seq):
    """
    Single-segment forward: s0 = f(o0), z ~ q(z|window), s1 ~ q(s1|window),
    with reparameterized samples for losses that need them.
    """
    s0 = start_enc(obs_seq[:, 0, :])
    z_mean, z_std = skill_enc(obs_seq[:, :-1, :], act_seq)
    z = reparameterize(z_mean, z_std)
    s1_mean, s1_std = s1_post_enc(obs_seq, act_seq)
    s1 = reparameterize(s1_mean, s1_std)
    return {
        "s0": s0,
        "z_mean": z_mean,
        "z_std": z_std,
        "z": z,
        "s1_post": (s1_mean, s1_std),
        "s1": s1,
    }


def _state_kl(post, prior):
    """KL divergence averaged over batch, latent dims, and samples (scalar)."""
    kl = kl_divergence(Normal(*post), Normal(*prior))  # [B, s_dim]
    return kl.mean()


def _dreamer_kl_clip(kl_scalar, free_nats: float):
    """Dreamer: L += max(free_nats, KL); gradient vanishes for KL below the floor."""
    if free_nats <= 0:
        return kl_scalar
    return torch.clamp(kl_scalar, min=free_nats)

def _segment_obs_recon_nll(obs_decoder, s0, s1, o0, oH):
    """Mean NLL for diagonal Gaussian decoders on o_0 and o_H (sum over obs dims)."""
    mu0, std0 = obs_decoder.forward_o0(s0)
    nll0 = -Normal(mu0, std0).log_prob(o0).sum(dim=-1).mean()
    muH, stdH = obs_decoder.forward_oH(s1)
    nllH = -Normal(muH, stdH).log_prob(oH).sum(dim=-1).mean()
    return nll0 + nllH



# ---------------------------------------------------------------------------
# E-step loss (encoders)
# ---------------------------------------------------------------------------

def get_E_loss(
    batch,
    start_enc,
    s1_post_enc,
    segment_dynamics,
    skill_enc,
    pi_theta,
    skill_prior,
    reward_model,
    obs_decoder,
    beta,
    alpha_s,
    reward_weight,
    recon_weight,
    kl_balance=False,
    kl_balance_alpha=0.8,
    free_nats: float = 1.0,
):
    obs_seq = batch["obs_seq"]
    act_seq = batch["act_seq"]
    goal_xy = batch["goal_xy"]
    min_goal_dist = batch["min_goal_dist"]

    info = _segment_latent_forward(start_enc, s1_post_enc, skill_enc, obs_seq, act_seq)

    a_loss = _action_nll(pi_theta, obs_seq, info["z"], act_seq)

    # skill KL — posterior side (prior detached)
    with torch.no_grad():
        z_pr_mean, z_pr_std = skill_prior(info["s0"])
    skill_kl_raw = kl_divergence(
        Normal(info["z_mean"], info["z_std"]),
        Normal(z_pr_mean, z_pr_std),
    ).mean()
    skill_kl = _dreamer_kl_clip(skill_kl_raw, free_nats)

    # state KL on s1 only — posterior side (segment dynamics detached)
    with torch.no_grad():
        z_det = info["z"].detach()
        mu_p, sig_p = segment_dynamics(info["s0"].detach(), z_det)
        s1_pr = (mu_p.detach(), sig_p.detach())
    kl_s1 = _state_kl(info["s1_post"], s1_pr)
    state_kl = _dreamer_kl_clip(kl_s1, free_nats)

    # r_mean, r_std = reward_model(info["s0"], info["z"].detach(), goal_xy)
    # reward_loss = -Normal(r_mean, r_std).log_prob(min_goal_dist).mean()

    o0 = obs_seq[:, 0, :]
    oH = obs_seq[:, -1, :]
    recon_loss = _segment_obs_recon_nll(
        obs_decoder, info["s0"], info["s1"], o0, oH
    )

    kl_weight = (1 - kl_balance_alpha) if kl_balance else 1.0
    E_loss = (
        a_loss
        + kl_weight * beta * skill_kl
        + kl_weight * alpha_s * state_kl
        # + reward_weight * reward_loss
        + recon_weight * recon_loss
    )
    return E_loss, {
        "E/a_loss": a_loss.item(),
        "E/skill_kl": skill_kl_raw.item(),
        "E/state_kl": kl_s1.item(),
        # "E/reward_loss": reward_loss.item(),
        "E/recon_loss": recon_loss.item(),
    }


# ---------------------------------------------------------------------------
# M-step loss (dynamics + generative heads)
# ---------------------------------------------------------------------------

def get_M_loss(
    batch,
    start_enc,
    s1_post_enc,
    segment_dynamics,
    skill_enc,
    pi_theta,
    skill_prior,
    reward_model,
    obs_decoder,
    beta,
    alpha_s,
    reward_weight,
    recon_weight,
    kl_balance=False,
    kl_balance_alpha=0.8,
    free_nats: float = 1.0,
):
    obs_seq = batch["obs_seq"]
    act_seq = batch["act_seq"]
    goal_xy = batch["goal_xy"]
    min_goal_dist = batch["min_goal_dist"]

    with torch.no_grad():
        info = _segment_latent_forward(start_enc, s1_post_enc, skill_enc, obs_seq, act_seq)

    s1_post_det = (info["s1_post"][0].detach(), info["s1_post"][1].detach())
    mu_p, sig_p = segment_dynamics(info["s0"].detach(), info["z"].detach())
    kl_s1 = _state_kl(s1_post_det, (mu_p, sig_p))
    state_kl = _dreamer_kl_clip(kl_s1, free_nats)

    a_loss = _action_nll(pi_theta, obs_seq, info["z"], act_seq)

    z_pr_mean, z_pr_std = skill_prior(info["s0"])
    skill_kl_raw = kl_divergence(
        Normal(info["z_mean"], info["z_std"]),
        Normal(z_pr_mean, z_pr_std),
    ).mean()
    skill_kl = _dreamer_kl_clip(skill_kl_raw, free_nats)

    r_mean, r_std = reward_model(info["s0"], info["z"], goal_xy)
    reward_loss = -Normal(r_mean, r_std).log_prob(min_goal_dist).mean()

    o0 = obs_seq[:, 0, :]
    oH = obs_seq[:, -1, :]
    s0_det = info["s0"].detach()
    s1_det = info["s1_post"][0].detach()
    recon_loss = _segment_obs_recon_nll(obs_decoder, s0_det, s1_det, o0, oH)

    kl_weight = kl_balance_alpha if kl_balance else 1.0
    M_loss = (
        a_loss
        + kl_weight * beta * skill_kl
        + kl_weight * alpha_s * state_kl
        + reward_weight * reward_loss
        + recon_weight * recon_loss
    )
    return M_loss, {
        "M/a_loss": a_loss.item(),
        "M/skill_kl": skill_kl_raw.item(),
        "M/state_kl": kl_s1.item(),
        "M/reward_loss": reward_loss.item(),
        "M/recon_loss": recon_loss.item(),
    }


# ---------------------------------------------------------------------------
# Evaluation (full KL, no EM detaching)
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_eval_losses(
    batch,
    start_enc,
    s1_post_enc,
    segment_dynamics,
    skill_enc,
    pi_theta,
    skill_prior,
    reward_model,
    obs_decoder,
    beta,
    alpha_s,
    reward_weight,
    recon_weight,
):
    obs_seq = batch["obs_seq"]
    act_seq = batch["act_seq"]
    goal_xy = batch["goal_xy"]
    min_goal_dist = batch["min_goal_dist"]

    info = _segment_latent_forward(start_enc, s1_post_enc, skill_enc, obs_seq, act_seq)

    a_loss = _action_nll(pi_theta, obs_seq, info["z"], act_seq)

    z_pr_mean, z_pr_std = skill_prior(info["s0"])
    skill_kl = kl_divergence(
        Normal(info["z_mean"], info["z_std"]),
        Normal(z_pr_mean, z_pr_std),
    ).mean()

    mu_p, sig_p = segment_dynamics(info["s0"], info["z"])
    state_kl = _state_kl(info["s1_post"], (mu_p, sig_p))

    r_mean, r_std = reward_model(info["s0"], info["z"], goal_xy)
    reward_loss = -Normal(r_mean, r_std).log_prob(min_goal_dist).mean()

    o0 = obs_seq[:, 0, :]
    oH = obs_seq[:, -1, :]
    recon_loss = _segment_obs_recon_nll(
        obs_decoder, info["s0"], info["s1"], o0, oH
    )

    total = (
        a_loss
        + beta * skill_kl
        + alpha_s * state_kl
        + reward_weight * reward_loss
        + recon_weight * recon_loss
    )
    return total, a_loss, skill_kl, state_kl, reward_loss, recon_loss


@torch.no_grad()
def eval_epoch(
    val_loader,
    start_enc,
    s1_post_enc,
    segment_dynamics,
    skill_enc,
    pi_theta,
    skill_prior,
    reward_model,
    obs_decoder,
    beta,
    alpha_s,
    reward_weight,
    recon_weight,
    device,
):
    for m in (
        start_enc,
        s1_post_enc,
        segment_dynamics,
        skill_enc,
        pi_theta,
        skill_prior,
        reward_model,
        obs_decoder,
    ):
        m.eval()

    sums = dict(total=0.0, a=0.0, skill_kl=0.0, state_kl=0.0, rew=0.0, recon=0.0)
    n = 0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        total, a_loss, skill_kl, state_kl, rew_loss, recon_loss = get_eval_losses(
            batch,
            start_enc,
            s1_post_enc,
            segment_dynamics,
            skill_enc,
            pi_theta,
            skill_prior,
            reward_model,
            obs_decoder,
            beta,
            alpha_s,
            reward_weight,
            recon_weight,
        )
        sums["total"] += total.item()
        sums["a"] += a_loss.item()
        sums["skill_kl"] += skill_kl.item()
        sums["state_kl"] += state_kl.item()
        sums["rew"] += rew_loss.item()
        sums["recon"] += recon_loss.item()
        n += 1

    if n == 0:
        return {k: None for k in sums}
    return {k: v / n for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_dreamer_checkpoint(
    path,
    start_enc,
    s1_post_enc,
    segment_dynamics,
    skill_enc,
    pi_theta,
    skill_prior,
    reward_model,
    obs_decoder,
    model_cfg,
    train_cfg,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "start_enc": start_enc.state_dict(),
            "s1_post_enc": s1_post_enc.state_dict(),
            "segment_dynamics": segment_dynamics.state_dict(),
            "skill_enc": skill_enc.state_dict(),
            "pi_theta": pi_theta.state_dict(),
            "skill_prior": skill_prior.state_dict(),
            "reward_model": reward_model.state_dict(),
            "obs_decoder": obs_decoder.state_dict(),
            "model_cfg": model_cfg.__dict__,
            "train_cfg": train_cfg.__dict__,
        },
        path,
    )
    print(f"checkpoint saved -> {path}")


def load_dreamer_checkpoint(
    path,
    start_enc,
    s1_post_enc,
    segment_dynamics,
    skill_enc,
    pi_theta,
    skill_prior,
    reward_model,
    obs_decoder,
    strict=True,
):
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    start_enc.load_state_dict(ckpt["start_enc"], strict=strict)
    s1_post_enc.load_state_dict(ckpt["s1_post_enc"], strict=strict)
    segment_dynamics.load_state_dict(ckpt["segment_dynamics"], strict=strict)
    skill_enc.load_state_dict(ckpt["skill_enc"], strict=strict)
    pi_theta.load_state_dict(ckpt["pi_theta"], strict=strict)
    skill_prior.load_state_dict(ckpt["skill_prior"], strict=strict)
    reward_model.load_state_dict(ckpt["reward_model"], strict=strict)
    if "obs_decoder" in ckpt:
        obs_decoder.load_state_dict(ckpt["obs_decoder"], strict=strict)
    else:
        print("[checkpoint] no obs_decoder in checkpoint; leaving random init")
    print(f"[checkpoint] loaded <- {path}")
    return ckpt


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def dreamer_training_with_val(
    save_path: str,
    train_loader,
    val_loader,
    start_enc,
    s1_post_enc,
    segment_dynamics,
    skill_enc,
    pi_theta,
    skill_prior,
    reward_model,
    obs_decoder,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: str = "cpu",
):
    all_models = [
        start_enc,
        s1_post_enc,
        segment_dynamics,
        skill_enc,
        pi_theta,
        skill_prior,
        reward_model,
        obs_decoder,
    ]
    for m in all_models:
        m.to(device)

    E_params = (
        list(start_enc.parameters())
        + list(s1_post_enc.parameters())
        + list(skill_enc.parameters())
    )
    E_optimizer = torch.optim.Adam(E_params, lr=train_cfg.lr)

    M_params = (
        list(segment_dynamics.parameters())
        + list(pi_theta.parameters())
        + list(skill_prior.parameters())
        + list(reward_model.parameters())
        + list(obs_decoder.parameters())
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

            for _ in range(train_cfg.e_steps):
                E_optimizer.zero_grad(set_to_none=True)
                e_loss, e_info = get_E_loss(
                    batch,
                    start_enc,
                    s1_post_enc,
                    segment_dynamics,
                    skill_enc,
                    pi_theta,
                    skill_prior,
                    reward_model,
                    obs_decoder,
                    train_cfg.beta,
                    train_cfg.alpha_s,
                    train_cfg.reward_weight,
                    train_cfg.recon_weight,
                    train_cfg.kl_balance,
                    train_cfg.kl_balance_alpha,
                    train_cfg.free_nats,
                )
                e_loss.backward()
                if train_cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(E_params, train_cfg.grad_clip)
                E_optimizer.step()
            e_run += e_loss.item()
            for k, v in e_info.items():
                e_info_sums[k] = e_info_sums.get(k, 0.0) + v

            for _ in range(train_cfg.m_steps):
                M_optimizer.zero_grad(set_to_none=True)
                m_loss, m_info = get_M_loss(
                    batch,
                    start_enc,
                    s1_post_enc,
                    segment_dynamics,
                    skill_enc,
                    pi_theta,
                    skill_prior,
                    reward_model,
                    obs_decoder,
                    train_cfg.beta,
                    train_cfg.alpha_s,
                    train_cfg.reward_weight,
                    train_cfg.recon_weight,
                    train_cfg.kl_balance,
                    train_cfg.kl_balance_alpha,
                    train_cfg.free_nats,
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

        if epoch % train_cfg.checkpoint_every == 0:
            save_dreamer_checkpoint(
                f"{save_path}/epochs/dreamer_epoch{epoch}.pth",
                start_enc,
                s1_post_enc,
                segment_dynamics,
                skill_enc,
                pi_theta,
                skill_prior,
                reward_model,
                obs_decoder,
                model_cfg,
                train_cfg,
            )

        val_metrics = eval_epoch(
            val_loader,
            start_enc,
            s1_post_enc,
            segment_dynamics,
            skill_enc,
            pi_theta,
            skill_prior,
            reward_model,
            obs_decoder,
            train_cfg.beta,
            train_cfg.alpha_s,
            train_cfg.reward_weight,
            train_cfg.recon_weight,
            device,
        )
        v_loss = val_metrics["total"]
        if v_loss is not None and v_loss < best_val_loss:
            best_val_loss = v_loss
            save_dreamer_checkpoint(
                f"{save_path}/dreamer_best.pth",
                start_enc,
                s1_post_enc,
                segment_dynamics,
                skill_enc,
                pi_theta,
                skill_prior,
                reward_model,
                obs_decoder,
                model_cfg,
                train_cfg,
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
            "val/recon_loss": val_metrics["recon"],
            "epoch": epoch,
        }
        for k, v in e_info_avg.items():
            log_dict[f"train/{k}"] = v
        for k, v in m_info_avg.items():
            log_dict[f"train/{k}"] = v
        wandb.log(log_dict, step=epoch)

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
