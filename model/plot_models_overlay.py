"""
compare_checkpoints_plots.py

Makes EACH figure contain ONE SUBPLOT PER CHECKPOINT.

It produces (for each sampled dataset state):
  1) global_map_grid      : dataset map + rollouts, one tile per checkpoint
  2) traj_grid            : detailed XY rollouts, one tile per checkpoint
  3) endpoints_vs_tawm_grid: endpoints + TAWM ellipses, one tile per checkpoint

Optionally (posterior demo mode):
  4) posterior_grid_train : overlay dataset subtrajectory + rollouts + TAWM per checkpoint
  5) posterior_grid_test  : same for test

Notes:
- This script avoids redefining functions twice.
- It caches loaded models per checkpoint for speed.
- It keeps S_stats (mean/std) per checkpoint and uses it ONLY for TAWM (p_psi) like your earlier code.
"""

# -------------------------
# Imports
# -------------------------
import os
import re
import time
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import gymnasium as gym
import mujoco
import minari
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform

# Your modules
from skill_model import SkillPolicy, SkillPosterior, SkillPrior, TAWM
from utils import pack_state_from_obs, read_antmaze_obs


# -------------------------
# Device + constants
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

state_dim = 29
action_dim = 8

# Subtrajectory length (for posterior demo + for rollout horizon default)
T = 40

# Checkpoints you want tiles for
checkpoints = [
    "../checkpoints/antmaze_diverse_detached_250_1.pth",
    "../checkpoints/antmaze_diverse_detached_250_0.01.pth",
    "../checkpoints/antmaze_diverse_detached_250_0.001.pth",
    "../checkpoints/antmaze_diverse_detached_250_0.0001.pth",
    "../checkpoints/antmaze_diverse_detached_250_10.pth",
    "../checkpoints/antmaze_diverse_detached_250_100.pth",
    "../checkpoints/antmaze_diverse_detached_250_1000.pth",
    "../checkpoints/antmaze_diverse_detached_250_10000.pth",
]


# -------------------------
# Output dirs + saving
# -------------------------
def _safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))

DIRNAME = "beta_values_test"
PLOT_DIR = os.path.join("plots", DIRNAME)
os.makedirs(PLOT_DIR, exist_ok=True)

def save_fig(fig, stem: str, meta: dict | None = None, out_dir=PLOT_DIR, close=True):
    os.makedirs(out_dir, exist_ok=True)
    meta = meta or {}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    parts = [stem, stamp] + [f"{k}-{meta[k]}" for k in sorted(meta.keys())]
    fname = "__".join(_safe(p) for p in parts) + ".png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    if close:
        plt.close(fig)
    print(f"[saved] {path}")
    return path


# -------------------------
# Dataset + env
# -------------------------
ant_maze_dataset = minari.load_dataset("D4RL/antmaze/medium-diverse-v1")

def recover_ant_env():
    return ant_maze_dataset.recover_environment()


# -------------------------
# Helpers: torch + dists
# -------------------------
def to_torch(x):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def policy_dist(mu, std):
    # tanh-squashed normal action dist
    base = Independent(Normal(mu, std.clamp_min(0.05)), 1)
    return TransformedDistribution(base, [TanhTransform(cache_size=1)])


# -------------------------
# MuJoCo state reset helpers
# -------------------------
def get_sim_handles(env):
    """Unwrap to reach model/data."""
    t = env
    for attr in ["env", "unwrapped"]:
        if hasattr(t, attr):
            t = getattr(t, attr)
    if hasattr(t, "model") and hasattr(t, "data"):
        return t, t.model, t.data
    raise RuntimeError("Could not access MuJoCo model/data from env. Check wrappers.")

def split_obs_to_qpos_qvel(s0_obs, s0_ag, env):
    """Convert dataset obs+achieved_goal into mujoco qpos/qvel."""
    _, model, data = get_sim_handles(env)
    nq, nv = int(model.nq), int(model.nv)

    s0_obs = np.asarray(s0_obs, np.float32).ravel()
    s0_ag  = np.asarray(s0_ag,  np.float32).ravel()

    qpos = data.qpos.ravel().copy()
    qvel = data.qvel.ravel().copy()

    qpos[0:2] = s0_ag
    qpos[2:nq] = s0_obs[: (nq - 2)]
    qvel[:nv] = s0_obs[(nq - 2) : (nq - 2 + nv)]

    return qpos.astype(np.float32), qvel.astype(np.float32)

def set_env_state(env, qpos, qvel):
    """Write qpos/qvel and forward."""
    _, model, data = get_sim_handles(env)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)


# -------------------------
# Random state sampling from dataset
# -------------------------
def collect_all_states(minari_ds):
    """
    Collect every (episode_idx, t, s_obs, s_ag) pair from dataset.
    """
    all_states = []
    for ep_idx, ep in enumerate(minari_ds.iterate_episodes()):
        obs = ep.observations["observation"]
        ag  = ep.observations["achieved_goal"]
        L = len(obs)
        for t in range(L):
            all_states.append((ep_idx, t, obs[t].astype(np.float32), ag[t].astype(np.float32)))
    return all_states

def sample_random_states_from_dataset(minari_ds, num_states=10, seed=0):
    all_states = collect_all_states(minari_ds)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(all_states), size=min(num_states, len(all_states)), replace=False)
    return [all_states[i] for i in idxs]

def collect_all_xy(minari_ds):
    all_xy = []
    for ep in minari_ds.iterate_episodes():
        xy = ep.observations["achieved_goal"][:, :2]
        all_xy.append(xy)
    return np.concatenate(all_xy, axis=0)


# -------------------------
# Checkpoint cache (models + per-ckpt stats)
# -------------------------
MODEL_CACHE = {}  # ckpt_path -> dict(models..., stats...)

def parse_beta(ckpt_path: str) -> str:
    base = os.path.basename(ckpt_path)
    m = re.search(r"_([0-9.]+)\.pth$", base)
    return m.group(1) if m else base

def get_models_for_checkpoint(ckpt_path: str, strict=True):
    if ckpt_path in MODEL_CACHE:
        return MODEL_CACHE[ckpt_path]

    q_phi    = SkillPosterior(state_dim=state_dim, action_dim=action_dim).to(device)
    pi_theta = SkillPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
    p_psi    = TAWM(state_dim=state_dim).to(device)
    p_omega  = SkillPrior(state_dim=state_dim).to(device)

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device("cpu"))
    q_phi.load_state_dict(ckpt["q_phi"], strict=strict)
    pi_theta.load_state_dict(ckpt["pi_theta"], strict=strict)
    p_psi.load_state_dict(ckpt["p_psi"], strict=strict)
    p_omega.load_state_dict(ckpt["p_omega"], strict=strict)

    stats = ckpt.get("S_stats", None)
    if stats is None:
        raise ValueError(f"{ckpt_path} missing S_stats (mean/std).")

    stats_dict = {"mean": stats["mean"], "std": stats["std"]}

    MODEL_CACHE[ckpt_path] = {
        "q_phi": q_phi,
        "pi_theta": pi_theta,
        "p_psi": p_psi,
        "p_omega": p_omega,
        "stats": stats_dict,
    }
    print(f"[cache] loaded <- {ckpt_path}")
    return MODEL_CACHE[ckpt_path]


# -------------------------
# TAWM helpers (per checkpoint stats)
# -------------------------
def standardize_state_np(s, mean, std):
    mean = np.asarray(mean, np.float32)
    std  = np.asarray(std,  np.float32)
    return (s - mean) / std

def unstandardize_mu_std_np(mu_std, std_std, mean, std):
    """
    If y_std = (y - mean)/std, and model outputs (mu_std, std_std),
    then:
      mu = mu_std * std + mean
      std = std_std * std
    """
    mean = np.asarray(mean, np.float32)
    std  = np.asarray(std,  np.float32)
    mu = mu_std * std + mean
    sd = std_std * std
    return mu, sd

def draw_cov_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellip = Ellipse(xy=mean, width=width, height=height, angle=theta, fill=False, **kwargs)
    ax.add_patch(ellip)


@torch.no_grad()
def tawm_xy_gaussian_for_ckpt(s0_env, z, models):
    """
    Uses p_psi from this ckpt and its S_stats to output:
      mean_xy (2,), cov_xy (2x2) in UNSTANDARDIZED XY space
    """
    p_psi = models["p_psi"]
    mean = models["stats"]["mean"]
    std  = models["stats"]["std"]

    # standardize s0 for p_psi (like your earlier sampling code)
    s0_std = standardize_state_np(s0_env, mean, std)
    s0_t = to_torch(s0_std).unsqueeze(0)
    z_t  = z.unsqueeze(0).to(device)

    mu_T_std, std_T_std = p_psi(s0_t, z_t)            # in standardized space
    mu_T_std  = mu_T_std.squeeze(0).cpu().numpy()
    std_T_std = std_T_std.squeeze(0).cpu().numpy()

    mu_T, std_T = unstandardize_mu_std_np(mu_T_std, std_T_std, mean, std)  # unstandardized

    mean_xy = mu_T[-2:]
    std_xy  = std_T[-2:]
    cov_xy  = np.diag(std_xy**2)
    return mean_xy, cov_xy


# -------------------------
# Rollouts: ONE checkpoint + ALL checkpoints
# -------------------------
@torch.no_grad()
def rollout_one_checkpoint(
    env,
    s0_obs_ds,
    s0_ag_ds,
    models,
    N_trajs=20,
    horizon=40,
    seed=0,
    resample_skill_per_traj=False,
    use_prior=True,
    z_fixed=None,
):
    """
    Rollouts for ONE checkpoint.
    Returns: (trajs_xy, z_used, s0_env)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    q_phi    = models["q_phi"]
    pi_theta = models["pi_theta"]
    p_omega  = models["p_omega"]

    q_phi.eval(); pi_theta.eval(); p_omega.eval()

    # set mujoco to dataset state
    s0_qpos, s0_qvel = split_obs_to_qpos_qvel(s0_obs_ds, s0_ag_ds, env)

    env.reset()
    set_env_state(env, s0_qpos, s0_qvel)
    obs = read_antmaze_obs(env)
    _, ag0, s0_env = pack_state_from_obs(obs)
    s0_t = to_torch(s0_env).unsqueeze(0)

    # choose z once (shared across trajectories unless resample)
    if z_fixed is not None:
        z0 = z_fixed.to(device).detach().clone().squeeze(0)
    elif use_prior:
        mu_pr, std_pr = p_omega(s0_t)
        z0 = (mu_pr + std_pr * torch.randn_like(mu_pr)).squeeze(0)
    else:
        raise ValueError("Need z_fixed if use_prior=False")

    trajs_xy = []
    for k in range(N_trajs):
        env.reset()
        set_env_state(env, s0_qpos, s0_qvel)

        cur_obs = read_antmaze_obs(env)
        _, ag_start, _ = pack_state_from_obs(cur_obs)

        if resample_skill_per_traj and (z_fixed is None) and use_prior:
            _, _, s0_env_local = pack_state_from_obs(cur_obs)
            s0_t_local = to_torch(s0_env_local).unsqueeze(0)
            mu_pr, std_pr = p_omega(s0_t_local)
            z = (mu_pr + std_pr * torch.randn_like(mu_pr)).squeeze(0)
        else:
            z = z0

        xy = [ag_start.copy()]
        for t in range(horizon):
            _, _, st = pack_state_from_obs(cur_obs)
            st_t = to_torch(st).unsqueeze(0)
            a_mu, a_std = pi_theta(st_t, z.unsqueeze(0))
            a = policy_dist(a_mu, a_std).sample().squeeze(0).cpu().numpy().astype(np.float32)

            cur_obs, _, term, trunc, _ = env.step(a)
            _, ag_t, _ = pack_state_from_obs(cur_obs)
            xy.append(ag_t.copy())
            if term or trunc:
                break

        trajs_xy.append(np.stack(xy, axis=0))

    return trajs_xy, z0.detach().cpu(), s0_env


@torch.no_grad()
def rollout_all_checkpoints(
    env,
    s0_obs_ds,
    s0_ag_ds,
    checkpoints,
    N_trajs=20,
    horizon=40,
    seed=0,
    z_strategy="own_prior",  # "own_prior" or "shared_first"
):
    """
    Returns:
      trajs_groups : list[ list[(Ti,2)] ]  one list-of-trajs per checkpoint
      z_list       : list[z_used_per_checkpoint]
      s0_env_list  : list[s0_env_per_checkpoint]
      titles       : list[str]
    """
    trajs_groups, z_list, s0_env_list, titles = [], [], [], []
    z_shared = None

    for i, ckpt in enumerate(checkpoints):
        models = get_models_for_checkpoint(ckpt)

        if z_strategy == "shared_first":
            if i == 0:
                trajs_xy, z0, s0_env = rollout_one_checkpoint(
                    env, s0_obs_ds, s0_ag_ds, models,
                    N_trajs=N_trajs, horizon=horizon, seed=seed,
                    resample_skill_per_traj=False, use_prior=True, z_fixed=None
                )
                z_shared = z0
            else:
                trajs_xy, z0, s0_env = rollout_one_checkpoint(
                    env, s0_obs_ds, s0_ag_ds, models,
                    N_trajs=N_trajs, horizon=horizon, seed=seed + 17*i,
                    resample_skill_per_traj=False, use_prior=True, z_fixed=z_shared
                )
        else:
            trajs_xy, z0, s0_env = rollout_one_checkpoint(
                env, s0_obs_ds, s0_ag_ds, models,
                N_trajs=N_trajs, horizon=horizon, seed=seed + 17*i,
                resample_skill_per_traj=False, use_prior=True, z_fixed=None
            )

        trajs_groups.append(trajs_xy)
        z_list.append(z0)
        s0_env_list.append(s0_env)
        titles.append(f"β={parse_beta(ckpt)}")

    return trajs_groups, z_list, s0_env_list, titles


# -------------------------
# Plotting: GRID versions (one tile per checkpoint)
# -------------------------
def plot_xy_trajectories_grid(
    trajs_xy_groups,
    titles,
    ncols=4,
    share_limits=True,
    stem="traj_grid",
    save=False,
    meta=None,
):
    G = len(trajs_xy_groups)
    nrows = int(math.ceil(G / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8*ncols, 6*nrows))
    axes = np.array(axes).reshape(-1)

    xlim = ylim = None
    if share_limits:
        all_pts = []
        for trajs in trajs_xy_groups:
            if len(trajs) > 0:
                all_pts.append(np.concatenate(trajs, axis=0))
        all_pts = np.concatenate(all_pts, axis=0)
        lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
        pad = 0.05 * (hi - lo + 1e-9)
        xlim = (lo[0]-pad[0], hi[0]+pad[0])
        ylim = (lo[1]-pad[1], hi[1]+pad[1])

    for i, ax in enumerate(axes):
        if i >= G:
            ax.axis("off")
            continue

        trajs = trajs_xy_groups[i]
        colors = cm.get_cmap("viridis", max(1, len(trajs)))

        for j, xy in enumerate(trajs):
            c = colors(j)
            ax.plot(xy[:, 0], xy[:, 1], "-", lw=1.5, alpha=0.9, color=c)
            ax.scatter(xy[0, 0],  xy[0, 1],  s=40, marker="o", color=c, edgecolor="k", zorder=3)
            ax.scatter(xy[-1,0], xy[-1,1], s=70, marker="*", color=c, edgecolor="k", zorder=3)
            tcolors = np.linspace(0, 1, len(xy))
            ax.scatter(xy[:,0], xy[:,1], c=tcolors, cmap="viridis", s=14, alpha=0.8)

        ax.set_title(titles[i])
        ax.set_aspect("equal", "box")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        if xlim: ax.set_xlim(*xlim)
        if ylim: ax.set_ylim(*ylim)

    plt.tight_layout()
    if save:
        save_fig(fig, stem=stem, meta=meta, close=True)
    else:
        plt.show()
    return fig


def plot_global_map_grid(
    all_xy,
    trajs_groups,
    titles,
    s0_xy,
    ncols=4,
    share_limits=True,
    stem="global_map_grid",
    save=False,
    meta=None,
):
    G = len(trajs_groups)
    nrows = int(math.ceil(G / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5*ncols, 6*nrows))
    axes = np.array(axes).reshape(-1)

    xlim = ylim = None
    if share_limits:
        pts = [all_xy]
        for trajs in trajs_groups:
            pts.append(np.concatenate(trajs, axis=0))
        pts = np.concatenate(pts, axis=0)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        pad = 0.03 * (hi - lo + 1e-9)
        xlim = (lo[0]-pad[0], hi[0]+pad[0])
        ylim = (lo[1]-pad[1], hi[1]+pad[1])

    for i, ax in enumerate(axes):
        if i >= G:
            ax.axis("off"); continue

        trajs = trajs_groups[i]
        ax.scatter(all_xy[:, 0], all_xy[:, 1], s=2, alpha=0.12, color="lightgray", zorder=1)

        colors = plt.cm.viridis(np.linspace(0, 1, len(trajs)))
        for traj, c in zip(trajs, colors):
            ax.plot(traj[:, 0], traj[:, 1], "-", lw=1.6, color=c, alpha=0.95, zorder=3)
            ax.scatter(traj[-1, 0], traj[-1, 1], marker="*", s=55, color=c, zorder=4)

        ax.scatter([s0_xy[0]], [s0_xy[1]], s=90, c="orange", edgecolors="k", zorder=5)

        ax.set_title(titles[i])
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        if xlim: ax.set_xlim(*xlim)
        if ylim: ax.set_ylim(*ylim)

    plt.tight_layout()
    if save:
        save_fig(fig, stem=stem, meta=meta, close=True)
    else:
        plt.show()
    return fig


def plot_endpoints_vs_tawm_grid(
    trajs_groups,
    s0_env_list,
    z_list,
    titles,
    checkpoints,
    s0_xy=None,
    ncols=4,
    stem="endpoints_vs_tawm_grid",
    save=False,
    meta=None,
):
    G = len(trajs_groups)
    nrows = int(math.ceil(G / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5*ncols, 6*nrows))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i >= G:
            ax.axis("off"); continue

        trajs_xy = trajs_groups[i]
        s0_env   = s0_env_list[i]
        z        = z_list[i].to(device)

        models = get_models_for_checkpoint(checkpoints[i])

        # endpoints
        end_xy = np.stack([traj[-1] for traj in trajs_xy], axis=0)

        # TAWM ellipse
        mean_pred, cov_pred = tawm_xy_gaussian_for_ckpt(s0_env, z, models)

        draw_cov_ellipse(ax, mean_pred, cov_pred, n_std=1.0,
                         edgecolor="darkorange", linewidth=2)
        draw_cov_ellipse(ax, mean_pred, cov_pred, n_std=2.0,
                         edgecolor="orange", linestyle="--", linewidth=1.5)

        ax.scatter(end_xy[:, 0], end_xy[:, 1],
                   s=65, marker="*", edgecolor="k", linewidths=0.6)
        ax.scatter([mean_pred[0]], [mean_pred[1]],
                   c="darkorange", s=60, marker="X")

        if s0_xy is not None:
            ax.scatter([s0_xy[0]], [s0_xy[1]],
                       s=60, c="tab:green", marker="o", edgecolor="k", linewidths=0.6)

        ax.set_title(titles[i])
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    plt.tight_layout()
    if save:
        save_fig(fig, stem=stem, meta=meta, close=True)
    else:
        plt.show()
    return fig


# -------------------------
# Posterior demo dataset (optional)
# -------------------------
def make_episode_splits(minari_dataset, train=0.8, val=0.0, test=0.2, seed=0):
    episodes = list(minari_dataset.iterate_episodes())
    n = len(episodes)
    idxs = list(range(n))
    rng = np.random.RandomState(seed)
    rng.shuffle(idxs)
    n_train = int(round(train * n))
    n_val   = int(round(val   * n))
    train_ids = idxs[:n_train]
    val_ids   = idxs[n_train:n_train+n_val]
    test_ids  = idxs[n_train+n_val:]
    return train_ids, val_ids, test_ids

class SubtrajDataset(Dataset):
    """
    items are tuples: (s0, state_seq, action_seq, sT)
    where state_seq has obs+ach_goal concatenated
    """
    def __init__(self, minari_dataset, T, episode_ids, stride=3):
        self.T = T
        self.items = []

        ep_id_set = set(episode_ids)
        for ep_idx, ep in enumerate(minari_dataset.iterate_episodes()):
            if ep_idx not in ep_id_set:
                continue

            obs = ep.observations["observation"]
            ach = ep.observations["achieved_goal"]
            act = ep.actions
            Ltot = len(obs)
            if Ltot < T + 1:
                continue

            state_ext = np.concatenate([obs, ach], axis=-1).astype(np.float32)

            for t in range(0, Ltot - T, stride):
                state_seq  = state_ext[t:t+T]
                s0 = state_seq[0]
                action_seq = act[t:t+T].astype(np.float32)
                sT = state_ext[t+T]
                self.items.append((s0, state_seq, action_seq, sT))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s0, S, A, sT = self.items[i]
        return {
            "s0": torch.as_tensor(s0, dtype=torch.float32),
            "state_sequence": torch.as_tensor(S, dtype=torch.float32),
            "action_sequence": torch.as_tensor(A, dtype=torch.float32),
            "sT": torch.as_tensor(sT, dtype=torch.float32),
        }


@torch.no_grad()
def sample_z_from_posterior_for_ckpt(S_seq, A_seq, models):
    q_phi = models["q_phi"]
    q_phi.eval()
    mu_z, std_z = q_phi(S_seq, A_seq)
    z = mu_z + std_z * torch.randn_like(mu_z)
    return z.squeeze(0)  # [Z]


def plot_posterior_grid(
    env,
    ds,
    checkpoints,
    K=5,
    tag="train",
    N_rollouts=20,
    horizon=T,
    ncols=4,
    seed=123,
):
    """
    For K random dataset subtrajectories:
      For each checkpoint:
        - sample z from posterior (that checkpoint’s q_phi) on that subtraj
        - rollout pi_theta with fixed z
        - overlay dataset subtrajectory + TAWM ellipses
      -> Save ONE FIGURE PER (k) containing one tile per checkpoint
    """
    rng = np.random.default_rng(seed)

    for k in range(K):
        idx = int(rng.integers(0, len(ds)))
        item = ds[idx]

        s0_raw = item["s0"].cpu().numpy()
        s0_obs_ds = s0_raw[:27]
        s0_ag_ds  = s0_raw[27:]
        demo_xy = item["state_sequence"].cpu().numpy()[:, -2:]  # last 2 dims = achieved_goal xy

        trajs_groups = []
        titles = []
        s0_env_list = []
        z_list = []

        for ckpt in checkpoints:
            models = get_models_for_checkpoint(ckpt)

            S_seq = item["state_sequence"].unsqueeze(0).to(device)
            A_seq = item["action_sequence"].unsqueeze(0).to(device)
            z = sample_z_from_posterior_for_ckpt(S_seq, A_seq, models)

            trajs_xy, z_used, s0_env = rollout_one_checkpoint(
                env,
                s0_obs_ds,
                s0_ag_ds,
                models,
                N_trajs=N_rollouts,
                horizon=horizon,
                seed=10000 + 31*k,
                resample_skill_per_traj=False,
                use_prior=False,
                z_fixed=z,
            )

            trajs_groups.append(trajs_xy)
            titles.append(f"β={parse_beta(ckpt)}")
            s0_env_list.append(s0_env)
            z_list.append(z.detach().cpu())

        # ---- plot grid with overlay per tile ----
        G = len(trajs_groups)
        nrows = int(math.ceil(G / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.8*ncols, 6*nrows))
        axes = np.array(axes).reshape(-1)

        # shared limits (include demo_xy)
        all_pts = [demo_xy]
        for trajs in trajs_groups:
            all_pts.append(np.concatenate(trajs, axis=0))
        all_pts = np.concatenate(all_pts, axis=0)
        lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
        pad = 0.06 * (hi - lo + 1e-9)
        xlim = (lo[0]-pad[0], hi[0]+pad[0])
        ylim = (lo[1]-pad[1], hi[1]+pad[1])

        for i, ax in enumerate(axes):
            if i >= G:
                ax.axis("off"); continue

            trajs_xy = trajs_groups[i]
            ckpt = checkpoints[i]
            models = get_models_for_checkpoint(ckpt)
            s0_env = s0_env_list[i]
            z = z_list[i].to(device)

            # rollouts
            colors = cm.get_cmap("viridis", max(1, len(trajs_xy)))
            for j, xy in enumerate(trajs_xy):
                c = colors(j)
                ax.plot(xy[:,0], xy[:,1], "-", lw=1.2, alpha=0.9, color=c)
                ax.scatter(xy[0,0], xy[0,1], s=25, marker="o", color=c, edgecolor="k", zorder=3)
                ax.scatter(xy[-1,0], xy[-1,1], s=50, marker="*", color=c, edgecolor="k", zorder=3)

            # dataset subtrajectory overlay
            ax.plot(demo_xy[:,0], demo_xy[:,1], "k--", lw=3, alpha=0.85)
            ax.scatter(demo_xy[0,0], demo_xy[0,1], s=80, marker="s", c="k", edgecolor="w", zorder=5)
            ax.scatter(demo_xy[-1,0], demo_xy[-1,1], s=90, marker="*", c="k", edgecolor="w", zorder=5)

            # TAWM ellipses
            mean_pred, cov_pred = tawm_xy_gaussian_for_ckpt(s0_env, z, models)
            draw_cov_ellipse(ax, mean_pred, cov_pred, n_std=1.0, edgecolor="darkorange", linewidth=2)
            draw_cov_ellipse(ax, mean_pred, cov_pred, n_std=2.0, edgecolor="orange", linestyle="--", linewidth=1.5)
            ax.scatter([mean_pred[0]], [mean_pred[1]], c="darkorange", s=60, marker="X", zorder=4)

            ax.set_title(titles[i])
            ax.set_aspect("equal", "box")
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)

        plt.tight_layout()
        meta = {"tag": tag, "k": k, "idx": idx}
        save_fig(fig, stem="posterior_grid", meta=meta, close=True)


# -------------------------
# Main
# -------------------------
def main():
    env = recover_ant_env()
    all_xy = collect_all_xy(ant_maze_dataset)

    # ---- 1) PRIOR rollouts from random dataset states (one grid per state) ----
    random_states = sample_random_states_from_dataset(
        ant_maze_dataset,
        num_states=50,
        seed=123
    )

    for k, (ep_idx, t, s0_obs_ds, s0_ag_ds) in enumerate(random_states):
        print(f"\n[prior-grid] episode {ep_idx}, timestep {t} (k={k})")

        trajs_groups, z_list, s0_env_list, titles = rollout_all_checkpoints(
            env,
            s0_obs_ds,
            s0_ag_ds,
            checkpoints,
            N_trajs=20,
            horizon=40,
            seed=1000 + k,
            z_strategy="shared_first",   # or "shared_first"
        )

        s0_xy = s0_ag_ds[:2]
        meta = {"mode": "prior", "ep": ep_idx, "t": t, "k": k}

        plot_global_map_grid(
            all_xy, trajs_groups, titles, s0_xy,
            ncols=4, share_limits=True,
            save=True, stem="global_map_grid", meta=meta
        )

        plot_xy_trajectories_grid(
            trajs_groups, titles,
            ncols=4, share_limits=True,
            save=True, stem="traj_grid", meta=meta
        )

        plot_endpoints_vs_tawm_grid(
            trajs_groups, s0_env_list, z_list, titles, checkpoints,
            s0_xy=s0_xy, ncols=4,
            save=True, stem="endpoints_vs_tawm_grid", meta=meta
        )

    # ---- 2) OPTIONAL: POSTERIOR rollouts demo (uncomment if you want) ----
    train_ids, _, test_ids = make_episode_splits(ant_maze_dataset, train=0.8, val=0.0, test=0.2, seed=0)
    train_ds = SubtrajDataset(ant_maze_dataset, T=T, episode_ids=train_ids, stride=3)
    test_ds  = SubtrajDataset(ant_maze_dataset, T=T, episode_ids=test_ids,  stride=3)
    print(f"\nposterior datasets: train={len(train_ds)}  test={len(test_ds)}")

    plot_posterior_grid(env, train_ds, checkpoints, K=10, tag="train", N_rollouts=25, horizon=T, ncols=4, seed=123)
    plot_posterior_grid(env, test_ds,  checkpoints, K=10, tag="test",  N_rollouts=25, horizon=T, ncols=4, seed=999)


if __name__ == "__main__":
    main()
