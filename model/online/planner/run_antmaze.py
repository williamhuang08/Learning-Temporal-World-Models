"""
End-to-end AntMaze planning with the Dreamer-style RSSM + CEM planner.

Usage:
    python -m model.online.planner.run_antmaze --checkpoint PATH
    python -m model.online.planner.run_antmaze --checkpoint PATH --num_trials 10
    # Start from a random training subtraj s0 (usually mid-episode):
    python -m model.online.planner.run_antmaze --checkpoint PATH --start dataset_random
    # Fixed subtraj index:
    python -m model.online.planner.run_antmaze --checkpoint PATH --start dataset_index --init_subtraj_idx 1234
"""

import argparse
import os
import numpy as np
import torch
import mujoco
import minari
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread

from model.online.training.config import ModelConfig, TrainConfig
from model.online.models import AbstractRSSM, TransformerSkillEncoder, RewardModel, AbstractSkillPrior
from model.online.models.ll_policy import SkillPolicy
from model.online.utils.buffer import DreamerSubtrajDataset, make_episode_splits
from model.online.planner.planner import DreamerCEMPlanner


# ---------------------------------------------------------------------------
# Environment helpers (from planner/plan_skills_antmaze.py)
# ---------------------------------------------------------------------------

def obs_to_state_vec(obs_dict):
    return np.concatenate(
        [obs_dict["observation"], obs_dict["achieved_goal"]], axis=-1
    ).astype(np.float32)


def xy_from_state(s):
    return s[..., -2:]


def strip_timelimit(env):
    while hasattr(env, "env") and env.__class__.__name__ == "TimeLimit":
        env = env.env
    return env


def get_sim_handles(env):
    t = env
    for attr in ("env", "unwrapped"):
        if hasattr(t, attr):
            t = getattr(t, attr)
    if hasattr(t, "model") and hasattr(t, "data"):
        return t, t.model, t.data


def set_env_state(env, qpos, qvel):
    _, model, data = get_sim_handles(env)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)


def read_antmaze_obs(env):
    t = env
    for attr in ("env", "unwrapped"):
        if hasattr(t, attr):
            t = getattr(t, attr)
    qpos = t.data.qpos.ravel()
    qvel = t.data.qvel.ravel()
    obs27 = np.concatenate([qpos[2:], qvel]).astype(np.float32)
    ag2 = qpos[:2].astype(np.float32)
    return {"observation": obs27, "achieved_goal": ag2}


def split_obs_to_qpos_qvel(s0_obs, s0_ag, env):
    """Map dataset-style (obs27, achieved_goal2) to MuJoCo qpos/qvel (same as offline plotting)."""
    _, model, data = get_sim_handles(env)
    nq, nv = int(model.nq), int(model.nv)
    s0_obs = np.asarray(s0_obs, np.float32).ravel()
    s0_ag = np.asarray(s0_ag, np.float32).ravel()
    qpos = data.qpos.ravel().copy()
    qvel = data.qvel.ravel().copy()
    qpos[0:2] = s0_ag
    qpos[2:nq] = s0_obs[: (nq - 2)]
    qvel[:nv] = s0_obs[(nq - 2) : (nq - 2 + nv)]
    return qpos.astype(np.float32), qvel.astype(np.float32)


# ---------------------------------------------------------------------------
# Skill execution in real environment
# ---------------------------------------------------------------------------

def run_skill_seq(env, planner, state_vec, z_seq, H=40,
                  goal_xy=None, goal_thresh2=1.0, deterministic=False,
                  executed_xy=None):
    """
    Execute a sequence of skills in the real environment.

    Args:
        env: gymnasium env
        planner: DreamerCEMPlanner (for policy_action)
        state_vec: np.ndarray [obs_dim]
        z_seq: [L, z_dim] skill tensor
        H: real timesteps per skill
        goal_xy: np.ndarray [2] or None
        goal_thresh2: squared distance threshold for goal
        deterministic: if True use mean policy action
        executed_xy: list of xy positions so far
    Returns:
        state_vec, executed_xy, done, per_skill_exec_xy, reached_goal
    """
    if executed_xy is None:
        executed_xy = [xy_from_state(state_vec).copy()]

    L = z_seq.shape[0]
    done = False
    per_skill_exec_xy = []

    for i in range(L):
        z = z_seq[i]
        skill_xy = []

        for t in range(H):
            a = planner.policy_action(state_vec, z, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            state_vec = obs_to_state_vec(obs)
            xy = xy_from_state(state_vec).copy()
            executed_xy.append(xy)
            skill_xy.append(xy)

            if goal_xy is not None and np.sum((xy - goal_xy) ** 2) < goal_thresh2:
                per_skill_exec_xy.append(np.asarray(skill_xy, dtype=np.float32))
                return state_vec, executed_xy, True, per_skill_exec_xy, True

            if done:
                per_skill_exec_xy.append(np.asarray(skill_xy, dtype=np.float32))
                return state_vec, executed_xy, True, per_skill_exec_xy, False

        per_skill_exec_xy.append(np.asarray(skill_xy, dtype=np.float32))

    return state_vec, executed_xy, False, per_skill_exec_xy, False


# ---------------------------------------------------------------------------
# Iterative replanning loop
# ---------------------------------------------------------------------------

def run_skills_iterative_replanning(
    env,
    planner,
    skill_seq_len=10,
    H=40,
    execute_n_skills=1,
    max_replans=50,
    goal_thresh2=1.0,
    deterministic=False,
    outdir=None,
    init_s0_full: np.ndarray | None = None,
):
    """
    Iteratively plan with CEM, execute the first skill(s), then replan.

    If init_s0_full is set [29] = concat(observation_27, achieved_goal_2), teleport the
    simulator to that state after reset (keeps desired_goal from reset). Typical source:
    a subtraj window start from the training dataset (usually mid-episode).
    """
    obs, _ = env.reset()
    goal_xy = obs["desired_goal"].astype(np.float32)[:2]

    if init_s0_full is not None:
        s0 = np.asarray(init_s0_full, dtype=np.float32).ravel()
        s0_obs, s0_ag = s0[:27], s0[27:]
        qpos, qvel = split_obs_to_qpos_qvel(s0_obs, s0_ag, env)
        set_env_state(env, qpos, qvel)
        ro = read_antmaze_obs(env)
        state_vec = obs_to_state_vec(ro)
    else:
        state_vec = obs_to_state_vec(obs)

    first_state_vec = state_vec.copy()
    executed_xy = [xy_from_state(state_vec).copy()]
    all_s0 = [xy_from_state(state_vec).copy()]
    reached_goal = False

    for repl in range(max_replans):
        cur_xy = xy_from_state(state_vec).copy()

        if np.sum((cur_xy - goal_xy) ** 2) < goal_thresh2:
            reached_goal = True
            print("Reached goal (before planning).")
            break

        all_s0.append(cur_xy.copy())

        eps_mean, eps_std, h, s = planner.plan(state_vec, skill_seq_len=skill_seq_len)

        eps_exec = eps_mean[:execute_n_skills]
        z_exec = planner.convert_epsilon_to_z(eps_exec, h, s)

        state_vec, executed_xy, done, per_skill_exec_xy, reached_goal = run_skill_seq(
            env, planner, state_vec, z_exec,
            H=H,
            goal_xy=goal_xy,
            goal_thresh2=goal_thresh2,
            deterministic=deterministic,
            executed_xy=executed_xy,
        )

        dist = np.sum((xy_from_state(state_vec) - goal_xy) ** 2)
        print(f"[replan {repl:03d}] xy={xy_from_state(state_vec)}  dist^2={dist:.3f}")

        if done:
            break

    if outdir is not None:
        save_final_trajectory(
            os.path.join(outdir, "final_executed_trajectory.png"),
            executed_xy, goal_xy, all_s0,
        )

    return (
        np.stack(executed_xy, axis=0),
        goal_xy,
        state_vec,
        first_state_vec,
        all_s0,
        reached_goal,
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_final_trajectory(outpath, executed_xy, goal_xy, all_s0, bg_path=None):
    exec_xy = np.asarray(executed_xy, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7.0, 6.5))

    if bg_path is not None and os.path.exists(bg_path):
        bg_img = imread(bg_path)
        margin = 1
        ax.imshow(bg_img, extent=(-10 - margin, 10 + margin, -10 - margin, 10 + margin),
                  origin="upper", aspect="equal", zorder=0, alpha=0.55)

    if exec_xy.shape[0] > 1:
        ax.plot(exec_xy[:, 0], exec_xy[:, 1], color="red", linewidth=3.0, zorder=2, label="executed")
        ax.scatter(exec_xy[0, 0], exec_xy[0, 1], s=60, color="red", zorder=3)

    ax.scatter(goal_xy[0], goal_xy[1], s=170, marker="*", color="black", zorder=4, label="goal")

    s0_arr = np.asarray(all_s0, dtype=np.float32)
    ax.scatter(s0_arr[:, 0], s0_arr[:, 1], s=30, marker="o", color="green",
               alpha=0.5, zorder=3, label="replan starts")

    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=240)
    plt.close(fig)
    print(f"saved -> {outpath}")


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_models_from_checkpoint(path, device="cpu"):
    """
    Load all models from a Dreamer checkpoint and return them with configs.
    """
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    mcfg = ModelConfig(**ckpt["model_cfg"])
    tcfg = TrainConfig(**ckpt["train_cfg"])

    rssm = AbstractRSSM(
        obs_dim=mcfg.obs_dim, action_dim=mcfg.action_dim,
        s_dim=mcfg.s_dim, z_dim=mcfg.z_dim, h_dim=mcfg.rssm_h_dim,
    )
    skill_enc = TransformerSkillEncoder(
        obs_dim=mcfg.obs_dim, action_dim=mcfg.action_dim,
        z_dim=mcfg.z_dim, d_model=mcfg.d_model,
        n_heads=mcfg.n_heads, n_layers=mcfg.n_layers, dropout=mcfg.dropout,
    )
    skill_prior = AbstractSkillPrior(
        s_dim=mcfg.s_dim, z_dim=mcfg.z_dim, h_dim=mcfg.h_dim,
    )
    reward_model = RewardModel(
        s_dim=mcfg.s_dim, z_dim=mcfg.z_dim, h_dim=mcfg.h_dim,
    )
    pi_theta = SkillPolicy(state_dim=mcfg.obs_dim, action_dim=mcfg.action_dim)

    rssm.load_state_dict(ckpt["rssm"])
    skill_enc.load_state_dict(ckpt["skill_enc"])
    pi_theta.load_state_dict(ckpt["pi_theta"])
    skill_prior.load_state_dict(ckpt["skill_prior"])
    reward_model.load_state_dict(ckpt["reward_model"])

    for m in (rssm, skill_enc, skill_prior, reward_model, pi_theta):
        m.to(device).eval()

    return rssm, skill_enc, skill_prior, reward_model, pi_theta, mcfg, tcfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Dreamer RSSM CEM planner for AntMaze")

    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    p.add_argument("--env_name", type=str, default="D4RL/antmaze/medium-diverse-v1")
    p.add_argument("--outdir", type=str, default="model/online/planner/results")
    p.add_argument("--bg_path", type=str, default="planner/planning/bg_img.jpeg")

    p.add_argument("--skill_seq_len", type=int, default=40)
    p.add_argument("--pop_size", type=int, default=100)
    p.add_argument("--n_iters", type=int, default=100)
    p.add_argument("--frac_keep", type=float, default=0.5)
    p.add_argument("--l2_pen", type=float, default=0.0)
    p.add_argument("--length_cost", type=float, default=0.0)
    p.add_argument("--execute_n_skills", type=int, default=1)
    p.add_argument("--max_replans", type=int, default=50)
    p.add_argument("--goal_thresh2", type=float, default=1.0)
    p.add_argument("--max_episode_steps", type=int, default=4000)

    p.add_argument("--num_trials", type=int, default=1)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--device", type=str, default=None)

    p.add_argument(
        "--start",
        type=str,
        default="reset",
        choices=("reset", "dataset_random", "dataset_index"),
        help="reset: env.reset(). dataset_*: teleport to a subtraj s0 from train split (usually mid-episode).",
    )
    p.add_argument(
        "--init_subtraj_idx",
        type=int,
        default=None,
        help="Used when --start dataset_index (required for that mode).",
    )
    p.add_argument("--dataset_stride", type=int, default=1, help="Stride when building subtraj dataset for init.")
    p.add_argument("--init_seed", type=int, default=0, help="Base seed for dataset_random (offset per trial).")

    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- load models ---
    rssm, skill_enc, skill_prior, reward_model, pi_theta, mcfg, tcfg = \
        load_models_from_checkpoint(args.checkpoint, device=device)
    H = mcfg.H

    # --- environment ---
    data = minari.load_dataset(args.env_name)
    env = data.recover_environment()
    env = strip_timelimit(env)
    env = TimeLimit(env, max_episode_steps=args.max_episode_steps)

    train_ds = None
    if args.start != "reset":
        train_ids, _, _ = make_episode_splits(data, seed=tcfg.seed)
        train_ds = DreamerSubtrajDataset(
            data, H=H, episode_ids=train_ids, stride=args.dataset_stride,
        )
        if len(train_ds) == 0:
            raise RuntimeError("No subtrajectories in train split for dataset init; check H / dataset.")
        if args.start == "dataset_index":
            if args.init_subtraj_idx is None:
                raise ValueError("--start dataset_index requires --init_subtraj_idx")
            if not (0 <= args.init_subtraj_idx < len(train_ds)):
                raise ValueError(f"init_subtraj_idx out of range [0, {len(train_ds)})")

    # --- planner ---
    planner = DreamerCEMPlanner(
        rssm=rssm,
        skill_prior=skill_prior,
        reward_model=reward_model,
        pi_theta=pi_theta,
        device=device,
        skill_seq_len=args.skill_seq_len,
        pop_size=args.pop_size,
        n_iters=args.n_iters,
        frac_keep=args.frac_keep,
        l2_pen=args.l2_pen,
        length_cost=args.length_cost,
    )

    # --- run trials ---
    num_successes = 0
    dists_to_goal = []

    for trial in range(args.num_trials):
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/{args.num_trials}")
        print(f"{'='*60}")

        trial_dir = os.path.join(args.outdir, f"trial_{trial:03d}")
        os.makedirs(trial_dir, exist_ok=True)

        init_s0 = None
        if args.start == "dataset_random":
            rng = np.random.default_rng(args.init_seed + trial)
            j = int(rng.integers(0, len(train_ds)))
            init_s0 = train_ds[j]["s0"].numpy()
            print(f"[init] dataset_random subtraj index {j} / {len(train_ds)}")
        elif args.start == "dataset_index":
            init_s0 = train_ds[args.init_subtraj_idx]["s0"].numpy()
            print(f"[init] dataset_index subtraj {args.init_subtraj_idx}")

        exec_xy, goal_xy, last_state, first_state, all_s0, reached_goal = \
            run_skills_iterative_replanning(
                env, planner,
                skill_seq_len=args.skill_seq_len,
                H=H,
                execute_n_skills=args.execute_n_skills,
                max_replans=args.max_replans,
                goal_thresh2=args.goal_thresh2,
                deterministic=args.deterministic,
                outdir=trial_dir,
                init_s0_full=init_s0,
            )

        last_xy = exec_xy[-1].astype(np.float32)
        dist = np.linalg.norm(goal_xy - last_xy)
        dists_to_goal.append(dist)

        if reached_goal:
            num_successes += 1
            print(f"Trial {trial + 1}: SUCCESS (dist={dist:.3f})")
        else:
            print(f"Trial {trial + 1}: FAIL (dist={dist:.3f})")

    success_rate = num_successes / max(1, args.num_trials)
    print(f"\n{'='*60}")
    print(f"Success rate: {success_rate:.2%} ({num_successes}/{args.num_trials})")
    print(f"Mean dist to goal: {np.mean(dists_to_goal):.3f}")
    print(f"{'='*60}")

    # --- histogram ---
    if len(dists_to_goal) > 1:
        fig, ax = plt.subplots()
        ax.hist(dists_to_goal, bins=30, color="skyblue", edgecolor="black")
        ax.set_title("Histogram of Distances to Goal")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Frequency")
        hist_path = os.path.join(args.outdir, "distance_hist.png")
        plt.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"saved -> {hist_path}")

    env.close()


if __name__ == "__main__":
    main()
