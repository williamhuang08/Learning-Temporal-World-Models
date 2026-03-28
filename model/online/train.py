"""
Training script for the Dreamer-style segment latent model with temporally-abstracted skills.

Usage:
    python -m model.online.train
    python -m model.online.train --epochs 500 --lr 1e-4 --beta 0.5
    python -m model.online.train --dataset_name "D4RL/antmaze/umaze-diverse-v1"
"""

import argparse
import random
import numpy as np
import torch
import minari
import wandb
from torch.utils.data import DataLoader

from model.online.training.config import ModelConfig, TrainConfig
from model.online.models import (
    SegmentObservationDecoder,
    StartStateEncoder,
    StatePosteriorTransformer,
    SegmentDynamics,
    TransformerSkillEncoder,
    RewardModel,
    AbstractSkillPrior,
)
from model.online.models.ll_policy import SkillPolicy
from model.online.utils.buffer import DreamerSubtrajDataset, dreamer_collate, compute_stats
from model.online.utils.buffer import make_episode_splits
from model.online.training.trainer import dreamer_training_with_val


def parse_args():
    parser = argparse.ArgumentParser(description="Train Dreamer segment latent model + skills")

    # model
    parser.add_argument("--obs_dim", type=int, default=ModelConfig.obs_dim)
    parser.add_argument("--action_dim", type=int, default=ModelConfig.action_dim)
    parser.add_argument("--s_dim", type=int, default=ModelConfig.s_dim)
    parser.add_argument("--z_dim", type=int, default=ModelConfig.z_dim)
    parser.add_argument("--h_dim", type=int, default=ModelConfig.h_dim)
    parser.add_argument("--H", type=int, default=ModelConfig.H)
    parser.add_argument("--d_model", type=int, default=ModelConfig.d_model)
    parser.add_argument("--n_heads", type=int, default=ModelConfig.n_heads)
    parser.add_argument("--n_layers", type=int, default=ModelConfig.n_layers)
    parser.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    parser.add_argument("--goal_dim", type=int, default=ModelConfig.goal_dim,
                        help="Raw goal dimension for reward head (AntMaze xy=2)")

    # training
    parser.add_argument("--beta", type=float, default=TrainConfig.beta)
    parser.add_argument("--alpha_s", type=float, default=TrainConfig.alpha_s)
    parser.add_argument("--reward_weight", type=float, default=TrainConfig.reward_weight)
    parser.add_argument("--recon_weight", type=float, default=TrainConfig.recon_weight,
                        help="Weight on Gaussian NLL for decoding o_0 and o_H from s_0,s_1")
    parser.add_argument("--kl_balance", action=argparse.BooleanOptionalAction, default=TrainConfig.kl_balance)
    parser.add_argument("--kl_balance_alpha", type=float, default=TrainConfig.kl_balance_alpha)
    parser.add_argument("--free_nats", type=float, default=TrainConfig.free_nats,
                        help="Dreamer KL floor for training (max(free_nats,KL)); 0 disables")
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--grad_clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--e_steps", type=int, default=TrainConfig.e_steps)
    parser.add_argument("--m_steps", type=int, default=TrainConfig.m_steps)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--checkpoint_every", type=int, default=TrainConfig.checkpoint_every)

    # dataset
    parser.add_argument("--dataset_name", type=str, default=TrainConfig.dataset_name)
    parser.add_argument("--stride", type=int, default=TrainConfig.stride)
    parser.add_argument("--train_frac", type=float, default=TrainConfig.train_frac)
    parser.add_argument("--val_frac", type=float, default=TrainConfig.val_frac)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)

    # logging / infra
    parser.add_argument("--wandb_project", type=str, default=TrainConfig.wandb_project)
    parser.add_argument("--wandb_run_name", type=str, default=TrainConfig.wandb_run_name)
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--normalize_obs", action="store_true",
                        help="Normalize observations with per-feature mean/std")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (default: auto-detect)")

    return parser.parse_args()


def main():
    args = parse_args()

    # --- reproducibility ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- configs ---
    model_cfg = ModelConfig(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        s_dim=args.s_dim,
        z_dim=args.z_dim,
        h_dim=args.h_dim,
        H=args.H,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        goal_dim=args.goal_dim,
    )
    train_cfg = TrainConfig(
        beta=args.beta,
        alpha_s=args.alpha_s,
        reward_weight=args.reward_weight,
        recon_weight=args.recon_weight,
        kl_balance=args.kl_balance,
        kl_balance_alpha=args.kl_balance_alpha,
        free_nats=args.free_nats,
        lr=args.lr,
        grad_clip=args.grad_clip,
        e_steps=args.e_steps,
        m_steps=args.m_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        dataset_name=args.dataset_name,
        stride=args.stride,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        save_path=args.save_path,
    )

    # --- dataset ---
    print(f"Loading dataset: {train_cfg.dataset_name}")
    minari_ds = minari.load_dataset(train_cfg.dataset_name)

    train_ids, val_ids, test_ids = make_episode_splits(
        minari_ds,
        train=train_cfg.train_frac,
        val=train_cfg.val_frac,
        test=train_cfg.test_frac,
        seed=train_cfg.seed,
    )
    print(f"Episodes  — train: {len(train_ids)}  val: {len(val_ids)}  test: {len(test_ids)}")

    train_ds = DreamerSubtrajDataset(minari_ds, H=model_cfg.H, episode_ids=train_ids,
                                     stride=train_cfg.stride)
    val_ds = DreamerSubtrajDataset(minari_ds, H=model_cfg.H, episode_ids=val_ids,
                                   stride=train_cfg.stride)

    if args.normalize_obs:
        S_mean, S_std = compute_stats(train_ds)
        train_ds.stats = (S_mean, S_std)
        val_ds.stats = (S_mean, S_std)
        print(f"Observation normalization enabled (mean/std from train set)")
    else:
        print(f"Observation normalization disabled")

    print(f"Segments  — train: {len(train_ds)}  val: {len(val_ds)}")
    print(f"  (removed: train {len(train_ds.removed_items)}, val {len(val_ds.removed_items)})")

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size,
        shuffle=True, collate_fn=dreamer_collate, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size,
        shuffle=False, collate_fn=dreamer_collate, drop_last=False,
    )

    # --- models ---
    start_enc = StartStateEncoder(obs_dim=model_cfg.obs_dim, s_dim=model_cfg.s_dim) # s0 encoder
    s1_post_enc = StatePosteriorTransformer( # s1 encoders
        obs_dim=model_cfg.obs_dim,
        action_dim=model_cfg.action_dim,
        s_dim=model_cfg.s_dim,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        dropout=model_cfg.dropout,
    )
    segment_dynamics = SegmentDynamics(s_dim=model_cfg.s_dim, z_dim=model_cfg.z_dim) # TAWM
    skill_enc = TransformerSkillEncoder( # z0
        obs_dim=model_cfg.obs_dim,
        action_dim=model_cfg.action_dim,
        z_dim=model_cfg.z_dim,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        dropout=model_cfg.dropout,
    )
    skill_prior = AbstractSkillPrior( # \hat{z0}
        s_dim=model_cfg.s_dim,
        z_dim=model_cfg.z_dim,
        h_dim=model_cfg.h_dim,
    )
    reward_model = RewardModel(
        s_dim=model_cfg.s_dim,
        z_dim=model_cfg.z_dim,
        goal_dim=model_cfg.goal_dim,
        h_dim=model_cfg.h_dim,
    )
    pi_theta = SkillPolicy( # ll policy
        state_dim=model_cfg.obs_dim,
        action_dim=model_cfg.action_dim,
    )
    obs_decoder = SegmentObservationDecoder( # o0, oH
        obs_dim=model_cfg.obs_dim,
        s_dim=model_cfg.s_dim,
        h_dim=model_cfg.h_dim,
    )

    total_params = sum(
        sum(p.numel() for p in m.parameters())
        for m in [
            start_enc,
            s1_post_enc,
            segment_dynamics,
            skill_enc,
            skill_prior,
            reward_model,
            pi_theta,
            obs_decoder,
        ]
    )
    print(f"\nTotal parameters: {total_params:,}")
    print(f"  Start encoder:        {sum(p.numel() for p in start_enc.parameters()):,}")
    print(f"  s1 posterior (TF):    {sum(p.numel() for p in s1_post_enc.parameters()):,}")
    print(f"  Segment dynamics:     {sum(p.numel() for p in segment_dynamics.parameters()):,}")
    print(f"  Skill encoder:        {sum(p.numel() for p in skill_enc.parameters()):,}")
    print(f"  Skill prior:    {sum(p.numel() for p in skill_prior.parameters()):,}")
    print(f"  Reward model:   {sum(p.numel() for p in reward_model.parameters()):,}")
    print(f"  Obs decoder:    {sum(p.numel() for p in obs_decoder.parameters()):,}")
    print(f"  Policy:         {sum(p.numel() for p in pi_theta.parameters()):,}")

    # --- wandb ---
    wandb.init(
        project=train_cfg.wandb_project,
        name=train_cfg.wandb_run_name or None,
        config={**model_cfg.__dict__, **train_cfg.__dict__},
    )

    # --- train ---
    print(f"\nStarting training for {train_cfg.epochs} epochs...")
    history = dreamer_training_with_val(
        save_path=train_cfg.save_path,
        train_loader=train_loader,
        val_loader=val_loader,
        start_enc=start_enc,
        s1_post_enc=s1_post_enc,
        segment_dynamics=segment_dynamics,
        skill_enc=skill_enc,
        pi_theta=pi_theta,
        skill_prior=skill_prior,
        reward_model=reward_model,
        obs_decoder=obs_decoder,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        device=device,
    )

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
