from dataclasses import dataclass, field
from datetime import date

from numpy.random import f

@dataclass
class ModelConfig:
    # Environment dimensions
    obs_dim: int = 29
    action_dim: int = 8
    goal_dim: int = 2        # raw goal (e.g. desired_goal xy) for distance-based planning cost

    # Skill dimension
    z_dim: int = 256
    h_dim: int = 256       # hidden layer width for MLPs

    # Temporal abstraction
    H: int = 10            # real timesteps per abstract timestep (T = H)

    # RSSM
    rssm_h_dim: int = 256       # deterministic GRU hidden state
    rssm_stoch_dim: int = 32    # stochastic latent per timestep

    @property
    def rssm_feature_dim(self) -> int:
        return self.rssm_h_dim + self.rssm_stoch_dim

    # Skill and state encoder (Transformer)
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

    # Skill prior type
    prior_type: str = "uni"   # "uni" for unimodal, "mog" for mixture-of-Gaussians


@dataclass
class TrainConfig:
    # Loss weights
    recon_weight: float = 1.0   # observation + reward reconstruction in L_RSSM
    free_nats: float = 1.0      # Dreamer-style KL floor; <=0 disables

    # KL balancing (DreamerV2-style, applied to RSSM per-step KL)
    kl_balance: bool = False
    kl_balance_alpha: float = 0.8

    # Optimiser
    lr: float = 2e-5
    grad_clip: float = 1.0

    # EM inner iterations per batch
    e_steps: int = 1
    m_steps: int = 1

    # Training schedule
    epochs: int = 10000
    batch_size: int = 100
    checkpoint_every: int = 50

    # Dataset
    dataset_name: str = "D4RL/antmaze/medium-diverse-v1"
    stride: int = 1
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 0

    # Logging
    wandb_project: str = "tawm-dreamer"
    wandb_run_name: str = ""
    save_path: str = ""

    def __post_init__(self):
        ds_short = self.dataset_name.rsplit("/", 1)[-1] if "/" in self.dataset_name else self.dataset_name
        summary = (
            f"{date.today().strftime('%Y-%m-%d')}"
            f"_{ds_short}"
            f"_lr{self.lr}"
            f"_H{self.H}"
            f"rssm_dim{self.rssm_stoch_dim}"
            f"_fn{self.free_nats}"
            f"_e{self.e_steps}m{self.m_steps}"
        )
        if not self.wandb_run_name:
            self.wandb_run_name = summary
        if not self.save_path:
            self.save_path = f"checkpoints/rssm_causal/{summary}"

    def make_summary(self, model_cfg: "ModelConfig") -> str:
        """Full summary string incorporating both train and model config."""
        ds_short = self.dataset_name.rsplit("/", 1)[-1] if "/" in self.dataset_name else self.dataset_name
        return (
            f"{date.today().strftime('%Y-%m-%d')}"
            f"_{ds_short}"
            f"_H{model_cfg.H}"
            f"_z{model_cfg.z_dim}"
            f"_rssm-h{model_cfg.rssm_h_dim}-s{model_cfg.rssm_stoch_dim}"
            f"_lr{self.lr}"
            f"_bs{self.batch_size}"
            f"_fn{self.free_nats}"
            f"_rw{self.recon_weight}"
            f"_e{self.e_steps}m{self.m_steps}"
        )
