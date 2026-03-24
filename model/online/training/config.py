from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Environment dimensions
    obs_dim: int = 29
    action_dim: int = 8

    # Latent dimensions
    s_dim: int = 16        # abstract state dimension
    z_dim: int = 256       # skill dimension
    h_dim: int = 256       # hidden layer width (matching NUM_NEURONS in skill_model.py)

    # Temporal abstraction
    H: int = 40            # real timesteps per abstract timestep

    # RSSM (deterministic hidden state dimension)
    rssm_h_dim: int = 256

    # Skill encoder (Transformer)
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

    # Skill prior type
    prior_type: str = "uni"   # "uni" for unimodal, "mog" for mixture-of-Gaussians


@dataclass
class TrainConfig:
    # Loss weights
    beta: float = 1.0          # skill KL weight
    alpha_s: float = 0.1       # state KL weight
    reward_weight: float = 1.0

    # KL balancing (DreamerV2-style)
    kl_balance: bool = True
    kl_balance_alpha: float = 0.8

    # Optimiser
    lr: float = 2e-5
    grad_clip: float = 1.0

    # EM inner iterations per batch
    e_steps: int = 1
    m_steps: int = 1

    # Training schedule
    epochs: int = 1000
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
        summary = (
            f"klbal-{self.kl_balance}_alpha-{self.kl_balance_alpha}"
            f"_beta-{self.beta}_as-{self.alpha_s}"
            f"_rw-{self.reward_weight}_lr-{self.lr}"
        )
        if not self.wandb_run_name:
            self.wandb_run_name = summary
        if not self.save_path:
            self.save_path = f"checkpoints/dreamer_rssm/{summary}"