import os
import torch
import numpy as np


# Load and save the model to a .pth file
def save_checkpoint(path, q_phi, pi_theta, p_psi, p_omega, B, T, Z_DIM, NUM_NEURONS, device):
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
    ckpt = torch.load(path, weights_only=False, map_location=torch.device('cpu'))
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

def pack_state_from_obs(obs):
    """
    Build 29-d state (observation + achieved goal).
    """
    obs_vec = np.asarray(obs["observation"], dtype=np.float32).ravel()
    ag = np.asarray(obs.get("achieved_goal", obs_vec[:2]), dtype=np.float32).ravel()

    state = np.concatenate([obs_vec.astype(np.float32), ag.astype(np.float32)], 0) # Combines the 27-d and 2-d tensors
    return obs_vec.astype(np.float32), ag.astype(np.float32), state

def read_antmaze_obs(env):
    """Reconstruct AntMaze dict-observation from MuJoCo state."""
    t = env
    for attr in ("env", "unwrapped"):
        if hasattr(t, attr):
            t = getattr(t, attr)
    qpos = t.data.qpos.ravel()
    qvel = t.data.qvel.ravel()
    obs27 = np.concatenate([qpos[2:], qvel]).astype(np.float32)
    ag2 = qpos[:2].astype(np.float32)
    return {"observation": obs27, "achieved_goal": ag2}
