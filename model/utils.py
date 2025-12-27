import os
import torch


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
    ckpt = torch.load(path)
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





