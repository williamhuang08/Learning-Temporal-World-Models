'''File where we will sample a set of waypoints, and plan a sequence of skills to have our pointmass travel through those waypoints'''

import minari
import os
import sys
from model.skill_model import SkillPolicy, SkillPosterior, SkillPrior, TAWM, MoGSkillPrior
from model.utils import load_checkpoint, pack_state_from_obs, read_antmaze_obs
from utils import obs_to_state_vec, xy_from_state
import numpy as np
import random
import torch
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from cem import cem, cem_variable_length

sys.path.append(os.path.abspath(".."))
from utils import SubtrajDataset, make_episode_splits, collate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load in the environment
env_name = 'D4RL/antmaze/medium-diverse-v1'
data = minari.load_dataset(env_name)
env = data.recover_environment()

def strip_timelimit(env):
    "Recovers the innermost env and removes timelimit wrapper"
    while hasattr(env, "env") and env.__class__.__name__ == "TimeLimit":
        env = env.env
    return env

env = strip_timelimit(env)
env = TimeLimit(env, max_episode_steps=4000) # set maximum number of episode steps

# Environment variables
prior_type = "mog"
skill_seq_len = 10
H = 40
replan_freq = H
state_dim = 29 # make sure to change these depending on the env!
a_dim = 8
h_dim = 256
z_dim = 256
batch_size = 100
lr = 1e-4
wd = 0.0
state_dependent_prior = True
state_dec_stop_grad = True
beta = 1.0
alpha = 1.0
ent_pen = 0
max_sig = None
fixed_sig =  0.0
n_iters = 100
a_dist = 'normal'
keep_frac = 0.5
use_epsilon = True
max_ep = None
cem_l2_pen = 0.0
var_pen = 0.0
render = False
variable_length = False
# max_replans = 2000 // H # run max 2000 timesteps
max_replans = 2000 // H
plan_length_cost = 0.0
encoder_type = 'state_action_sequence'
term_state_dependent_prior = False
init_state_dependent = True
random_goal = False # determines if we select a goal at random from dataset (random_goal=True) or use pre-set one from environment

# filename = 'antmaze_diverse_detached_250_1.pth'
# filename = 'kl_balancing/antmaze_diverse_detached_klbalance_epoch75_beta1.0_gamma0.001.pth'
# filename = 'kl_balancing_MoG_knn/antmaze_diverse_detached_klbalance_mog_knn_epoch100_beta0.001_gamma1-lambda0.1.pth'
filename = 'kl_balancing_MoG/beta_gamma/antmaze_diverse_detached_klbalance_mog_epoch100_beta0.001_gamma1-filtered.pth'
PATH = '../checkpoints/' + filename

OUTDIR = "planning"
PLANS_DIR = os.path.join(OUTDIR, "plans_per_replan15")
os.makedirs(PLANS_DIR, exist_ok=True)

skillpost = SkillPosterior(state_dim=state_dim, action_dim=a_dim).to(device)
llpolicy = SkillPolicy(state_dim=state_dim, action_dim=a_dim).to(device)
tawm = TAWM(state_dim=state_dim).to(device)
# skillprior = SkillPrior(state_dim=state_dim).to(device)
skillprior = MoGSkillPrior(state_dim=state_dim).to(device)
_ = load_checkpoint(PATH, skillpost, llpolicy, tawm, skillprior)


train_ids, val_ids, test_ids = make_episode_splits(
    data, train=0.8, val=0.0, test=0.2, seed=0
)
print(f"train episodes:{len(train_ids)}  val episodes:{len(val_ids)}  test episodes:{len(test_ids)}")

train_ds = SubtrajDataset(data, T=H, episode_ids=train_ids, stride=3)
val_ds   = SubtrajDataset(data, T=H, episode_ids=val_ids,   stride=3)
test_ds  = SubtrajDataset(data, T=H, episode_ids=test_ids,  stride=3)

print(f"train subtrajs:{len(train_ds)}  val subtrajs:{len(val_ds)}  test subtrajs:{len(test_ds)}")

B = 100  
train_loader = DataLoader(train_ds, batch_size=B, shuffle=True,  collate_fn=collate, drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=B, shuffle=False, collate_fn=collate, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=B, shuffle=False, collate_fn=collate, drop_last=False)

def _get_start_xy_from_item(item):
    s0, S, A, sT = item
    return s0[-2:].astype(np.float32), S.astype(np.float32)

def build_xy_cache(minari_dataset):
    episodes_xy = []
    starts = []
    all_xy = []

    # scan all subtrajectories
    for i, item in enumerate(train_ds.items):
        xy, states = _get_start_xy_from_item(item)
        episodes_xy.append(states[:, -2:])
        starts.append(xy)
        all_xy.append(states[:, -2:])

    return episodes_xy, np.stack(starts, axis=0), np.concatenate(all_xy, axis=0)

episodes_xy, episodes_start_xy, all_xy = build_xy_cache(data)

def pick_nearby_ep(episodes_start_xy, current_xy):
    d2 = np.sum((episodes_start_xy - current_xy.reshape(1, 2))**2, axis=1)
    idx = np.where(d2<=0.2**2)[0]
    idx = idx[np.argsort(d2[idx])]
    return idx[:40].tolist()

def build_antmaze_background(minari_dataset, bins=300, stride=1):
    xs, ys = [], []

    for ep in minari_dataset.iterate_episodes():
        ag = ep.observations["achieved_goal"].astype(np.float32)
        ag = ag[::stride]
        xs.append(ag[:, 0])
        ys.append(ag[:, 1])

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    occ, xedges, yedges = np.histogram2d(x, y, bins=bins)

    bg_img = np.log1p(occ)  
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])

    return bg_img, extent, x_centers, y_centers, occ

@torch.no_grad()
def mog_mean_std(skillprior, s):
    "Samples one of the K Gaussians for each batch element and samples "
    logits, mean, std = skillprior(s)          
    k = torch.distributions.Categorical(logits=logits).sample()                 
    # gather the mean and std of the corresponding kth gaussian
    idx = k.unsqueeze(-1).unsqueeze(-1).expand(*mean.shape[:-2], 1, mean.shape[-1]) 
    mu_k  = mean.gather(dim=-2, index=idx).squeeze(-2)   
    std_k = std.gather(dim=-2, index=idx).squeeze(-2)    
    return mu_k, std_k


@torch.no_grad()
def policy_action(llpolicy, state_vec, z_vec, deterministic=True):
    """
    Review:
    Uses the low level policy to sample an action.
    """
    s = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)  
    z = z_vec.view(1, -1)                                                      
    mu, std = llpolicy(s, z)

    if deterministic:
        a = mu
    else:
        std = std.clamp_min(0.05)
        a = mu + std * torch.randn_like(mu)

    a = torch.tanh(a)
    return a.squeeze(0).cpu().numpy()

def convert_epsilon_to_z(epsilon, s0_vec, prior_type):
    """
    Converts sequence of epsilons to a sequence of skills.

    epsilon: [B, L, Z]
    s0_vec:  [state_dim]
    returns z_seq: [B, L, Z]
    """
    s = torch.tensor(s0_vec, dtype=torch.float32, device=device).unsqueeze(0)  # [1,state_dim]
    B, L, _ = epsilon.shape
    s = s.expand(B, -1)

    z_seq = []
    for i in range(L):
        if prior_type == "uni":
            mu_z, sigma_z = skillprior(s) # [B,Z]
            eps_i = epsilon[:, i, :] # [B,Z]
            z_i = mu_z + sigma_z * eps_i # [B,Z]
            z_seq.append(z_i.unsqueeze(1)) # [B,1,Z]
        else:   
            logits, mu_z, sigma_z = skillprior(s) # [B,Z]
            k = torch.distributions.Categorical(logits=logits).sample().item()
            mu_k  = mu_z[0, k]                             
            std_k = sigma_z[0, k]    
            eps_i = epsilon[:, i, :] # [B,Z]
            z_i = mu_k + std_k * eps_i # [B,Z]
            z_seq.append(z_i.unsqueeze(1)) # [B,1,Z]
        
        s_mean, _ = tawm(s, z_i)
        s = s_mean
    return torch.cat(z_seq, dim=1)
                

def get_expected_cost_variable_length(s0, skill_seq, lengths, goal_state, use_epsilons=True, plot=False):
		'''
        Returns the cost of each skill_seq in the batch dictated by the length of the skill. 
          
		s0 is initial state, [batch_size, 1, s_dim]
		skill sequence is a [batch_size, skill_seq_len, z_dim] tensor that representents a skill_seq_len sequence of skills
		'''
		batch_size = s0.shape[0]
		goal_state = torch.cat(batch_size * [goal_state],dim=0)
		s_i = s0
		
		skill_seq_len = skill_seq.shape[1]
		pred_states = [s_i]
		costs = (lengths == 0)*torch.mean((xy_from_state(s_i) - xy_from_state(goal_state))**2,dim=-1).squeeze() # compute costs for skills that do not run
		for i in range(skill_seq_len):
			# z_i = skill_seq[:,i:i+1,:] # might need to reshape
			if use_epsilons:
				mu_z, sigma_z = skillprior(s_i)
				z_i = mu_z + sigma_z*skill_seq[:,i:i+1,:]
                
			else:
				z_i = skill_seq[:,i:i+1,:]
			s_mean, s_sig = tawm(s_i,z_i)
			
			# sample s_i+1 using reparameterize
			s_sampled = s_mean
			# s_sampled = self.reparameterize(s_mean, s_sig)
			s_i = s_sampled

			cost_i = (lengths == i+1)*torch.mean((xy_from_state(s_i) - xy_from_state(goal_state))**2,dim=-1).squeeze() # select the cost at the time dictated by the length of the batch
			costs += cost_i
			
			pred_states.append(s_i)
		
		if plot:
			plt.figure()
			plt.scatter(s0[0,0,0].detach().cpu().numpy(),s0[0,0,1].detach().cpu().numpy(), label='init state')
			plt.scatter(goal_state[0,0,0].detach().cpu().numpy(),goal_state[0,0,1].detach().cpu().numpy(),label='goal',s=300)
			# plt.xlim([0,25])
			# plt.ylim([0,25])
			pred_states = torch.cat(pred_states,1)
			for i in range(batch_size):
				# ipdb.set_trace()
				plt.plot(pred_states[i,:lengths[i].item()+1,0].detach().cpu().numpy(),pred_states[i,:lengths[i].item()+1,1].detach().cpu().numpy())
				
			plt.savefig('pred_states_cem_variable_length')


			plt.figure()
			plt.scatter(s0[0,0,0].detach().cpu().numpy(),s0[0,0,1].detach().cpu().numpy(), label='init state')
			plt.scatter(goal_state[0,0,0].detach().cpu().numpy(),goal_state[0,0,1].detach().cpu().numpy(),label='goal',s=300)
			# plt.xlim([0,25])
			# plt.ylim([0,25])
			# pred_states = torch.cat(pred_states,1)
			for i in range(batch_size):
				# ipdb.set_trace()
				plt.plot(pred_states[i,:,0].detach().cpu().numpy(),pred_states[i,:,1].detach().cpu().numpy())
				
			plt.savefig('pred_states_cem_variable_length_FULL_SEQ')
		return costs

def get_expected_cost_for_cem(s0, eps_seq, goal_xy, prior_type, length_cost=0.0):
    """
    Returns the cost of the eps_seq which is the minimum distance along the whole skill sequence to the goal.

    s0: [B,1,sd] ,eps_seq: [B,L,Z] ,goal_xy: [2] 
    returns: [B] costs
    """
    s = s0.squeeze(1) # [B, D]

    goal_xy = goal_xy.view(1, 2).expand(s.shape[0], -1)  # [B,2] (keep second dimension (2) unchanged)

    B, L, _ = eps_seq.shape

    costs = []
    pred_states = [s]
    # cost at t=0
    costs.append(((s[:, -2:] - goal_xy) ** 2).mean(dim=-1))
    for i in range(L):
        if prior_type == "uni":
            mu_z, sigma_z = skillprior(s)       
        else:
            mu_z, sigma_z = mog_mean_std(skillprior, s)

        eps_i = eps_seq[:, i, :]
        z_i = mu_z + sigma_z * eps_i           

        s, _ = tawm(s, z_i)                   
        costs.append(((s[:, -2:] - goal_xy) ** 2).mean(dim=-1) + (i+1)*length_cost) # sequences are preferred when they are closer to the goal earlier on in the sequence
        pred_states.append(s)

    costs = torch.stack(costs, dim=1) # [B,L+1]
    best, _ = torch.min(costs, dim=1)  # [B]
    # return costs[:, -1]
    return best

def run_skills_iterative_replanning(env,
    skill_seq_len=skill_seq_len,
    H=H,
    execute_n_skills=1,   
    max_replans=40000//H,
    use_epsilon=True,
    goal_thresh2=1.0,
    deterministic=False
):
    """
    Starting at current state, use CEM to find the best skill sequence, execute the first skill, then replan and repeat
    """

    obs, _ = env.reset()
    state_vec = obs_to_state_vec(obs)
    first_state_vec = state_vec.copy()
    goal_xy = obs["desired_goal"].astype(np.float32)[:2]
    executed_xy = [state_vec[-2:].copy()]

    first_eps_mean = None
    last_eps_mean = None
    last_eps_std = None
    last_s0_vec = None
    all_s0 = [state_vec[-2:].copy()]

    for repl in range(max_replans):
        cur_xy = state_vec[-2:].copy()
        # stop if already at goal
        if np.sum((state_vec[-2:] - goal_xy) ** 2) < goal_thresh2:
            print("Reached goal (before planning).")
            break

        all_s0.append(cur_xy.copy())
        # CEM
        last_s0_vec = state_vec.copy()
        s_batch = torch.tensor(state_vec, dtype=torch.float32, device=device).view(1,1,-1).expand(batch_size, 1, state_dim)   # [B,1,sd]
        goal_xy_t = torch.tensor(goal_xy, dtype=torch.float32, device=device)

        cost_fn = lambda eps_seq: get_expected_cost_for_cem(s_batch, eps_seq, goal_xy_t, prior_type, length_cost=plan_length_cost)

        if random.random() < 0.5:
            eps_mean = torch.zeros((skill_seq_len, z_dim), device=device)
            eps_std  = torch.ones((skill_seq_len, z_dim), device=device)
        else:
            if last_eps_mean:
                eps_mean = last_eps_mean
                eps_std = last_eps_std
            else:
                eps_mean = torch.zeros((skill_seq_len, z_dim), device=device)
                eps_std  = torch.ones((skill_seq_len, z_dim), device=device)
                 
                 

        eps_mean, eps_std = cem(eps_mean, eps_std, cost_fn,pop_size=batch_size, frac_keep=keep_frac, n_iters=n_iters,l2_pen=cem_l2_pen)
        if first_eps_mean == None:
             first_eps_mean = eps_mean

        last_eps_mean = eps_mean.detach().clone()
        last_eps_std = eps_std.detach().clone()

        # starting from this state, tawm dist for following 10 skills
        plan_means_xy, plan_stds_xy = tawm_plan_xy(state_vec, eps_mean, prior_type, n_std = 1)

        # call run_skill_seq
        eps_exec = eps_mean[:execute_n_skills]  # (execute_n_skills, Z)

        state_vec, executed_xy, done, per_skill_exec_xy = run_skill_seq(env,state_vec,eps_exec,prior_type,use_epsilon=use_epsilon,H=H,goal_xy=goal_xy,goal_thresh2=goal_thresh2,deterministic=deterministic,executed_xy=executed_xy)

        executed_skill_xy = per_skill_exec_xy[0]

        print(f"replan {repl}] xy={state_vec[-2:]} dist to the goal={np.sum((state_vec[-2:] - goal_xy)**2):.3f}")

        nearby_idx = pick_nearby_ep(episodes_start_xy, cur_xy)
        nearby_trajs = [episodes_xy[i] for i in nearby_idx]

        outpath = os.path.join(PLANS_DIR, f"replan_{repl:05d}.png")
        save_replan_figure(outpath, all_xy, cur_xy, goal_xy, plan_means_xy, plan_stds_xy, executed_xy, executed_skill_xy, nearby_trajs, title=f"replan {repl}")
        # save_replan_figure(outpath, all_xy, cur_xy, goal_xy, plan_means_xy, plan_stds_xy, executed_xy, executed_skill_xy, title=f"replan {repl}")

        if done:
            break

    save_final_trajectory(os.path.join(OUTDIR, f"final_executed_trajectory.png"), executed_xy, goal_xy, all_s0)
    return np.stack(executed_xy, axis=0), goal_xy, last_s0_vec, last_eps_mean, first_state_vec, first_eps_mean, all_s0


def run_skill_seq(env, state_vec, eps_seq, prior_type, use_epsilon=True, H=40,
                  goal_xy=None, goal_thresh2=1.0, deterministic=False,
                  executed_xy=None):
    """
    Executes a skill seq using the policy and returns the new state (and xy-locations).
    """
    if executed_xy is None:
        executed_xy = [state_vec[-2:].copy()]

    L = eps_seq.shape[0]
    done = False
    per_skill_exec_xy = []

    for i in range(L):
        # choose z for this skill
        s_t = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)  # [1,sd]
        if use_epsilon:
            if prior_type == "uni":
                mu_z, sigma_z = skillprior(s_t)                         
                z = mu_z + sigma_z * eps_seq[i:i+1, :]  
            else:
                mu_z, sigma_z = mog_mean_std(skillprior, s_t)
                z = mu_z + sigma_z * eps_seq[i:i+1, :]

        skill_xy = []
        # execute low-level steps for H 
        for t in range(H):
            a = policy_action(llpolicy, state_vec, z, deterministic=deterministic)  
            obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            state_vec = obs_to_state_vec(obs)
            xy = state_vec[-2:].copy()
            executed_xy.append(xy)
            skill_xy.append(xy)

            if goal_xy is not None:
                if np.sum((state_vec[-2:] - goal_xy) ** 2) < goal_thresh2:
                    per_skill_exec_xy.append(np.asarray(skill_xy, dtype=np.float32))
                    return state_vec, executed_xy, True, per_skill_exec_xy

            if done:
                per_skill_exec_xy.append(np.asarray(skill_xy, dtype=np.float32))
                return state_vec, executed_xy, True, per_skill_exec_xy
        per_skill_exec_xy.append(np.asarray(skill_xy, np.float32))
    return state_vec, executed_xy, False, per_skill_exec_xy



@torch.no_grad()
def tawm_plan_xy(s0_vec_np, eps_plan, prior_type, n_std=2.0):
    """
    Takes a skill sequence plan and plots the distributions by conditioning on each skill and previous state. 
    """
    s = torch.tensor(s0_vec_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1,sd]

    means_xy = [s[0, -2:].cpu().numpy().copy()]
    stds_xy  = [np.zeros(2, dtype=np.float32)]

    L = eps_plan.shape[0]
    for i in range(L):
        if prior_type == "uni":
            mu_z, sigma_z = skillprior(s)              
            z = mu_z + sigma_z * eps_plan[i:i+1, :] 
        else:
            mu_z, sigma_z = mog_mean_std(skillprior, s)
            z = mu_z + sigma_z * eps_plan[i:i+1, :]  

        s_mean, s_std = tawm(s, z)                
        s = s_mean                                 

        mean_xy = s_mean[0, -2:].cpu().numpy()
        std_xy  = s_std[0, -2:].cpu().numpy()     

        means_xy.append(mean_xy.copy())
        stds_xy.append(std_xy.copy())

    return np.stack(means_xy, axis=0), np.stack(stds_xy, axis=0)


def add_green_blob(ax, xy, std_xy, n_std=2.0, alpha=0.20):
    """
    Draw a green ellipse blob.
    """
    ell = Ellipse(xy=(xy[0], xy[1]),width=2.0 * n_std * std_xy[0],height=2.0 * n_std * std_xy[1],angle=0.0,facecolor="green",edgecolor="green",alpha=alpha,linewidth=1.5,zorder=2,)
    ax.add_patch(ell)



def plot_plan_blobs_vs_exec_with_bg(all_xy,
    executed_xy, plan_means_xy, plan_stds_xy, goal_xy, outpath="plan_blobs_vs_exec_bg.png",
    n_std=2.0
):
    exec_xy = np.asarray(executed_xy, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(6.8, 6.4))

    # ax.imshow(
    #     bg_img.T,                 
    #     extent=bg_extent,
    #     origin="lower",
    #     alpha=0.35,
    #     aspect="equal",
    # )

    # occ_nonzero = occ[occ > 0]
    # if occ_nonzero.size > 0:
    #     thr = np.percentile(occ_nonzero, 10)  # tune: 5–20 works well
    #     X, Y = np.meshgrid(x_centers, y_centers, indexing="xy")
    #     ax.contour(X, Y, occ.T, levels=[thr], linewidths=1.2)
    ax.scatter(all_xy[:, 0], all_xy[:, 1],
               s=2, alpha=0.15, color="lightgray",
               label="dataset states", zorder=1)

    if len(exec_xy) > 0:
        ax.plot(exec_xy[:, 0], exec_xy[:, 1], linewidth=2.5, label="executed", zorder=3)
        ax.scatter(exec_xy[0, 0], exec_xy[0, 1], s=60, zorder=4)

    ax.plot(plan_means_xy[:, 0], plan_means_xy[:, 1], linestyle="--",
            linewidth=2.0, label="TAWM plan mean", zorder=2)

    for i in range(1, len(plan_means_xy)):
        add_green_blob(ax, plan_means_xy[i], plan_stds_xy[i], n_std=n_std, alpha=0.18)

    ax.scatter(goal_xy[0], goal_xy[1], s=140, marker="*", color="black", label="goal", zorder=5)

    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"saved -> {outpath}")

def save_replan_figure(
    outpath,
    all_xy,
    current_xy,
    goal_xy,
    plan_means_xy,
    plan_stds_xy,
    executed_xy_so_far,
    executed_skill_xy,               
    nearby_episode_xys,              
    title=None,
):
    exec_xy = np.asarray(executed_xy_so_far, dtype=np.float32)
    skill_xy = np.asarray(executed_skill_xy, dtype=np.float32) if executed_skill_xy is not None else None

    fig, ax = plt.subplots(figsize=(7.2, 6.6))

    # Background scatter (light)
    if all_xy is not None and all_xy.shape[0] > 0:
        ax.scatter(all_xy[:, 0], all_xy[:, 1], s=2, alpha=0.10, color="lightgray", zorder=1)

    # # Nearby dataset subtrajectories (thin)
    for traj in nearby_episode_xys:
        if traj.shape[0] > 1:
            ax.plot(traj[:, 0], traj[:, 1], linewidth=1.0, alpha=0.25, zorder=2)

    # Just-executed skill (red)
    if skill_xy is not None and skill_xy.shape[0] > 0:
        ax.plot(skill_xy[:, 0], skill_xy[:, 1], color="red", linewidth=3.0, alpha=0.95, label="executed skill", zorder=6)

    # Current position
    ax.scatter(current_xy[0], current_xy[1], s=80, marker="o", color="blue", label="current", zorder=7)

    # Plan mean + blobs
    ax.plot(plan_means_xy[:, 0], plan_means_xy[:, 1], linestyle="--", color="green",
            linewidth=2.0, label="TAWM plan mean", zorder=5)
    for i in range(1, len(plan_means_xy)):
        add_green_blob(ax, plan_means_xy[i], plan_stds_xy[i], n_std=1, alpha=0.18)

    # Goal
    ax.scatter(goal_xy[0], goal_xy[1], s=160, marker="*", color="black", label="goal", zorder=8)

    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.25)
    if title is not None:
        ax.set_title(title)

    ax.legend(loc="best")

    # Bounds
    pts = [current_xy.reshape(1,2), goal_xy.reshape(1,2), plan_means_xy]
    if exec_xy.shape[0] > 0:
        pts.append(exec_xy)
    if skill_xy is not None and skill_xy.shape[0] > 0:
        pts.append(skill_xy)
    pts = np.vstack(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.10 * (hi - lo + 1e-6)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])

    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close(fig)
    print(f"saved -> {outpath}")

def save_final_trajectory(outpath, executed_xy, goal_xy, all_s0):
    exec_xy = np.asarray(executed_xy, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7.0, 6.5))

    if exec_xy.shape[0] > 1:
        ax.plot(exec_xy[:, 0], exec_xy[:, 1], color="red", linewidth=3.0, zorder=2, label="executed")
        ax.scatter(exec_xy[0, 0], exec_xy[0, 1], s=60, color="red", zorder=3)

    # ax.scatter(all_s0[:, 0], all_s0[:, 1], s=170, marker="*", color="green", zorder=4, label="skill start")

    ax.scatter(goal_xy[0], goal_xy[1], s=170, marker="*", color="black", zorder=4, label="goal")

    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    pts = np.vstack([exec_xy, goal_xy.reshape(1,2)]) if exec_xy.size else goal_xy.reshape(1,2)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.10 * (hi - lo + 1e-6)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])

    plt.tight_layout()
    plt.savefig(outpath, dpi=240)
    plt.close(fig)
    print(f"saved -> {outpath}")


exec_xy, goal_xy, last_s0_vec, last_eps_mean, first_s0_vec, first_eps_mean, all_s0 = run_skills_iterative_replanning(env,skill_seq_len=skill_seq_len,H=H,execute_n_skills=1,max_replans=max_replans,use_epsilon=True,goal_thresh2=1.0,deterministic=False)

env.close()

# if last_eps_mean is not None and last_s0_vec is not None:
#     ant_maze_dataset = minari.load_dataset('D4RL/antmaze/medium-diverse-v1')

#     planned_means_xy, planned_stds_xy = taww_plan_xy(first_s0_vec, first_eps_mean)

#     all_xy = []
#     for ep in ant_maze_dataset.iterate_episodes():
#         xy = ep.observations["achieved_goal"][:, :2]  
#         all_xy.append(xy)

#     all_xy = np.concatenate(all_xy, axis=0)
#     plot_plan_blobs_vs_exec_with_bg(
#         all_xy, exec_xy, planned_means_xy, planned_stds_xy, goal_xy ,outpath="plan_blobs_vs_exec_bg.png",
#         n_std=2.0
#     )

