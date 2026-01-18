'''File where we will sample a set of waypoints, and plan a sequence of skills to have our pointmass travel through those waypoints'''

import minari
from model.skill_model import SkillPolicy, SkillPosterior, SkillPrior, TAWM
from model.utils import load_checkpoint, pack_state_from_obs, read_antmaze_obs
from utils import obs_to_state_vec, xy_from_state
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from cem import cem, cem_variable_length

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load in the environment
env_name = 'D4RL/antmaze/medium-diverse-v1'
data = minari.load_dataset(env_name)
env = data.recover_environment()

def strip_timelimit(env):
    while hasattr(env, "env") and env.__class__.__name__ == "TimeLimit":
        env = env.env
    return env

env = strip_timelimit(env)
env = TimeLimit(env, max_episode_steps=4000)

# Environment variables
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
max_replans = 40000 // H
plan_length_cost = 0.0
encoder_type = 'state_action_sequence'
term_state_dependent_prior = False
init_state_dependent = True
random_goal = False # determines if we select a goal at random from dataset (random_goal=True) or use pre-set one from environment

filename = 'antmaze_diverse_detached_250_1.pth'
PATH = '../checkpoints/' + filename

skillpost = SkillPosterior(state_dim=state_dim, action_dim=a_dim).to(device)
llpolicy = SkillPolicy(state_dim=state_dim, action_dim=a_dim).to(device)
tawm = TAWM(state_dim=state_dim).to(device)
skillprior = SkillPrior(state_dim=state_dim).to(device)
_ = load_checkpoint(PATH, skillpost, llpolicy, tawm, skillprior)

env.reset() # reset the env config start/end states
obs = read_antmaze_obs(env)
_, ag0, s0 = pack_state_from_obs(obs) # convert obs to a state
s0_torch = torch.cat([torch.tensor(s0,dtype=torch.float32).to(device=device).reshape(1,1,-1) for _ in range(batch_size)])
# [batch_size, 1, state_dim]: want to sample batch_size candidate skill seqs and score each skill seq starting from s0

skill_seq = torch.zeros((1,skill_seq_len,z_dim),device=device) # [1, skill_seq_len, z_dim]
print('skill_seq.shape: ', skill_seq.shape)
skill_seq.requires_grad = True

@torch.no_grad()
def policy_action(llpolicy, state_vec, z_vec, deterministic=True):
    """
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

def convert_epsilon_to_z(epsilon, s0_vec):
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
        mu_z, sigma_z = skillprior(s) # [B,Z]
        eps_i = epsilon[:, i, :] # [B,Z]
        z_i = mu_z + sigma_z * eps_i # [B,Z]
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

def get_expected_cost_for_cem(s0, eps_seq, goal_xy, length_cost=0.0):
    """
    Returns the cost of the eps_seq which is the minimum distance along the whole skill sequence to the goal.

    s0: [B,1,sd] 
    eps_seq: [B,L,Z] 
    goal_xy: [2] 
    returns: [B] costs or best cost [1]
    """
    s = s0.squeeze(1)    

    goal_xy = goal_xy.view(1, 2).expand(s.shape[0], -1)  # [B,2]

    B, L, _ = eps_seq.shape

    costs = []
    # cost at t=0
    costs.append(((s[:, -2:] - goal_xy) ** 2).mean(dim=-1))

    for i in range(L):
        mu_z, sigma_z = skillprior(s)       
        eps_i = eps_seq[:, i, :]             
        z_i = mu_z + sigma_z * eps_i          

        s, _ = tawm(s, z_i)                   
        costs.append(((s[:, -2:] - goal_xy) ** 2).mean(dim=-1) + (i+1)*length_cost)

    costs = torch.stack(costs, dim=1) # [B,L+1]
    best, _ = torch.min(costs, dim=1)  # [B]
    return best

def run_skills_iterative_replanning(env,
    skill_seq_len=10,
    H=40,
    execute_n_skills=1,   
    max_replans=2000//40,
    use_epsilon=True,
    goal_thresh2=1.0,
    deterministic=True
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
    last_s0_vec = None

    for repl in range(max_replans):
        # stop if already at goal
        if np.sum((state_vec[-2:] - goal_xy) ** 2) < goal_thresh2:
            print("Reached goal (before planning).")
            break

        # CEM
        last_s0_vec = state_vec.copy()
        s_batch = torch.tensor(state_vec, dtype=torch.float32, device=device).view(1,1,-1).expand(batch_size, 1, state_dim)   # [B,1,sd]
        goal_xy_t = torch.tensor(goal_xy, dtype=torch.float32, device=device)

        cost_fn = lambda eps_seq: get_expected_cost_for_cem(s_batch, eps_seq, goal_xy_t,length_cost=plan_length_cost)

        eps_mean = torch.zeros((skill_seq_len, z_dim), device=device)
        eps_std  = torch.ones((skill_seq_len, z_dim), device=device)

        eps_mean, eps_std = cem(eps_mean, eps_std, cost_fn,pop_size=batch_size, frac_keep=keep_frac, n_iters=n_iters,l2_pen=cem_l2_pen)
        if first_eps_mean == None:
             first_eps_mean = eps_mean

        last_eps_mean = eps_mean.detach().clone()

        # call run_skill_seq
        eps_exec = eps_mean[:execute_n_skills]  # (execute_n_skills, Z)

        state_vec, executed_xy, done = run_skill_seq(env,state_vec,eps_exec,use_epsilon=use_epsilon,H=H,goal_xy=goal_xy,goal_thresh2=goal_thresh2,deterministic=deterministic,executed_xy=executed_xy)

        print(f"replan {repl}] xy={state_vec[-2:]} dist to the goal={np.sum((state_vec[-2:] - goal_xy)**2):.3f}")

        if done:
            break

    return np.stack(executed_xy, axis=0), goal_xy, last_s0_vec, last_eps_mean, first_state_vec, first_eps_mean


def run_skill_seq(env, state_vec, eps_seq, use_epsilon=True, H=40,
                  goal_xy=None, goal_thresh2=1.0, deterministic=True,
                  executed_xy=None):
    """
    Executes a skill seq using the policy and returns the new state (and xy-location).
    """
    if executed_xy is None:
        executed_xy = [state_vec[-2:].copy()]

    L = eps_seq.shape[0]
    done = False

    for i in range(L):
        # choose z for this skill
        s_t = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)  # [1,sd]
        if use_epsilon:
            mu_z, sigma_z = skillprior(s_t)                         
            z = mu_z + sigma_z * eps_seq[i:i+1, :]  

        # execute low-level steps for H 
        for t in range(H):
            a = policy_action(llpolicy, state_vec, z, deterministic=deterministic)  
            obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            state_vec = obs_to_state_vec(obs)
            executed_xy.append(state_vec[-2:].copy())

            if goal_xy is not None:
                if np.sum((state_vec[-2:] - goal_xy) ** 2) < goal_thresh2:
                    return state_vec, executed_xy, True

            if done:
                return state_vec, executed_xy, True

    return state_vec, executed_xy, False



@torch.no_grad()
def taww_plan_xy(s0_vec_np, eps_plan, n_std=2.0):
    """
    Takes a skill sequence plan and plots the distributions by conditioning on each skill and previous state. 
    """
    s = torch.tensor(s0_vec_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1,sd]

    means_xy = [s[0, -2:].cpu().numpy().copy()]
    stds_xy  = [np.zeros(2, dtype=np.float32)]

    L = eps_plan.shape[0]
    for i in range(L):
        mu_z, sigma_z = skillprior(s)              
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


def plot_plan_blobs_vs_exec(env, executed_xy, plan_means_xy, plan_stds_xy, goal_xy, outpath="plan_blobs_vs_exec.png", n_std=2.0):
    """
    Plots executed trajectory, TAWM waypoints, and goal
    """
    exec_xy = np.asarray(executed_xy, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    fig, ax = plt.subplots(figsize=(6.5, 6))

    # executed trajectory in red
    if len(exec_xy) > 0:
        ax.plot(exec_xy[:, 0], exec_xy[:, 1], color="red", linewidth=2.5, label="executed", zorder=3)
        ax.scatter(exec_xy[0, 0], exec_xy[0, 1], color="red", s=60, zorder=4)

    # planned means in green
    ax.plot(plan_means_xy[:, 0], plan_means_xy[:, 1], linestyle="--", color="green",
            linewidth=2.0, label="TAWM plan mean", zorder=2)

    # green blobs along the plan
    for i in range(1, len(plan_means_xy)):  
        add_green_blob(ax, plan_means_xy[i], plan_stds_xy[i], n_std=n_std, alpha=0.18)

    # goal
    ax.scatter(goal_xy[0], goal_xy[1], s=140, marker="*", color="black", label="goal", zorder=5)

    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # bounds
    pts = np.vstack([exec_xy, plan_means_xy, goal_xy.reshape(1,2)]) if len(exec_xy) else np.vstack([plan_means_xy, goal_xy.reshape(1,2)])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.08 * (hi - lo + 1e-6)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"saved -> {outpath}")

exec_xy, goal_xy, last_s0_vec, last_eps_mean, first_s0_vec, first_eps_mean = run_skills_iterative_replanning(env,skill_seq_len=skill_seq_len,H=H,execute_n_skills=1,max_replans=max_replans,use_epsilon=True,goal_thresh2=1.0,deterministic=True)

# Optional: plot last plan vs executed
if last_eps_mean is not None and last_s0_vec is not None:
    planned_means_xy, planned_stds_xy = taww_plan_xy(first_s0_vec, first_eps_mean)
    plot_plan_blobs_vs_exec(env, exec_xy, planned_means_xy, planned_stds_xy, goal_xy,
                            outpath="plan_blobs_vs_exec.png", n_std=2.0)
