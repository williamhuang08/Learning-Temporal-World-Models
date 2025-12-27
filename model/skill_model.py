import torch
import torch.nn as nn
import torch.nn.functional as F

# Each layer contains 256 neurons
NUM_NEURONS = 256
# The dimension of the abstract skill variable, z
Z_DIM = 256

# Skill Posterior, q_phi
class SkillPosterior(nn.Module):
    def __init__(self, state_dim, action_dim, h_dim=NUM_NEURONS, n_gru_layers=4):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_emb = nn.Sequential(
            nn.Linear(state_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

        self.bi_gru = nn.GRU(
            input_size=h_dim + action_dim,
            hidden_size=h_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=n_gru_layers
        )

        self.mean = MeanNetwork(in_dim=2*h_dim, out_dim=Z_DIM)
        self.std  = StandardDeviationNetwork(in_dim=2*h_dim, out_dim=Z_DIM)


    def forward(self, state_sequence, action_sequence):
        # state_sequence: [B, T, state_dim]
        s_emb = self.state_emb(state_sequence)                 
        x_in  = torch.cat([s_emb, action_sequence], dim=-1)   
        feats, _ = self.bi_gru(x_in)                          
        seq_emb = feats[:, -1, :] # *** use last time step, not mean ***
        mean = self.mean(seq_emb)
        std  = self.std(seq_emb)
        return mean, std


# Low-Level Skill-Conditioned Policy, pi_theta
class SkillPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, h_dim=NUM_NEURONS, a_dist='normal', max_sig=None, fixed_sig=None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.a_dist = a_dist
        self.max_sig = max_sig
        self.fixed_sig = fixed_sig

        self.layers = nn.Sequential(
            nn.Linear(state_dim + Z_DIM, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, action_dim)
        )
        self.sig_head  = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, action_dim)
        )

    def forward(self, state, z):
        # state: [B*T, state_dim], z: [B*T, Z_DIM]
        x = torch.cat([state, z], dim=-1)
        feats = self.layers(x)
        mean  = self.mean_head(feats)
        if self.max_sig is None:
            sig = F.softplus(self.sig_head(feats))
        else:
            sig = self.max_sig * torch.sigmoid(self.sig_head(feats))
        if self.fixed_sig is not None:
            sig = self.fixed_sig * torch.ones_like(sig)
        return mean, sig

        

# Temporally-Abstract World Model, p_psi
class TAWM(nn.Module):
    """
    Input: initial state, along with the abstract skill
    Output: mean and std over terminal state

    1. 2-layer shared network w/ ReLU activations for initial state and abstract skill (concatenated)
    2. Extract mean and std of layer 1's output
    """
    def __init__(self, state_dim, h_dim=NUM_NEURONS, per_element_sigma=True):
        super().__init__()
        self.state_dim = state_dim
        self.per_element_sigma = per_element_sigma

        self.layers = nn.Sequential(
            nn.Linear(state_dim + Z_DIM, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, state_dim)
        )
        if per_element_sigma:
            self.sig_head = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, state_dim),
                nn.Softplus()
            )
        else:
            self.sig_head = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, 1),
                nn.Softplus()
            )

    def forward(self, s0, z):
        # s0: [B, state_dim], z: [B, Z_DIM]
        x = torch.cat([s0, z], dim=-1)
        feats = self.layers(x)
        mean  = self.mean_head(feats)
        sig   = self.sig_head(feats)
        if not self.per_element_sigma:
            sig = sig.expand(-1, self.state_dim)
        return mean, sig


# Skill Prior, p_omega
class SkillPrior(nn.Module):
    """
    Input: Initial state, s0, in the trajectory
    Output: mean and std over the abstract skill, z

    1. 2-layer shared network w/ ReLU activations for the initial state
    2. Extract mean and std of layer 1's output
    """
    def __init__(self, state_dim, h_dim=NUM_NEURONS):
        super().__init__()
        self.state_dim = state_dim
        self.layers = nn.Sequential(
            nn.Linear(state_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, Z_DIM)
        )
        self.sig_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, Z_DIM),
            nn.Softplus()
        )

    def forward(self, s0):
        feats = self.layers(s0)
        mean = self.mean_head(feats)
        std  = self.sig_head(feats)
        return mean, std


class MeanNetwork(nn.Module):
    """
    Input: tensor to calculate mean
    Output: mean of input w/ dimension out_dim

    1. 2-layer network w/ ReLU activation for the first layer
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(in_dim, NUM_NEURONS)
        self.fc2 = nn.Linear(NUM_NEURONS, out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        
        
class StandardDeviationNetwork(nn.Module):
    """
    Input: tensor to calculate std
    Output: std of input w/ dimension out_dim

    Note: the standard deviation is lower and upper bounded at 0.05 and 2.0
    - if std is 0, then log(std) -> inf
    - if std is large, then can affect training

    1. 2-layer linear network with ReLU activation after first layer and softplus after second

    """
    def __init__(self, in_dim, out_dim, min_std=0.05, max_std=5.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, NUM_NEURONS)
        self.fc2 = nn.Linear(NUM_NEURONS, out_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.min_std = min_std
        self.max_std = max_std
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        std = self.softplus(x) 
        #+ self.min_std  # lower bound
        #std = torch.clamp(std, max=self.max_std)
        return std
