from math import inf
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import TanhTransform
from torch.optim import Adam
import numpy as np


class Policy(nn.Module):
    def __init__(
        self,
        latent_size: int,
        action_size: int,
        hidden_size: int,
        min_std: float = 1e-4,
        init_std: float = 5,
        mean_scale: float = 5,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, action_size * 2),
        )
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.raw_init_std = np.log(np.exp(self.init_std) - 1)

    def forward(self, latent_state):
        model_out = self.model(latent_state)
        mean, std = torch.chunk(model_out, 2, -1)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        std = F.softplus(std + self.raw_init_std) + self.min_std
        dist = torch.distributions.Normal(mean, std)
        dist = torch.distributions.TransformedDistribution(dist, TanhTransform())
        dist = torch.distributions.Independent(dist, 1)
        return dist

class DreamerAgent(object):

    def __init__(self,
        action_size: int,
        belief_size: int,
        state_size: int,
        hidden_size: int,
        planning_horizon: int,
        policy_lr: float,
        critic_lr: float,
        transition_model,
        reward_model,
        min_action: -inf,
        max_action: inf,
        device: torch.device,
    ):
        super().__init__()
        self.latent_size = belief_size + state_size
        self.action_size = action_size
        self.planning_horizon = planning_horizon
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.min_action = min_action
        self.max_action = max_action

        self.policy = Policy(
            self.latent_size,
            action_size,
            hidden_size,
        ).to(device)
        self.policy_optim = Adam(self.policy.parameters(), policy_lr)
        self.critic = nn.Sequential(
            nn.Linear(self.latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
        ).to(device)
        self.critic_optim = Adam(self.critic.parameters(), critic_lr)
    
    def act(self, belief, state, training: bool = True):
        latent = torch.cat([belief, state], dim=1)
        action_dist = self.policy(latent)
        if training:
            action = action_dist.rsample()
        else:
            action = action_dist.mode()
        return action
    
    def train(self):
        policy_loss = torch.tensor(0)
        value_loss = torch.tensor(0)
        return policy_loss, value_loss