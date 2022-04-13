from math import inf
from typing import Union

import numpy as np
import torch
from torch import jit, nn
from torch.distributions import TanhTransform, Independent, Normal, TransformedDistribution
from torch.nn import functional as F
from torch.optim import Adam

from models import bottle


def freeze(module: Union[nn.Module, jit.ScriptModule]):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze(module: Union[nn.Module, jit.ScriptModule]):
    for p in module.parameters():
        p.requires_grad = True


class PolicyModel(nn.Module):
    def __init__(
        self,
        latent_size: int,
        action_size: int,
        hidden_size: int,
        min_std: float = 1e-4,
        init_std: float = 5,
        mean_scale: float = 5,
        activation_function="elu",
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size * 2)
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.raw_init_std = np.log(np.exp(self.init_std) - 1)

    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        model_out = self.fc3(hidden).squeeze(dim=1)
        mean, std = torch.chunk(model_out, 2, -1)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        std = F.softplus(std + self.raw_init_std) + self.min_std
        dist = Normal(mean, std)
        dist = TransformedDistribution(dist, TanhTransform())
        dist = Independent(dist, 1)
        return dist


class ValueModel(nn.Module):
    def __init__(self, latent_size, hidden_size, activation_function="elu"):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        value = self.fc3(hidden).squeeze(dim=1)
        return value


class DreamerAgent(object):
    def __init__(
        self,
        action_size: int,
        belief_size: int,
        state_size: int,
        hidden_size: int,
        horizon: int,
        policy_lr: float,
        critic_lr: float,
        gamma: float,
        lam: float,
        grad_clip_norm: float,
        activation_function: str,
        transition_model,
        reward_model,
        min_action: -inf,
        max_action: inf,
        device: torch.device,
    ):
        super().__init__()
        self.latent_size = belief_size + state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lam = lam
        self.grad_clip_norm = grad_clip_norm
        self.horizon = horizon
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.min_action = min_action
        self.max_action = max_action
        self.device = device

        self.policy_model = PolicyModel(
            self.latent_size,
            action_size,
            hidden_size,
            activation_function=activation_function,
        ).to(device)
        self.policy_optim = Adam(self.policy_model.parameters(), policy_lr)
        self.value_model = ValueModel(
            self.latent_size, hidden_size, activation_function
        ).to(device)
        self.value_optim = Adam(self.value_model.parameters(), critic_lr)

    def act(self, belief, state, training: bool = True):
        action_dist = self.policy_model(belief, state)
        if training:
            action = action_dist.rsample()
        else:
            action = action_dist.mode()
        return action

    def train(self, beliefs, states):
        # beliefs: BxLxH
        # states: BxLxZ
        freeze(self.transition_model)
        freeze(self.reward_model)
        B, L, _ = beliefs.shape
        beliefs = torch.reshape(beliefs, [B * L, -1])
        states = torch.reshape(states, [B * L, -1])
        imag_beliefs = []
        imag_states = []
        imag_actions = []
        for h in range(self.horizon):
            actions = self.act(beliefs, states)
            imag_beliefs.append(beliefs)
            imag_states.append(states)
            imag_actions.append(actions)
            beliefs, states, _, _ = self.transition_model(
                states, actions.unsqueeze(dim=0), beliefs
            )
            beliefs = beliefs.squeeze(dim=0)
            states = states.squeeze(dim=0)

        # I x (B*L) x _
        imag_beliefs = torch.stack(imag_beliefs).to(self.device)
        imag_states = torch.stack(imag_states).to(self.device)
        imag_actions = torch.stack(imag_actions).to(self.device)
        freeze(self.value_model)
        imag_rewards = bottle(self.reward_model, (imag_beliefs, imag_states))
        imag_values = bottle(self.value_model, (imag_beliefs, imag_states))
        unfreeze(self.value_model)

        # Compute returns: the hard way
        # returns = torch.zeros([self.horizon, B*L, self.horizon + 1]).to(self.device)
        # for tau in range(self.horizon):
        #     for idx in range(B*L):
        #         for k in range(1, self.horizon + 1):
        #             h = min(tau + k, self.horizon)
        #             for n in range(h - 1, tau - 1, -1):
        #                 returns[tau, idx, k] += self.gamma ** (n - tau) * imag_rewards[n - 1, idx]
        #             returns[tau, idx, k] += self.gamma ** (h - tau) * imag_values[h - 1, idx]

        #         for n in range(1, self.horizon):
        #             returns[tau, idx, 0] += self.lam ** (n - 1) * returns[tau, idx, n]
        #         returns[tau, idx, 0] *= (1 - self.lam)
        #         returns[tau, idx, 0] += returns[tau, idx, self.horizon]
        # policy_loss = -torch.mean(returns[:, :, 0])

        # Compute returns: the DP way, from https://github.com/juliusfrost/dreamer-pytorch
        discount_arr = self.gamma * torch.ones_like(imag_rewards)
        returns = self.compute_return(
            imag_rewards[:-1],
            imag_values[:-1],
            discount_arr[:-1],
            bootstrap=imag_values[-1],
            lambda_=self.lam,
        )
        # Make the top row 1 so the cumulative product starts with discount^0
        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        policy_loss = -torch.mean(discount * returns)

        # Detach tensors which have gradients through policy model for value loss
        value_beliefs = imag_beliefs.detach()[:-1]
        value_states = imag_states.detach()[:-1]
        value_discount = discount.detach()
        value_target = returns.detach()
        value_pred = bottle(self.value_model, (value_beliefs, value_states))
        value_loss = F.mse_loss(value_discount * value_target, value_pred)

        self.policy_optim.zero_grad()
        self.value_optim.zero_grad()

        nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_clip_norm)
        nn.utils.clip_grad_norm_(self.value_model.parameters(), self.grad_clip_norm)

        policy_loss.backward()
        value_loss.backward()

        self.policy_optim.step()
        self.value_optim.step()

        unfreeze(self.reward_model)
        unfreeze(self.transition_model)

        return policy_loss, value_loss

    def compute_return(
        self,
        reward: torch.Tensor,
        value: torch.Tensor,
        discount: torch.Tensor,
        bootstrap: torch.Tensor,
        lambda_: float,
    ):
        """
        Compute the discounted reward for a batch of data.
        reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
        Bootstrap is [batch, 1]
        """
        next_values = torch.cat([value[1:], bootstrap[None]], 0)
        target = reward + discount * next_values * (1 - lambda_)
        timesteps = list(range(reward.shape[0] - 1, -1, -1))
        outputs = []
        accumulated_reward = bootstrap
        for t in timesteps:
            inp = target[t]
            discount_factor = discount[t]
            accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
            outputs.append(accumulated_reward)
        returns = torch.flip(torch.stack(outputs), [0])
        return returns
