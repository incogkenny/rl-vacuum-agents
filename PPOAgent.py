import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        policy = self.actor(state)
        value = self.critic(state)
        return policy, value

class PPOAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=3e-4, clip_epsilon=0.2, batch_size=64, update_steps=4):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.update_steps = update_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.memory = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs, _ = self.model(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def store(self, transition):
        self.memory.append(transition)

    def compute_returns_and_advantages(self, rewards, values, dones, next_value):
        returns = []
        advantages = []
        gae = 0
        value = next_value
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * 0.95 * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            value = values[step]
            returns.insert(0, gae + values[step])
        return returns, advantages

    def update(self):
        states, actions, log_probs_old, rewards, dones, values = zip(*self.memory)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        values = torch.FloatTensor(values).to(self.device)

        with torch.no_grad():
            _, next_value = self.model(states[-1].unsqueeze(0))
        next_value = next_value.item()

        returns, advantages = self.compute_returns_and_advantages(rewards, values, dones, next_value)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        for _ in range(self.update_steps):
            idx = np.random.permutation(len(states))
            for i in range(0, len(states), self.batch_size):
                batch_idx = idx[i:i+self.batch_size]
                s_batch = states[batch_idx]
                a_batch = actions[batch_idx]
                logp_old_batch = log_probs_old[batch_idx]
                adv_batch = advantages[batch_idx]
                ret_batch = returns[batch_idx]

                probs, value = self.model(s_batch)
                dist = torch.distributions.Categorical(probs)
                logp = dist.log_prob(a_batch)
                ratio = torch.exp(logp - logp_old_batch)

                # Clipped surrogate loss
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(value.squeeze(), ret_batch)

                loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.memory = []

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
