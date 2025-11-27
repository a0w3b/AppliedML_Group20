"""UNI-Vaasa, Applied ML / Group20 Project work: RL-Mario — Agent script."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- DQN Model ---
class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(self.device),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).to(self.device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device),
            torch.tensor(dones, dtype=torch.float32).to(self.device)
        )



    def __len__(self):
        return len(self.buffer)

# --- Epsilon-Greedy Action Selection ---
def select_action(state, policy_net, epsilon, env, device, encourage_high_jump=False):
    """
    Select action using epsilon-greedy policy.
    
    Args:
        state: Current game state
        policy_net: Policy network
        epsilon: Exploration rate
        env: Environment
        device: Device (cpu/cuda)
        encourage_high_jump: If True, bias exploration towards high jump (action 4)
    """
    if random.random() < epsilon:
        # During exploration, bias towards high jump action if encouraged
        if encourage_high_jump:
            # When stuck or exploring, try high jump more often
            if random.random() < 0.75:  # 75% chance to try high jump (increase exploration bias)
                return 4  # High jump action (right + A + B)
            # Also try normal jump with remaining probability
            elif random.random() < 0.5:  # 50% chance for normal jump if not high-jump
                return 2  # Normal jump action (right + A)
        # Even when not explicitly encouraged, sometimes try high jump
        elif random.random() < 0.25:  # 25% chance to try high jump during normal exploration
            return 4
        return env.action_space.sample()
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = policy_net(state_tensor)
        return q_values.argmax().item()
