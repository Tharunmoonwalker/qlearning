import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random

# ==== Hyperparameters ====
EPISODES = 100
GAMMA = 0.99
LR = 1e-4

# ==== Environment ====
env = gym.make("CarRacing-v3", render_mode="human", continuous=True)
obs_shape = env.observation_space.shape
action_shape = env.action_space.shape[0]

# ==== Policy Network ====
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, action_shape),
            nn.Tanh()  # for [-1,1] range
        )

    def forward(self, x):
        x = torch.FloatTensor(x).permute(2, 0, 1).unsqueeze(0) / 255.0
        x = self.conv(x)
        x = x.view(1, -1)
        return self.fc(x)[0]

policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=LR)

# ==== REINFORCE buffers ====
log_probs = []
rewards = []

def select_action(state):
    action = policy(state).detach().numpy()
    # scale outputs: steer [-1,1], gas [0,1], brake [0,1]
    action[1] = np.clip((action[1] + 1) / 2, 0, 1)  # gas
    action[2] = np.clip((action[2] + 1) / 2, 0, 1)  # brake
    dist = torch.distributions.Normal(torch.tensor(action), torch.tensor([0.2, 0.2, 0.2]))
    sampled = dist.sample()
    log_prob = dist.log_prob(sampled).sum()
    log_probs.append(log_prob)
    return sampled.numpy()

def update_policy():
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + GAMMA * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    loss = 0
    for log_prob, R in zip(log_probs, returns):
        loss -= log_prob * R

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # clear memory
    del log_probs[:]
    del rewards[:]

# ==== Training Loop ====
for ep in range(EPISODES):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step = 0

    print(f"\n🚗 Episode {ep + 1}/{EPISODES} starting...")
    
    while not done:
        env.render()
        time.sleep(0.01)

        action = select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        total_reward += reward
        done = terminated or truncated
        step += 1

    update_policy()
    print(f"✅ Episode {ep + 1} finished. Total reward: {int(total_reward)}, steps: {step}")

env.close()
