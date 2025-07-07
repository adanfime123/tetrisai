# dqn_tetris.py
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from tetris_env import TetrisEnv

# Hyperparameters
EPISODES = 1000
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
MEMORY_SIZE = 10000

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_shape[0] * input_shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay Memory
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_, d):
        self.buffer.append((s, a, r, s_, d))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.stack(states), actions, rewards, np.stack(next_states), dones)

    def __len__(self):
        return len(self.buffer)

# Training function
def train():
    env = TetrisEnv()
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(input_shape, n_actions).to(device)
    target_net = DQN(input_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPSILON_START
    total_steps = 0

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            # Learn
            if len(memory) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards).unsqueeze(1).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

                current_q = policy_net(states).gather(1, actions)
                max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards + GAMMA * max_next_q * (1 - dones)

                loss = F.mse_loss(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_steps += 1

        # Update target network
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {epsilon:.3f}")
        if episode % 100 == 0:
            torch.save(policy_net.state_dict(), f"model_ep{episode}.pth")

if __name__ == "__main__":
    train()
    