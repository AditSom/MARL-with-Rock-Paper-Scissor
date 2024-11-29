import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Neural Network for the Q-value function
class DQN(nn.Module): #TODO: Make variable size DQN 
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, env,config):
        self.env = env
        self.config = config
        self.input_dim = env.total_agents * 3
        self.q_network = DQN(input_dim=self.input_dim, output_dim=config.n_actions)
        self.target_network = DQN(input_dim=self.input_dim, output_dim=config.n_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.batch_size = 1
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.config.n_actions-1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state)
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_q_network(self,input):
        states,actions,rewards,next_states,dones = input
        states = torch.FloatTensor(states).unsqueeze(0)
        actions = torch.LongTensor([actions])
        rewards = torch.FloatTensor([rewards])
        next_states = torch.FloatTensor(next_states).unsqueeze(0)
        dones = torch.FloatTensor([dones])
        current_q_values = self.q_network(states)
        current_q_values = current_q_values[0][actions]
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)