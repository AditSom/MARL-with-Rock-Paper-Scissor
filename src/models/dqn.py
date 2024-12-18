import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Neural Network for the Q-value function
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super(DQN, self).__init__()
        # Initialize the first fully connected layer
        self.fc0 = nn.Linear(input_dim, config.hidden_dim[0])
        self.hidden_dim = config.hidden_dim
        # Initialize hidden layers based on the configuration
        for i in range(1, len(config.hidden_dim)):
            setattr(self, f'fc{i}', nn.Linear(config.hidden_dim[i-1], config.hidden_dim[i]))
        # Initialize the final fully connected layer
        self.fc_final = nn.Linear(config.hidden_dim[-1], output_dim)

    def forward(self, x):
        # Pass input through the first layer and apply ReLU activation
        x = torch.relu(self.fc0(x))
        # Pass through hidden layers with ReLU activation
        for i in range(1, len(self.hidden_dim)):
            x = torch.relu(getattr(self, f'fc{i}')(x))
        # Output layer
        return self.fc_final(x)

class Agent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        # Determine input dimension based on whether distance input is used
        input_dim_multiplier = 4 if config.distance_input else 3
        self.input_dim = env.total_agents * input_dim_multiplier
        # Initialize Q-network and target network
        self.q_network = DQN(input_dim=self.input_dim, output_dim=config.n_actions, config=config)
        self.target_network = DQN(input_dim=self.input_dim, output_dim=config.n_actions, config=config)
        # Copy weights from Q-network to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        # Set up optimizer and loss function
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()
        # Initialize replay memory
        self.memory = deque(maxlen=10000)
        # Set discount factor, batch size, and exploration parameters
        self.gamma = 0.99
        self.batch_size = config.batch_size
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Exploration: select a random action
            return random.randint(0, self.config.n_actions - 1), None
        else:
            # Exploitation: select the action with the highest Q-value
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state)
                return q_values.argmax().item(), q_values.detach().numpy()

    def store_transition(self, state, action, reward, next_state, done):
        # Store a transition in replay memory
        self.memory.append((state, action, reward, next_state, done))

    def update_q_network(self, input):
        # Store transitions in memory if experience replay is enabled
        if self.config.experience_replay:
            self.store_transition(*input)
            if len(self.memory) < self.batch_size:
                return 0
            # Sample a batch of transitions from memory
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
        else:
            states, actions, rewards, next_states, dones = input

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute current Q-values
        current_q_values = self.q_network(states)
        if self.config.experience_replay:
            current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        else:
            current_q_values = current_q_values[0][actions]

        # Compute next Q-values
        next_q_values = self.target_network(next_states).max(1)[0]
        # Compute target Q-values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        # Perform gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().numpy()

    def update_target_network(self):
        # Update target network weights with Q-network weights
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, path):
        # Save the Q-network model to the specified path
        torch.save(self.q_network.state_dict(), path)