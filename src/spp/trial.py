import numpy as np
from my_gym import Config, Environment
import tensorflow as tf
import random
from collections import deque

if __name__ == "__main__":
    config_file = "config.yaml"
    config = Config(config_file)
    env = Environment(config)

    # Load configuration
    config = Config("mypackage/config.yaml")  # Path to your YAML file
    env = Environment(config)  # Initialize the environment

    # Number of steps in one episode
    max_steps = 50  # Adjust based on the problem

# DQN for Agent Type 0
class DQNType0(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(DQNType0, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,))
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.fc3 = tf.keras.layers.Dense(output_dim)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

# DQN for Agent Type 1
class DQNType1(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(DQNType1, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,))
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.fc3 = tf.keras.layers.Dense(output_dim)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

# Agent Class for Type 0
class AgentType0:
    def __init__(self, env):
        self.env = env
        self.grid_size = env.grid_size
        self.n_actions = 4  # Up, Down, Left, Right

        self.input_dim = self.grid_size * self.grid_size  # Flattened grid

        # DQN networks
        self.q_network = DQNType0(input_dim=self.input_dim, output_dim=self.n_actions)
        self.target_network = DQNType0(input_dim=self.input_dim, output_dim=self.n_actions)
        self.target_network.set_weights(self.q_network.get_weights())

        # Optimizer and loss
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        # Replay buffer
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            state = np.expand_dims(state, axis=0).astype(np.float32)
            q_values = self.q_network(state)
            return np.argmax(q_values.numpy()[0])

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_q_network(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states).astype(np.float32)
        actions = np.array(actions)
        rewards = np.array(rewards).astype(np.float32)
        next_states = np.array(next_states).astype(np.float32)
        dones = np.array(dones).astype(np.float32)

        # Compute target Q-values
        next_q_values = self.target_network(next_states)
        max_next_q_values = np.max(next_q_values.numpy(), axis=1)
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Compute current Q-values
        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            indices = np.arange(self.batch_size)
            action_q_values = tf.gather_nd(q_values, np.vstack([indices, actions]).T)
            loss = self.loss_fn(target_q_values, action_q_values)

        # Update network parameters
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())
        
    # Agent Class for Type 1 (similar to Type 0)
class AgentType1:
    def __init__(self, env):
        self.env = env
        self.grid_size = env.grid_size
        self.n_actions = 4  # Up, Down, Left, Right

        self.input_dim = self.grid_size * self.grid_size  # Flattened grid

        # DQN networks
        self.q_network = DQNType1(input_dim=self.input_dim, output_dim=self.n_actions)
        self.target_network = DQNType1(input_dim=self.input_dim, output_dim=self.n_actions)
        self.target_network.set_weights(self.q_network.get_weights())

        # Optimizer and loss
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        # Replay buffer
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            state = np.expand_dims(state, axis=0).astype(np.float32)
            q_values = self.q_network(state)
            return np.argmax(q_values.numpy()[0])

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_q_network(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states).astype(np.float32)
        actions = np.array(actions)
        rewards = np.array(rewards).astype(np.float32)
        next_states = np.array(next_states).astype(np.float32)
        dones = np.array(dones).astype(np.float32)

        # Compute target Q-values
        next_q_values = self.target_network(next_states)
        max_next_q_values = np.max(next_q_values.numpy(), axis=1)
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Compute current Q-values
        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            indices = np.arange(self.batch_size)
            action_q_values = tf.gather_nd(q_values, np.vstack([indices, actions]).T)
            loss = self.loss_fn(target_q_values, action_q_values)

        # Update network parameters
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())
        
def get_agent_state(env, agent_id):
# Use the full grid as the observation
    grid = env.grid.copy()
    state = grid.flatten()
    return state

    
# # Run one episode
# print("Initial Environment:")
# env.render()

# for step in range(max_steps):
#     if env.done:
#         print(f"Episode ended after {step} steps.")
#         break

#     # Select random actions for all agents
#     action_list = random_action_selector(env.total_agents)
    
#     # Take a step in the environment
#     env.step(action_list)
    
#     # Render the environment
#     print(f"Step {step + 1}:")

# # Check results
# if env.done:
#     print(f"Winner: Agent Type {env.winner}")
# else:
#     print("Episode ended without a winner.")

    