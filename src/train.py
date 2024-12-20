import random
import numpy as np
from collections import deque
from gym import Config, Environment
import matplotlib.pyplot as plt
import copy
from models.get_model import get_model
import tqdm
import shutil
from args import get_args
import os

# Helper function to flatten positions
def flatten_positions(positions, distance=False):
    if not distance:
        return [feature for position in positions for feature in [position['agent'], position['position'][0], position['position'][1]]]
    return [
        feature for position in positions 
        for feature in [position['agent'], position['position'][0], position['position'][1], position['distance']]
    ]

class training:
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.agents = []
        self.num_episodes = config.num_episodes
        self.max_steps = config.max_steps
        self.rewards_per_episode = []
        self.steps_per_episode = []
        self.q_tracker = []
        self.loss_tracker = []

    # Get actions for all agents
    def get_action(self):
        self.actions = []
        counter = 0
        q_tracker = []
        for i, agent in enumerate(self.agents):
            for _ in range(self.env.agents[i]):
                agent_idx = counter
                if agent_idx not in self.env.captured_agents:
                    state = copy.deepcopy(self.env.positions)
                    state[agent_idx]['agent'] = 4
                    if self.config.distance_input:
                        state = self.inject_distance(state, counter)
                    self.actions.append(agent.select_action(flatten_positions(state, self.config.distance_input))[0])
                    q_tracker.append(agent.select_action(flatten_positions(state, self.config.distance_input))[1])
                else:
                    self.actions.append(-1)
                    q_tracker.append(0)
                counter += 1
        self.q_tracker.append(q_tracker)
        return self.actions

    # Inject distance information into the state
    def inject_distance(self, state, agent_idx):
        for i in range(self.env.total_agents):
            distance = np.sqrt((state[agent_idx]['position'][0] - state[i]['position'][0])**2 + (state[agent_idx]['position'][1] - state[i]['position'][1])**2)
            state[i]['distance'] = distance
        return state

    # Train the agent with the given experience
    def train_agent(self, prev_state, actions, rewards, next_state, done):
        counter = 0
        loss_track = []
        for i, agent in enumerate(self.agents):
            for _ in range(self.env.agents[i]):
                agent_idx = counter
                if agent_idx not in self.env.captured_agents:
                    state = copy.deepcopy(prev_state)
                    next_state_copy = copy.deepcopy(next_state)
                    state[agent_idx]['agent'] = 4
                    next_state_copy[agent_idx]['agent'] = 4
                    if self.config.distance_input:
                        state = self.inject_distance(state, counter)
                        next_state_copy = self.inject_distance(next_state_copy, counter)
                    loss = agent.update_q_network((
                        flatten_positions(state, self.config.distance_input),
                        actions[agent_idx],
                        rewards[i],
                        flatten_positions(next_state_copy, self.config.distance_input),
                        int(done)
                    ))
                    loss_track.append(float(loss))
                else:
                    loss_track.append(0)
                counter += 1
        self.loss_tracker.append(loss_track)

    # Create directories to store results
    def create_results(self):
        self.config.save_path = self.config.save_path + self.config.model + "_" + self.config.tag + "/"
        if not os.path.exists(self.config.save_path + "results/"):
            os.makedirs(self.config.save_path + "results/")
        shutil.copyfile("config.yaml", self.config.save_path + "results/config.yaml")
        if not os.path.exists(self.config.save_path + "weights/"):
            os.makedirs(self.config.save_path + "weights/")
        if not os.path.exists(self.config.save_path + "animations/"):
            os.makedirs(self.config.save_path + "animations/")
        if self.config.max_step_ani:
            if not os.path.exists(self.config.save_path + "animations_max_steps/"):
                os.makedirs(self.config.save_path + "animations_max_steps/")

    # Save results after training
    def save_results(self):
        self.save_model_weights()
        self.plot_learning_progress()

    # Save the model weights
    def save_model_weights(self):
        for i, agent in enumerate(self.agents):
            agent.save_model(self.config.save_path + f"weights/agent_{i}.pt")

    # Plot learning progress
    def plot_learning_progress(self):
        plt.figure(figsize=(12, 5))

        # Plot total reward per episode for each agent
        plt.subplot(1, 2, 1)
        for i in range(self.env.n_agents):
            rewards = [episode_rewards[i] for episode_rewards in self.rewards_per_episode]
            plt.plot(rewards, label=f'Agent {i + 1}')
        plt.title('Total Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()

        # Plot number of steps per episode
        plt.subplot(1, 2, 2)
        plt.plot(self.steps_per_episode)
        plt.title('Steps per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Number of Steps')

        # Save and show the plot
        plt.tight_layout()
        plt.savefig(self.config.save_path + 'training_process.png')
        plt.show()

        # Plot Q-values for each agent
        q_values = np.array(self.q_tracker)
        q_values = q_values.reshape(q_values.shape[0], self.env.total_agents, self.config.n_actions)
        plt.figure(figsize=(20, 10))
        for i in range(self.env.total_agents):
            plt.subplot(self.env.total_agents // 2 + self.env.total_agents % 2, 2, i + 1)
            for k in range(self.config.n_actions):
                plt.plot(q_values[:, i, k], label=f'Action {k}')
            plt.title(f'Agent {i + 1} Q-Values')
            plt.xlabel('Episode')
            plt.ylabel('Q-Value')
            plt.legend()
        plt.tight_layout()
        plt.savefig(self.config.save_path + 'all_agents_q_values.png')
        plt.show()

        # Plot the loss for each agent
        loss = np.array(self.loss_tracker)
        loss = loss.reshape(loss.shape[0], self.env.total_agents)
        plt.figure(figsize=(12, 5))
        for i in range(self.env.total_agents):
            plt.plot(loss[:, i], label=f'Agent {i + 1}')
        plt.title('Loss per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.config.save_path + 'loss.png')
        plt.show()

    # Main training loop
    def train(self):
        done = False
        self.create_results()
        for episode in tqdm.tqdm(range(self.num_episodes)):
            total_rewards = [0] * self.env.n_agents
            self.env.reset()
            self.env.render()
            done = False
            monitor = {}
            for steps in range(self.max_steps):
                if not done:
                    prev_state = copy.deepcopy(self.env.positions)
                    actions = self.get_action()
                    monitor[steps] = {
                        "prev_state": prev_state,
                        "actions": actions,
                        "positions": self.env.positions
                    }
                    next_state, rewards, done = self.env.step(actions, steps, episode)
                    self.train_agent(prev_state, actions, rewards, next_state, done)
                    for i, reward in enumerate(rewards):
                        total_rewards[i] += reward
                    step = steps
            for agent in self.agents:
                agent.update_target_network()
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay
            self.rewards_per_episode.append(total_rewards)
            self.steps_per_episode.append(step)
            if episode % 100 == 0:
                print(f"Episode {episode + 1}: Total Rewards: {total_rewards}, Steps: {step}")
        self.save_results()

if __name__ == "__main__":
    # Load configuration and initialize environment
    args = get_args()
    config = Config("config.yaml", args)
    env = Environment(config)
    Agent = get_model(config.model, env, config)
    trainer = training(config, env)

    # Initialize agents
    agents = [Agent(env, config) for _ in range(env.n_agents)]
    trainer.agents = agents

    # Start training
    trainer.train()
