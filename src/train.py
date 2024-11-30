import random
import numpy as np
from collections import deque
from gym import Config, Environment
import matplotlib.pyplot as plt
import copy
from models.get_model import get_model
import tqdm
import shutil
import os
from args import get_args


# Helper function to flatten positions
def flatten_positions(positions):
    return [
        feature for position in positions 
        for feature in [position['agent'], position['position'][0], position['position'][1]]
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

    def get_action(self):
        self.actions = []
        for i, agent in enumerate(self.agents):
                    for agent_idx in range(self.env.agents[i]):
                        if agent_idx not in self.env.captured_agents:
                            state = copy.deepcopy(self.env.positions)
                            state[agent_idx]['agent'] = 4
                            self.actions.append(agent.select_action(flatten_positions(state)))
        return self.actions
    
    def train_agent(self, prev_state, actions, rewards, next_state, done):
        for i, agent in enumerate(self.agents):
                    for agent_idx in range(self.env.agents[i]):
                        if agent_idx not in self.env.captured_agents:
                            state = copy.deepcopy(prev_state)
                            next_state_copy = copy.deepcopy(next_state)
                            state[agent_idx]['agent'] = 4
                            next_state_copy[agent_idx]['agent'] = 4
                            agent.update_q_network((
                                flatten_positions(state),
                                actions[agent_idx],
                                rewards[i],
                                flatten_positions(next_state_copy),
                                int(done)
                            ))
    def create_results(self):
        # Create a folder to store the results if it does not exist
        self.config.save_path = self.config.save_path + self.config.model + "_" + self.config.tag + "/"
        if not os.path.exists(self.config.save_path + "results/"):
            os.makedirs(self.config.save_path + "results/")
        # Copy the configuration file to the results folder using shutil
        shutil.copyfile("config.yaml", self.config.save_path + "results/config.yaml")
        # Create a weights folder to store the model weights if it does not exist
        if not os.path.exists(self.config.save_path + "weights/"):
            os.makedirs(self.config.save_path + "weights/")
        # Create a folder to store the animations if it does not exist
        if not os.path.exists(self.config.save_path + "animations/"):
            os.makedirs(self.config.save_path + "animations/")
        
    def save_results(self):
        # Save the model weights
        self.save_model_weights()
        # Plot the learning progress
        self.plot_learning_progress()

    # Save the model weights
    def save_model_weights(self):
        for i, agent in enumerate(self.agents):
            # Use pytorch's save method to save the model weights
            agent.save_model(self.config.save_path + f"weights/agent_{i}.pt")

    # Plotting function
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

    def train(self):
        done = False
        frames = None
        self.create_results()
        for episode in tqdm.tqdm(range(self.num_episodes)):
            total_rewards = [0] * self.env.n_agents
            self.env.reset() 
            self.env.render()
            done = False
            for steps in range(self.max_steps):
              if not done:
                prev_state = copy.deepcopy(self.env.positions)

                # Select actions for each agent
                actions = self.get_action()

                # Take actions in the environment

                next_state, rewards, done = self.env.step(actions, steps,episode)

                # Update agents with experience
                self.train_agent(prev_state, actions, rewards, next_state, done)

                # Accumulate rewards and increment step counter
                for i, reward in enumerate(rewards):
                    total_rewards[i] += reward
                step = steps

            # Update target networks and epsilon for each agent
            for agent in self.agents:
                agent.update_target_network()
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay

            # Track rewards and steps
            self.rewards_per_episode.append(total_rewards)
            self.steps_per_episode.append(step)

            # Log progress
            if episode % 100 == 0:
                print(f"Episode {episode + 1}: Total Rewards: {total_rewards}, Steps: {step}")
        self.save_results()
    

if __name__ == "__main__":
    # Load configuration and initialize environment
    args = get_args()
    config = Config("config.yaml", args)    
    env = Environment(config)
    Agent = get_model(config.model, env, config)
    trainer = training(config,env)

    # Initialize agents
    agents = [Agent(env, config) for _ in range(env.n_agents)]
    trainer.agents = agents

    # Hyperparameters
    num_episodes = config.num_episodes
    max_steps = config.max_steps

    # Tracking rewards and steps
    rewards_per_episode = []
    steps_per_episode = []

    # Training loop
    trainer.train()