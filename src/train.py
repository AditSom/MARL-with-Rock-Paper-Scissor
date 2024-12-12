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
def flatten_positions(positions,distance=False):
    if distance == False:
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

    def get_action(self):
        self.actions = []
        counter = 0
        for i, agent in enumerate(self.agents):
                    for iter in range(self.env.agents[i]):
                        agent_idx = counter
                        if agent_idx not in self.env.captured_agents:
                            state = copy.deepcopy(self.env.positions)
                            state[agent_idx]['agent'] = 4
                            if self.config.distance_input == True:
                                state = self.inject_distance(state,counter)
                            self.actions.append(agent.select_action(flatten_positions(state,self.config.distance_input))[0])
                            self.q_tracker.append(agent.select_action(flatten_positions(state,self.config.distance_input))[1])
                        counter += 1
        return self.actions
    
    def inject_distance(self, state,agent_idx):
        for i in range(self.env.total_agents):
            # Calculate the distance between the agent and the other agents
            distance = np.sqrt((state[agent_idx]['position'][0] - state[i]['position'][0])**2 + (state[agent_idx]['position'][1] - state[i]['position'][1])**2)
            # Inject the distance into the state
            state[i]['distance'] = distance
        return state

    
    def train_agent(self, prev_state, actions, rewards, next_state, done):
        counter = 0
        for i, agent in enumerate(self.agents):
                    for iter in range(self.env.agents[i]):
                        agent_idx = counter
                        if agent_idx not in self.env.captured_agents:
                            state = copy.deepcopy(prev_state)
                            next_state_copy = copy.deepcopy(next_state)
                            state[agent_idx]['agent'] = 4
                            next_state_copy[agent_idx]['agent'] = 4
                            if config.distance_input == True:
                                state = self.inject_distance(state,counter)
                                next_state_copy = self.inject_distance(next_state_copy,counter)
                            agent.update_q_network((
                                flatten_positions(state,self.config.distance_input),
                                actions[agent_idx],
                                rewards[i],
                                flatten_positions(next_state_copy,self.config.distance_input),
                                int(done)
                            ))
                        counter += 1

                        
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
        
        
        # Plot all the Q-values
        # plt.figure(figsize=(12, 5))
        # for i in range(self.env.n_agents):
        #     q_values = [q_values[i] for q_values in self.q_tracker]
        #     plt.plot(q_values, label=f'Agent {i + 1}')
        # plt.title('Q-values')
        # plt.xlabel('Step')
        # plt.ylabel('Q-value')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(self.config.save_path + 'q_values.png')
        # plt.show()

    def train(self):
        done = False
        frames = None
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
                
                # Select actions for each agent
                actions = self.get_action()

                # Take actions in the environment
                # For every step; store the previous state, actions, rewards, next state, done and position of the agents in monitor
                monitor[steps] = {
                    "prev_state": prev_state,
                    "actions": actions,
                    "positions": self.env.positions}
                next_state, rewards, done = self.env.step(actions, steps,episode)
                #print(prev_state)
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
    config = Config("config.yaml",args)    
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