import random
import numpy as np
from collections import deque
from gym import Config, Environment
import matplotlib.pyplot as plt
import copy
import tqdm
import random
import numpy as np
from collections import deque
from models.get_model import get_model
from gym import Config, Environment
import matplotlib.pyplot as plt
import copy
import tqdm


# Helper function to flatten positions
def flatten_positions(positions):
    return [
        feature for position in positions 
        for feature in [position['agent'], position['position'][0], position['position'][1]]
    ]

class testing:
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.agents = []
        self.num_episodes = config.num_episodes
        self.max_steps = config.max_steps
        self.rewards_per_episode = []
        self.steps_per_episode = []
        self.winner = []

    def get_action(self):
        self.actions = []
        for i, agent in enumerate(self.agents):
                    for agent_idx in range(self.env.agents[i]):
                        if agent_idx not in self.env.captured_agents:
                            state = copy.deepcopy(self.env.positions)
                            state[agent_idx]['agent'] = 4
                            self.actions.append(agent.select_action(flatten_positions(state)))
        return self.actions
    
    def load_agent_weights(self):
        for i,agent in enumerate(self.agents):
            agent.load_model(self.config.save_path + f"weights/agent_{i}.pt")

    def test(self):
        done = False
        for episode in range(self.num_episodes):
            total_rewards = [0] * self.env.n_agents
            self.env.reset() 
            self.env.render()
            done = False
            for steps in tqdm.tqdm(range(self.max_steps)):
              if not done:
                prev_state = copy.deepcopy(self.env.positions)

                # Select actions for each agent
                actions = self.get_action()

                # Take actions in the environment
                next_state, rewards, done = self.env.step(actions, steps,episode)

                # Accumulate rewards and increment step counter
                for i, reward in enumerate(rewards):
                    total_rewards[i] += reward
                step = steps.copy()

            # Update target networks and epsilon for each agent
            for agent in self.agents:
                agent.update_target_network()
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay

            # Track rewards and steps
            self.rewards_per_episode.append(total_rewards)
            self.steps_per_episode.append(step)
            if done:
                self.winner.append(self.env.winner)
            
            # Log progress
            if episode % 100 == 0:
                print(f"Episode {episode + 1}: Total Rewards: {total_rewards}, Steps: {step}")
        self.metrics()

    def metrics(self):
        # Print number of wins for each agent
        for i in range(self.env.n_agents):
            print(f"Agent {i + 1} wins: {self.winner.count(i)}")
            # Save metrics
            with open(self.config.save_path + 'testing_metrics.txt', 'w') as f:
                f.write(f"Number of wins for each agent: {self.winner.count(i)}")
        self.plot_learning_progress()

    # Plotting function
    def plot_learning_progress(self):
        plt.figure(figsize=(12, 5))

        # Plot total reward per episode for each agent
        plt.subplot(1, 2, 1)
        # Run a sequential average to smooth the reward curve
        if config.smooth_plot:
            for i in range(self.env.n_agents):
                rewards = [episode_rewards[i] for episode_rewards in self.rewards_per_episode]
                smoothed_rewards = np.convolve(rewards, np.ones(100)/100, mode='valid')
                plt.plot(smoothed_rewards, label=f'Agent {i + 1}')
        else:
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
        plt.savefig(self.config.save_path + 'testing_process.png')
        plt.show()

if __name__ == "__main__":
    # Load configuration and initialize environment
    config = Config("config.yaml")    
    env = Environment(config)
    Agent = get_model(config.model, env, config)
    test = testing(config,env)
    # Initialize agents
    agents = [Agent(env, config) for _ in range(env.n_agents)]
    test.agents = agents
    test.load_agent_weights()

    # Hyperparameters
    num_episodes = config.num_episodes
    max_steps = config.max_steps

    
    # Testing performance
    test.test()