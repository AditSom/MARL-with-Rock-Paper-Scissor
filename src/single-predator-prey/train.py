
import torch
import random
import numpy as np
from collections import deque
from model import DQN, Agent
from my_gym import Config, Environment
import matplotlib.pyplot as plt
import copy
import tqdm
config = Config("config.yaml")
env = Environment(config)
agents = []
for env_agent in range(env.n_agents):
    # allot a model for each agent
    agents.append(Agent(env, config))
num_episodes = 1000
rewards_per_episode = []
steps_per_episode = []
def flatten_positions(positions):
    flattened_positions = [[d['agent'],d['position'][0],d['position'][1]] for d in positions]
    flatten_positions = [item for sublist in flattened_positions for item in sublist]
    return flatten_positions

for episode in range(num_episodes):
    total_rewards = [0] * env.n_agents
    steps = 0
    done = False
    env = Environment(config)
    env.render()
    while steps < 500 and not done:
        counter = 0
        actions = []
        prev_state = copy.deepcopy(env.positions)

        for i in range(env.n_agents):
            agent = agents[i]
            for j in range(env.agents[i]):
                if counter not in env.captured_agents:
                    state = copy.deepcopy(env.positions)
                    state[counter]['agent'] = 4
                    actions.append(agent.select_action(flatten_positions(state)))
                    counter += 1
        next_state, rewards, done = env.step(actions,steps)

        if done==True: done = 1 
        else: done = 0
        counter = 0

        for i in range(env.n_agents):
            agent = agents[i]
            for j in range(env.agents[i]):
                # dont use store_transition
                if counter not in env.captured_agents:
                    state = copy.deepcopy(prev_state)
                    temp_next_state = copy.deepcopy(next_state)
                    state[counter]['agent'] = 4
                    temp_next_state[counter]['agent'] = 4
                    state = flatten_positions(state)
                    temp_next_state = flatten_positions(temp_next_state)
                    agent.update_q_network((state, actions[counter], rewards[i], temp_next_state, done))
                    counter += 1
        state = next_state
        for i in range(env.n_agents):
            total_rewards[i] += rewards[i]
        steps += 1
    for env_agent in range(env.n_agents):
        agent = agents[env_agent]
        agent.update_target_network()

    for i in range(env.n_agents):
        agent = agents[i]
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

    rewards_per_episode.append(total_rewards)
    steps_per_episode.append(steps)

    for i in range(env.n_agents):
        agent = agents[i]
    
    # Print rewards  and steps per agent
    if episode % 100 == 0:
        for i in range(env.n_agents):
            print(f"Episode {episode + 1}, Agent {i + 1}: Total Reward: {total_rewards[i]}, Steps: {steps},{done}")
            # Plot the learning process
    env.reset()
plt.figure(figsize=(12, 5))

# Plot total reward per episode for each agent
plt.subplot(1, 2, 1)
for i in range(env.n_agents):
    rewards = [episode_rewards[i] for episode_rewards in rewards_per_episode]
    plt.plot(rewards, label=f'Agent {i + 1}')
plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()

# Plot number of steps per episode
plt.subplot(1, 2, 2)
plt.plot(steps_per_episode)
plt.title('Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Number of Steps')
plt.savefig('learning_process.png')
plt.tight_layout()
plt.show()
