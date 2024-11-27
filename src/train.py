import random
import numpy as np
from collections import deque
from spp.model import DQN, Agent
from my_gym import Config, Environment
import matplotlib.pyplot as plt
import copy
import tqdm
from spp.train import flatten_positions, training 

if __name__ == "__main__":
    # Load configuration and initialize environment
    config = Config("config.yaml")    
    env = Environment(config)

    # Initialize agents
    agents = [Agent(env, config) for _ in range(env.n_agents)]

    # Hyperparameters
    num_episodes = config.num_episodes
    max_steps = config.max_steps

    # Tracking rewards and steps
    rewards_per_episode = []
    steps_per_episode = []

    trainer = training(config,env)
    trainer.plot_learning_progress()