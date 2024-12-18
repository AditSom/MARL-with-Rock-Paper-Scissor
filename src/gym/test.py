import numpy as np
from config import Config
from environment import Environment

if __name__ == "__main__":
    # Load configuration from the YAML file
    config_file = "config.yaml"
    config = Config(config_file)
    
    # Initialize the environment with the loaded configuration
    env = Environment(config)

    # Number of steps in one episode
    max_steps = 50  # Adjust based on the problem

    # Function to select random actions for all agents
    def random_action_selector(total_agents):
        # Each agent can take one of 4 possible actions: 0 (up), 1 (down), 2 (left), 3 (right)
        return np.random.choice(4, total_agents).tolist()

    # Print the initial state of the environment
    print("Initial Environment:")
    env.render()
    
    step = 0  # Initialize step counter
    
    # Run the episode until the environment signals it's done
    while not env.done:
        # Select random actions for all agents
        action_list = random_action_selector(env.total_agents)
        
        # Take a step in the environment with the selected actions
        env.step(action_list, step)
        
        # Render the environment after the step
        #print(f"Step {step + 1}:")
        
        # Check if the episode has ended
        if env.done:
            print(f"Episode ended after {step} steps.")
            break
        
        step += 1  # Increment step counter

    # Check and print the results after the episode ends
    if env.done:
        print(f"Winner: Agent Type {env.winner + 1}")
    else:
        print("Episode ended without a winner.")