import numpy as np
from gym import Config, Environment

if __name__ == "__main__":
    config_file = "config.yaml"
    config = Config(config_file)
    env = Environment(config)

    # Load configuration
    config = Config("mypackage/config.yaml")  # Path to your YAML file
    env = Environment(config)  # Initialize the environment

    # Number of steps in one episode
    max_steps = 50  # Adjust based on the problem

    # Random action selector
    def random_action_selector(total_agents):
        # Each agent can take one of 4 possible actions: 0 (up), 1 (down), 2 (left), 3 (right)
        return np.random.choice(4, total_agents).tolist()

    # Run one episode
    print("Initial Environment:")
    env.render()

    for step in range(max_steps):
        if env.done:
            print(f"Episode ended after {step} steps.")
            break

        # Select random actions for all agents
        action_list = random_action_selector(env.total_agents)
        
        # Take a step in the environment
        env.step(action_list)
        
        # Render the environment
        print(f"Step {step + 1}:")

    # Check results
    if env.done:
        print(f"Winner: Agent Type {env.winner}")
    else:
        print("Episode ended without a winner.")

    