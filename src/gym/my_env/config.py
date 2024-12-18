import yaml  # Import the YAML library for handling YAML files

class Config:
    def __init__(self, yaml_file=None, args=None):
        # Default configuration settings
        self.grid_size = 10  # Size of the grid (grid_size x grid_size)
        self.agents = [3, 2]  # Number of agents for each type
        self.prey_predator_combo = {
            0: 1,    # Agent type 0 is prey for agent type 1
            1: None  # Agent type 1 has no prey
        }  # Predator-prey relationships between agent types
        self.reward = {
            'win': 10,          # Reward for winning the game
            'lose': -10,        # Penalty for losing the game
            'captured': 5,      # Reward for capturing another agent
            'eliminated': -3,   # Penalty for being eliminated by another agent
            'step': -1          # Penalty for each step taken
        }  # Reward structure for different events
        self.animation = True  # Enable or disable animation
        self.ani_save_path = "animation.gif"  # File path to save the animation
        self.fps = 5  # Frames per second for the animation
        self.position_random = True  # Randomize initial positions if True
        self.positions = [
            [[0, 0], [1, 1], [2, 2]],  # Predefined positions for agents of type 0
            [[3, 3], [4, 4]]           # Predefined positions for agents of type 1
        ]  # Used when position_random is False
        self.capture = True  # If True, predators capture prey; else they eliminate them
        self.track_grid = False  # Enable or disable grid tracking
        self.time_step = 100000  # Total number of time steps in the simulation
        self.n_actions = 5  # Number of possible actions for agents

        # Load configuration from a YAML file if provided
        if yaml_file:
            self.load_from_yaml(yaml_file)

        # Load configuration from command-line arguments if provided
        if args:
            self.load_from_args(args)

    def load_from_args(self, args):
        # Update configuration based on command-line arguments
        for key, value in vars(args).items():
            if value is not None:
                if 'reward' not in key:
                    # Set attribute directly if it's not a reward parameter
                    setattr(self, key, value)
                else:
                    # Update the reward dictionary for reward parameters
                    reward_key = key.split('_')[1]
                    self.reward[reward_key] = value

    def load_from_yaml(self, yaml_file):
        # Update configuration based on a YAML file
        with open(yaml_file, 'r') as file:
            yaml_config = yaml.safe_load(file)  # Load YAML content
            for key, value in yaml_config.items():
                # Set attributes based on keys and values from the YAML file
                setattr(self, key, value)
