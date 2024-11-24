import yaml

class Config:
    def __init__(self, yaml_file=None):
        # Default configuration
        self.grid_size = 10  # Size of the grid (10x10)
        self.agents = [3, 2]  # Number of agents for each type
        self.prey_predator_combo = {0: 1, 1: 0}  # Predator-prey relationships
        self.reward = {
            'win': 10,          # Reward for winning
            'lose': -10,        # Penalty for losing
            'captured': 5,      # Reward for capturing
            'eliminated': -3,   # Penalty for being eliminated
            'step': -1          # Penalty for each step
        }
        self.animation = True  # Enable/disable animation
        self.ani_save_path = "animation.gif"  # Path to save the animation
        self.fps = 5  # Frames per second for animation
        self.position_random = True  # Randomize initial positions
        self.positions = [  # Predefined positions (used if position_random is False)
            [[0, 0], [1, 1], [2, 2]],  # Positions for agents of type 0
            [[3, 3], [4, 4]]  # Positions for agents of type 1
        ]
        self.capture = True  # Predators capture or eliminate prey
        self.track_grid = False
        self.time_step = 100000
        # Load and override parameters from a YAML file if provided
        if yaml_file:
            self.load_from_yaml(yaml_file)
        self.n_actions = 5

    def load_from_yaml(self, yaml_file):
        with open(yaml_file, 'r') as file:
            yaml_config = yaml.safe_load(file)
            for key, value in yaml_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    print(f"Warning: Unrecognized configuration parameter '{key}' in {yaml_file}")


