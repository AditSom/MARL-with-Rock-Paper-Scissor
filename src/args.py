import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for prey-predator MARL simulation")
    
    # Simulation parameters
    parser.add_argument("--grid_size",'-g', type=int, help="Grid size for the simulation (e.g., 15x15).")
    parser.add_argument("--agents",'-a', type=int, nargs=2, help="Number of agents of each type (e.g., [type0, type1]).")
    parser.add_argument("--prey_predator_combo", type=str, 
                        help="Prey-predator interaction rules in format '0:1,1:None'.")
    
    # Reward parameters
    parser.add_argument("--reward_win",'-rw', type=float, help="Reward for winning.")
    parser.add_argument("--reward_lose",'-rl', type=float, help="Penalty for losing.")
    parser.add_argument("--reward_captured",'-rc', type=float, help="Reward for capturing prey.")
    parser.add_argument("--reward_eliminated",'-re', type=float, help="Penalty for elimination.")
    parser.add_argument("--reward_step",'-rs', type=float, help="Penalty for each step taken.")
    parser.add_argument("--reward_preystep",'-rps', type=float, help="Penalty for prey steps.")
    
    # Animation and visualization
    parser.add_argument("--animation", type=bool, help="Enable or disable animation.")
    parser.add_argument("--fps", type=int, help="Frames per second for animation.")
    parser.add_argument("--position_random", type=bool, help="Enable or disable random positions.")
    parser.add_argument("--capture", type=bool, help="Prey elimination or capture behavior.")
    
    # Training hyperparameters
    parser.add_argument("--num_episodes",'-ep', type=int, help="Number of episodes for training.")
    parser.add_argument("--max_steps",'-s', type=int, help="Maximum steps per episode.")
    parser.add_argument("--model", type=str, help="Model type (e.g., 'dqn').")
    parser.add_argument("--hidden_dim", type=int, nargs='+', 
                        help="Hidden dimensions of the neural network.")
    parser.add_argument("--save_path", type=str, 
                        help="Path to save experimental results.")
    parser.add_argument("--tag",'-t', type=str, help="Tag for the experiment.")
    parser.add_argument("--config_file", type=str, help="Path to the custom config file.")
    parser.add_argument("--smooth_plot", type=bool, help="Smooth the training plot.")
    parser.add_argument("--ani_batch_size", type=int, help="Batch size for animation.")
    
    return parser.parse_args()
