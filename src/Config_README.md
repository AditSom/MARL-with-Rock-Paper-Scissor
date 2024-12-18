# Configuration File for MARL with Rock-Paper-Scissor

This README explains all the parameters in the configuration file for the MARL with Rock-Paper-Scissor environment and how to adjust them. All of the recommended values are as in the given config.yaml

## Grid Settings

- **grid_size**: *(Default: 5)*  
    Sets the size of the grid environment. The grid will be `grid_size x grid_size`.

    *To change*: Set `grid_size` to the desired integer value. 

## Agent Settings

- **agents**: *(Default: [1, 1])*  
    Specifies the number of agents of each type. For example, `[1, 2]` means 1 agents of type 0 and 2 agents of type 1.

    *To change*: Modify the list to include the desired number of agents for each type.

## Prey-Predator Relationship

- **prey_predator_combo**:  
    Defines which agent types can capture others.
    - `0: 1` means agents of type 1 can capture agents of type 0.
    - `1: None` means agents of type 0 cannot capture agents of type 1.

    *To change*: Adjust the mapping to reflect the desired prey-predator relationships.

## Reward Settings

- **reward**:  
    Configures the reward system.
    - **win**: *(Default: 1)* Reward for winning.
    - **lose**: *(Default: -1)* Penalty for losing.
    - **captured**: *(Default: 1)* Reward for capturing another agent.
    - **eliminated**: *(Default: -1)* Penalty for being eliminated.
    - **step**: *(Default: -0.1)* Penalty per timestep in an episode.
    - **prey_step**: *(Default: 0.1)* Penalty per step taken by the prey.
    - **distance**: *(Default: -0.1)* Penalty per unit distance.
    - **boundary**: *(Default: -0.1)* Penalty for hitting the boundary.

    *To change*: Update the corresponding values under `reward`.

## Animation Settings

- **animation**: *(Default: True)*  
    Enables or disables animation.

    *To change*: Set to `True` to enable or `False` to disable.

- **fps**: *(Default: 3)*  
    Frames per second for the animation.

    *To change*: Set `fps` to the desired frame rate.

## Position Settings

- **position_random**: *(Default: True)*  
    Determines if agents start at random positions.

    *To change*: Set to `True` for random positions or `False` for fixed positions.

## Capture Settings

- **capture**: *(Default: False)*  
    Specifies whether prey is captured or eliminated upon being caught.

    *To change*: Set to `True` to eliminate or `False` to capture.

## Training Hyperparameters

- **n_actions**: *(Default: 5)*  
    Number of possible actions for agents.

    *To change*: Adjust `n_actions` to the desired number. It can either be 4 or 5. Setting it to 4 would remove stay action.

- **num_episodes**: *(Default: 10000)*  
    Total number of training episodes.

    *To change*: Set `num_episodes` to the desired value.

- **max_steps**: *(Default: 15)*  
    Maximum steps allowed per episode.

    *To change*: Update `max_steps` accordingly.

- **model**: *(Default: "dqn")*  
    Specifies the model type used.

    *To change*: Set `model` to the desired model name. Currently only supports dqn.

- **hidden_dim**: *(Default: [64, 128, 256, 128, 64])*  
    Defines the dimensions of hidden layers in the neural network.

    *To change*: Modify the list with desired layer sizes.

## Save Path for Results

- **save_path**: *(Default: "results/")*  
    Directory where results are saved.

    *To change*: Set `save_path` to the desired directory path.

## Tag for the Experiment

- **tag**: *(Default: "trial_57")*  
    A unique identifier for the experiment.

    *To change*: Assign a new string to `tag`.

## Path to the Custom Config File

- **config_file**: *(Default: "src/config.yaml")*  
    Path to the configuration file.

    *To change*: Update `config_file` with the new file path.

## Plot Settings

- **smooth_plot**: *(Default: True)*  
    Enables smoothing of the training plot. 

    *To change*: Set to `True` to smooth or `False` to disable.

## Animation Batch Size

- **ani_batch_size**: *(Default: 50)*  
    Number of episodes to animate in a batch.

    *To change*: Adjust `ani_batch_size` to the desired size.

## Distance Settings

- **distance**: *(Default: True)*  
    Activates distance-based calculations.

    *To change*: Set to `True` to enable or `False` to disable.

- **distance_type**: *(Default: "average")*  
    Type of distance calculation ("average" or "last"). Average takes the distance throughout the entire episode and takes this average as the penalty. Setting it to last calculates the distances between the prey-predator combo only during the last time-step.

    *To change*: Set `distance_type` to `"average"` or `"last"`.

## Animation Settings for Max Step

- **max_step_ani**: *(Default: True)*  
    Animates episodes that reach maximum steps.

    *To change*: Set to `True` to enable or `False` to disable.

## Boundary Settings

- **boundary**: *(Default: True)*  
    Determines if boundaries are enabled. If true, trying to cross the boundary penalizes the agent.

    *To change*: Set to `True` to enable boundaries or `False` to disable.

## Distance Input Settings

- **distance_input**: *(Default: True)*  
    Gives the distance as an input to the model.

    *To change*: Set to `True` to include or `False` to exclude.

## Experience Replay Settings

- **experience_replay**: *(Default: True)*  
    Enables experience replay during training.

    *To change*: Set to `True` to enable or `False` to disable.

## Batch Size for Training

- **batch_size**: *(Default: 32)*  
    Batch size for training updates.

    *To change*: Set `batch_size` to the desired number.
