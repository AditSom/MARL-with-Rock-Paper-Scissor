import numpy as np  
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from .config import Config
import copy
import os
import moviepy as mpy

class Environment:
    def __init__(self, config):
        """
        Initialize the environment with the given configuration.
        """
        self.config = config
        self.grid_size = config.grid_size
        self.agents = config.agents  # List of agent counts per agent type
        self.n_agents = len(self.agents)  # Number of different agent types
        self.total_agents = sum(self.agents)  # Total number of agents
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.prey_predator_combo = config.prey_predator_combo  # Mapping of predators to their prey
        self.captured_agents = []  # List to keep track of captured agents
        self.done = False  # Indicator for end of the simulation
        self.rewards = config.reward  # Reward structure from the configuration
        self.updated_agent_count = []  # Updated count of agents after each step
        self.capture_record = []  # Record of captures and eliminations
        self.eliminated_record = [False] * self.n_agents  # Record of eliminated agent types
        self.reward = []  # Rewards for each agent type
        self.distances = {i: [] for i in range(self.total_agents)}  # Distance tracking for each agent
        self.winner = None  # Winning agent type
        if self.config.animation:
            self.store_positions = []  # Positions stored for animation
        self.frames_accumulated = []  # Frames accumulated for creating animations
        self.timestep = 0
        self.prev_pos = None

    def render(self):
        """
        Initialize the environment and update the grid with agents' starting positions.
        """
        self.start_positions()
        self.update_grid()

    def start_positions(self):
        """
        Set the initial positions of the agents on the grid.
        """
        self.positions = []
        for i in range(self.n_agents):
            for j in range(self.agents[i]):
                if self.config.position_random:
                    # Randomly assign positions, ensuring no overlaps
                    pos = list(np.random.randint(0, self.grid_size, size=2))
                    while pos in [p['position'] for p in self.positions]:
                        pos = list(np.random.randint(0, self.grid_size, size=2))
                    self.positions.append({'agent': i, 'position': pos})
                else:
                    # Assign predefined positions from the configuration
                    pos = self.config.positions[i][j]
                    self.positions.append({'agent': i, 'position': pos})
        # Store positions for animation if enabled
        if self.config.animation:
            self.store_positions.append(copy.deepcopy(self.positions))

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.done = False
        self.captured_agents = []
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.positions = []
        self.reward = [0] * self.n_agents
        self.winner = None
        self.eliminated_record = [False] * self.n_agents
        self.capture_record = []
        self.distances = {i: [] for i in range(self.total_agents)}
        self.boundary = {}
        if self.config.animation:
            self.store_positions = []

    def update_grid(self, position=None, print_grid=True):
        """
        Update the grid with the current positions of the agents.
        """
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        if position is not None:
            # Update grid with provided positions
            for i in range(self.total_agents):
                if position[i]['agent'] != -1:
                    x, y = position[i]['position']
                    self.grid[x, y] = position[i]['agent'] + 1
        else:
            # Update grid with current positions of agents
            for i in range(self.total_agents):
                if i not in self.captured_agents:
                    x, y = self.positions[i]['position']
                    self.grid[x, y] = self.positions[i]['agent'] + 1

    def render_grid(self):
        """
        Print the current grid to the console.
        """
        for row in self.grid:
            print(' '.join(str(cell) for cell in row))
        print()

    def action_update(self, action, agent_id):
        """
        Update the position of an agent based on the action taken.
        """
        pos = copy.deepcopy(self.positions[agent_id]['position'])
        if action == 0:  # Move down
            if pos[0] == 0:
                self.boundary[agent_id] = True  # Agent hits the boundary
            pos[0] = max(0, pos[0] - 1)
        elif action == 1:  # Move up
            if pos[0] == self.grid_size - 1:
                self.boundary[agent_id] = True
            pos[0] = min(self.grid_size - 1, pos[0] + 1)
        elif action == 2:  # Move left
            if pos[1] == 0:
                self.boundary[agent_id] = True
            pos[1] = max(0, pos[1] - 1)
        elif action == 3:  # Move right
            if pos[1] == self.grid_size - 1:
                self.boundary[agent_id] = True
            pos[1] = min(self.grid_size - 1, pos[1] + 1)
        # Check if the new position is occupied by another agent
        for i in range(self.total_agents):
            if i != agent_id and self.prev_positions[i]['position'] == pos:
                # If the new position has the predator, the prey can't move
                if self.prey_predator_combo[self.positions[i]['agent']] == self.prev_positions[agent_id]['agent']:
                    pos = self.positions[agent_id]['position']  # Revert to original position
                    break
        self.positions[agent_id]['position'] = pos

    def capture_check(self, timestep=None):
        """
        Check if any agents have been captured or eliminated.
        """
        self.capture_record = [[0] * self.n_agents, [0] * self.n_agents]
        for i in range(self.total_agents):
            if i not in self.captured_agents:
                for j in range(self.total_agents):
                    if j not in self.captured_agents and self.positions[i]['position'] == self.positions[j]['position']:
                        # Check for predator-prey interaction
                        if self.prey_predator_combo[self.positions[i]['agent']] == self.positions[j]['agent']:
                            if self.config.capture:
                                # Convert the captured agent to the agent type of the captor
                                self.positions[j]['agent'] = self.positions[i]['agent']
                            else:
                                # Eliminate the captured agent
                                self.positions[j]['agent'] = -1
                                self.positions[j]['position'] = [-1, -1]
                                self.captured_agents.append(j)
                            # Update capture and elimination records
                            self.capture_record[0][self.positions[i]['agent']] += 1
                            self.capture_record[1][self.positions[j]['agent']] += 1
        # Update the count of agents per type
        agent_count = [0] * self.n_agents
        for i in range(self.total_agents):
            if i not in self.captured_agents:
                agent_count[self.positions[i]['agent']] += 1
        self.updated_agent_count = agent_count
        # Check if any agent types have been eliminated
        for i in range(self.n_agents):
            if agent_count[i] == 0 and not self.eliminated_record[i]:
                self.eliminated_record[i] = True  # Mark the agent type as eliminated
        # Determine if the simulation is done and identify the winner
        if self.config.capture:
            if max(agent_count) == self.total_agents:
                self.done = True
                self.winner = agent_count.index(max(agent_count))
        else:
            if self.eliminated_record.count(False) == 1:
                self.done = True
                self.winner = self.eliminated_record.index(False)

    def distance_update(self):
        """
        Update the distances between agents for reward calculations.
        """
        for i in range(self.total_agents):
            if i not in self.captured_agents:
                for j in range(self.total_agents):
                    if j not in self.captured_agents:
                        if self.positions[i]['agent'] == self.prey_predator_combo[self.positions[j]['agent']]:
                            # Agent i is prey for agent j
                            distance = -1 * (abs(self.positions[i]['position'][0] - self.positions[j]['position'][0]) +
                                             abs(self.positions[i]['position'][1] - self.positions[j]['position'][1]))
                            self.distances[i].append(distance)
                        if self.positions[j]['agent'] == self.prey_predator_combo[self.positions[i]['agent']]:
                            # Agent i is predator for agent j
                            distance = (abs(self.positions[i]['position'][0] - self.positions[j]['position'][0]) +
                                        abs(self.positions[i]['position'][1] - self.positions[j]['position'][1]))
                            self.distances[i].append(distance)

    def reward_update(self):
        """
        Update the rewards for each agent based on captures, eliminations, distance, and steps.
        """
        self.reward = [0] * self.n_agents
        # Update distances if required
        if self.config.distance:
            if self.config.distance_type == 'average':
                self.distance_update()
            if self.config.distance_type == 'last' and self.done:
                self.distance_update()
        # Assign rewards if the simulation is done
        if self.done:
            for i in range(self.n_agents):
                if i == self.winner:
                    self.reward[i] = self.rewards['win']
                else:
                    # Assign win reward to prey agents that survived
                    if self.prey_predator_combo[i] == 'None' and self.updated_agent_count[i] > 0:
                        self.reward[i] = self.rewards['win']
                    else:
                        self.reward[i] = self.rewards['lose']
            # Add distance-based rewards
            if self.config.distance:
                for i in range(self.total_agents):
                    if len(self.distances[i]) > 0:
                        average_distance = np.mean(self.distances[i])
                        self.reward[self.positions[i]['agent']] += self.rewards['distance'] * average_distance
                    else:
                        self.reward[self.positions[i]['agent']] += self.rewards['distance'] * 0
            # Add rewards for captures, eliminations, and steps
            for i in range(self.n_agents):
                if self.prey_predator_combo[i] != 'None':
                    self.reward[i] += (self.capture_record[0][i] * self.rewards['captured'] +
                                       self.capture_record[1][i] * self.rewards['eliminated'] +
                                       self.rewards['step'])
                else:
                    self.reward[i] += (self.capture_record[0][i] * self.rewards['captured'] +
                                       self.capture_record[1][i] * self.rewards['eliminated'] +
                                       self.rewards['prey_step'])
                # Add boundary penalty if applicable
                if self.config.boundary:
                    if i in self.boundary.keys():
                        self.reward[i] += self.rewards['boundary']

    def render_entire_episode(self, ep, ani=False):
        """
        Render the progression of the episode as an animation and save it as a video.
        """
        # Compile video every 50 episodes or if animation is forced
        if (ep + 1) % 50 == 0 or ani:
            for frame in range(len(self.store_positions)):
                self.update_grid(self.store_positions[frame], print_grid=False)
                grid_copy = copy.deepcopy(self.grid)
                temp_dict = {
                    'grid': grid_copy,
                    'episode': ep + 1,
                    'step': frame + 1
                }
                self.frames_accumulated.append(copy.deepcopy(temp_dict))
            # Create the animation
            fig, ax = plt.subplots()
            grid_data = np.zeros((self.grid_size, self.grid_size), dtype=int)
            im = ax.imshow(grid_data, cmap="viridis", vmin=0, vmax=self.n_agents + 1)
            fig.colorbar(im, ax=ax)

            def update(frame):
                grid = self.frames_accumulated[frame]['grid']
                im.set_data(copy.deepcopy(grid))
                ax.set_title(f"Step {self.frames_accumulated[frame]['step']}, Episode {self.frames_accumulated[frame]['episode']}")
                return [im]

            # Determine output path for the animation
            output_dir = os.path.join(self.config.save_path, "animations")
            if ani:
                output_dir = os.path.join(self.config.save_path, "animations_max_steps")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"animation_batch_{ep // 50}.gif")

            # Save the animation
            ani = FuncAnimation(fig, update, frames=len(self.frames_accumulated), interval=100, blit=True)
            ani.save(output_path, fps=self.config.fps, writer="pillow")
            # Close the figure to prevent memory leaks
            plt.close(fig)
            self.frames_accumulated = []

    def plot_grid(self):
        """
        Plot the final grid at the end of the episode.
        """
        if not hasattr(self, "grid"):
            print("No grid data to plot. Ensure steps have been performed.")
            return
        cmap = plt.cm.get_cmap('viridis', self.n_agents + 1)
        self.update_grid(print_grid=False)
        grid = np.array(self.grid)
        plt.figure(figsize=(self.config.grid_size, self.config.grid_size))
        plt.imshow(grid, cmap=cmap, interpolation='nearest')
        plt.colorbar(ticks=range(self.n_agents + 1), label='Agent Type')
        plt.title('Agent Grid')
        plt.xlabel('X')
        plt.ylabel('Y')
        output_path = os.path.join(self.config.save_path, "grid.png")
        plt.savefig(output_path)

    def step(self, action_list, timestep, ep):
        """
        Perform a step in the environment using the provided list of actions.
        """
        self.prev_positions = copy.deepcopy(self.positions)
        for i in range(self.total_agents):
            if i not in self.captured_agents:
                self.action_update(action_list[i], i)
        # Check for captures and eliminations
        self.capture_check(timestep)
        ani = False
        # Check if maximum steps have been reached
        if timestep + 1 == self.config.max_steps:
            self.done = True
            if ep > 5000:
                ani = self.config.max_step_ani
        # Update rewards
        self.reward_update()
        # Store positions for animation
        if self.config.animation:
            self.store_positions.append(copy.deepcopy(self.positions))
        # Update the grid at specified intervals
        if self.config.track_grid is not False and timestep % self.config.track_grid == 0:
            self.update_grid()
        # Render the entire episode if done and animation is enabled
        if self.done and self.config.animation:
            self.render_entire_episode(ep, ani)
        return copy.deepcopy(self.positions), copy.deepcopy(self.reward), copy.deepcopy(self.done)

if __name__ == "__main__":
    config = "config.yaml"
    config = Config(config)
    env = Environment(config)

    done = False
    env.render()
    timestep = 0
    ep = 0
    while not done:
        # Generate random actions for all agents
        action_list = [np.random.choice(4) for _ in range(env.total_agents)]
        positions, reward, done = env.step(action_list, timestep, ep)
        env.render()  # Render the grid after each step
        print(f"Actions: {action_list}, Reward: {reward}, Done: {done}")
        timestep += 1