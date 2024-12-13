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
        self.config = config
        self.grid_size = config.grid_size
        self.agents = config.agents
        self.n_agents = len(self.agents)
        self.total_agents = sum(self.agents)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.prey_predator_combo = config.prey_predator_combo
        self.captured_agents = []
        self.done = False
        self.rewards = config.reward
        self.updated_agent_count = []
        self.capture_record = []
        self.eliminated_record = [False] * self.n_agents
        self.reward = []
        self.winner = None
        if self.config.animation:
            self.store_positions = []
        self.frames_accumulated = []
        self.timestep = 0
        self.prev_pos = None

    def render(self):
        """Initialize the environment and update the grid."""
        self.start_positions()
        self.update_grid()

    def start_positions(self):
        """Set the initial positions of the agents."""
        self.positions = []
        for i in range(self.n_agents):
            for j in range(self.agents[i]):
                if self.config.position_random:
                    pos = list(np.random.randint(0, self.grid_size, size=2))
                    
                    while pos in [p['position'] for p in self.positions]:
                        pos = list(np.random.randint(0, self.grid_size, size=2))
                    self.positions.append({'agent': i, 'position': pos})
                else:
                    pos = self.config.positions[i][j]
                    self.positions.append({'agent': i, 'position': pos})
        # Check if the initial positions are valid and reassign if necessary
        self.store_positions.append(copy.deepcopy(self.positions))

    def reset(self):
        """Reset the environment to its initial state."""
        self.done = False
        self.captured_agents = []
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.positions = []
        self.reward = [0] * self.n_agents
        self.winner = None
        self.eliminated_record = [False] * self.n_agents
        self.capture_record = []
        self.boundary = {}
        if self.config.animation:
            self.store_positions = []
        
    def update_grid(self, position=None, print_grid=True):
        """Update the grid with the current positions of the agents."""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        if position is not None:
            for i in range(self.total_agents):
                if position[i]['agent']!=-1:
                    self.grid[position[i]['position'][0], position[i]['position'][1]] = position[i]['agent'] + 1
        else:
            for i in range(self.total_agents):
                if i not in self.captured_agents:
                    self.grid[self.positions[i]['position'][0], self.positions[i]['position'][1]] = self.positions[i]['agent'] + 1

    def render_grid(self):
        """Print the current grid to the console."""
        for row in self.grid:
            print(' '.join(str(cell) for cell in row))
        print()

    def action_update(self, action, agent_id):
        """Update the position of an agent based on the action taken."""
        pos = copy.deepcopy(self.positions[agent_id]['position'])
        if action == 0:  # Move down\
            if pos[0] == 0:
                self.boundary[agent_id] = True
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
                # If the new position has the predator then the prey can't move
                if self.prey_predator_combo[self.positions[i]['agent']] == self.prev_positions[agent_id]['agent']:
                    pos = self.positions[agent_id]['position']
                    break
        
        self.positions[agent_id]['position'] = pos

    def capture_check(self,timestep=None):
        """Check if any agents have been captured."""
        self.capture_record = [[0] * self.n_agents, [0] * self.n_agents]
        for i in range(self.total_agents):
            if i not in self.captured_agents:
                for j in range(self.total_agents):
                    if j not in self.captured_agents and self.positions[i]['position'] == self.positions[j]['position']:
                        if self.prey_predator_combo[self.positions[i]['agent']] == self.positions[j]['agent']:
                            if self.config.capture:
                                self.positions[j]['agent'] = self.positions[i]['agent']
                            else:
                                self.positions[j]['agent'] = -1
                                self.positions[j]['position'] = [-1, -1]
                                self.captured_agents.append(j)
                            self.capture_record[0][self.positions[i]['agent']] += 1
                            self.capture_record[1][self.positions[j]['agent']] += 1

        agent_count = [0] * self.n_agents
        for i in range(self.total_agents):
            if i not in self.captured_agents:
                agent_count[self.positions[i]['agent']] += 1
        self.updated_agent_count = agent_count

        for i in range(self.n_agents):
            if agent_count[i] == 0 and not self.eliminated_record[i]:
                self.eliminated_record[i] = True
                #print(f"Agent {i + 1} eliminated")

        if self.config.capture:
            if max(agent_count) == self.total_agents:
                self.done = True
                self.winner = agent_count.index(max(agent_count))
                #print(f"Agent {self.winner} wins")
        else:
            if self.eliminated_record.count(False) == 1:
              self.done = True
              self.winner = self.eliminated_record.index(False)
              #print(f"Agent {self.winner} wins")
                    
    def reward_update(self):
        """Update the rewards for each agent."""
        self.reward = [0] * self.n_agents
        if self.done:
            for i in range(self.n_agents):
                if i == self.winner:
                    self.reward[i] = self.rewards['win']
                else:
                    # if the agent is prey and not captured, give it a reward
                    if self.prey_predator_combo[i] == 'None' and self.updated_agent_count[i] > 0:
                        self.reward[i] = self.rewards['win'] 
                    else:
                        self.reward[i] = self.rewards['lose']
                if self.config.distance:
                           # print('True1')
                            for j in range(self.total_agents):
                                if j not in self.captured_agents:
                                    # print(self.positions[j]['agent'],self.prey_predator_combo[i])
                                    if self.positions[j]['agent'] == self.prey_predator_combo[i]:
                                        #print('True')
                                        self.reward[i] += self.rewards['distance'] * (abs(self.positions[i]['position'][0] - self.positions[j]['position'][0]) + abs(self.positions[i]['position'][1] - self.positions[j]['position'][1]))
                                    # if i is a prey and j is a predator, then the prey gets a reward for the distance between them
                                    if self.positions[i]['agent'] == self.prey_predator_combo[self.positions[j]['agent']]:
                                        self.reward[i] += -1*self.rewards['distance'] * (abs(self.positions[i]['position'][0] - self.positions[j]['position'][0]) + abs(self.positions[i]['position'][1] - self.positions[j]['position'][1]))
        else:
            for i in range(self.n_agents):
                if self.prey_predator_combo[i]!='None':
                    self.reward[i] = self.capture_record[0][i] * self.rewards['captured'] +self.capture_record[1][i] * self.rewards['eliminated']+self.rewards['step']
                else:
                    self.reward[i] = self.capture_record[0][i] * self.rewards['captured'] +self.capture_record[1][i] * self.rewards['eliminated']+self.rewards['prey_step']
                if self.config.boundary:
                    if i in self.boundary.keys():
                        self.reward[i] += self.rewards['boundary']

    def render_entire_episode(self, ep):
        """
        Render the progression of the episode as an animation, save it as a video, 
        and group episodes into batches of 50 videos.
        """
        # Compile video every 50 episodes
        if (ep+1)%50 == 0:
            for frame in range(len(self.store_positions)):
                self.update_grid(self.store_positions[frame], print_grid=False)
                grid_copy = copy.deepcopy(self.grid)
                temp_dict = {}
                temp_dict = {
                    'grid': grid_copy,
                    'episode': ep+1,
                    'step': frame + 1
                }
                self.frames_accumulated.append(copy.deepcopy(temp_dict))
            fig, ax = plt.subplots()
            grid_data = np.zeros((self.grid_size, self.grid_size), dtype=int)
            im = ax.imshow(grid_data, cmap="viridis", vmin=0, vmax=self.n_agents + 1)
            fig.colorbar(im, ax=ax)
            def update(frame):
                grid = self.frames_accumulated[frame]['grid']
                im.set_data(copy.deepcopy(grid))
                ax.set_title(f"Step { self.frames_accumulated[frame]['step']}, Episode {self.frames_accumulated[frame]['episode']}")
                return [im]

            output_path = os.path.join(self.config.save_path, f"animations/animation_batch_{ep // 50}.gif")
            ani = FuncAnimation(fig, update, frames=len(self.frames_accumulated), interval=100, blit=True)
            ani.save(output_path,fps=self.config.fps,writer="pillow")
            #print(f"Compiled video saved to {output_path}")
            # Close the figure to prevent memory leaks
            
            plt.close(fig)
            self.frames_accumulated = []

    def plot_grid(self):
        """Plot the final grid at the end of the episode."""
        if not hasattr(self, "grid"):
            print("No grid data to plot. Ensure steps have been performed.")
            return
        cmap = plt.cm.get_cmap('viridis', 4)
        self.update_grid(print_grid=False)
        print(self.positions)
        grid = np.array(self.grid)
        plt.figure(figsize=(self.config.grid_size, self.config.grid_size))
        plt.imshow(grid, cmap=cmap, interpolation='nearest')
        plt.colorbar(ticks=range(self.n_agents + 1), label='Agent Type')
        plt.title('Agent Grid')
        plt.xlabel('X')
        plt.ylabel('Y')
        output_path = self.config.save_path + "grid.png"
        plt.savefig(output_path)

    def step(self, action_list, timestep, ep):
        """Perform a step in the environment."""
        self.prev_positions = copy.deepcopy(self.positions)
        for i in range(self.total_agents):
            if i not in self.captured_agents:
                self.action_update(action_list[i], i)
        self.capture_check(timestep)
        if timestep+1 == self.config.max_steps:
            self.done = True
        self.reward_update()
        if self.config.animation:
            self.store_positions.append(copy.deepcopy(self.positions))
        if self.config.track_grid is not False and timestep % self.config.track_grid == 0:
            self.update_grid()
        if self.done and self.config.animation:
            # self. grid()
            self.render_entire_episode(ep)
        return copy.deepcopy(self.positions), copy.deepcopy(self.reward), copy.deepcopy(self.done)

if __name__ == "__main__":
    config = "config.yaml"
    config = Config(config)
    env = Environment(config)

    done = False
    env.render()
    while not done:
        action = np.random.choice(4)  # Random action for the predator
        grid, reward, done = env.step(action)
        env.render()
        print(f"Action: {action}, Reward: {reward}, Done: {done}")