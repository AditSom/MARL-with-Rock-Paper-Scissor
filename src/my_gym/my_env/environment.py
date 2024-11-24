import numpy as np  # type: ignore
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .config import Config
import copy

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
                else:
                    pos = self.config.positions[i][j]
                self.positions.append({'agent': i, 'position': pos})

    def reset(self):
        """Reset the environment to its initial state."""
        self.done = False
        self.captured_agents = []
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.positions = []
        self.reward = [0] * self.n_agents
        self.winner = None
        if self.config.animation:
            self.store_positions = []

    def update_grid(self, position=None, print_grid=True):
        """Update the grid with the current positions of the agents."""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        if position is not None:
            for i in range(self.total_agents):
                if i not in self.captured_agents:
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
        pos = self.positions[agent_id]['position']
        if action == 0:  # Move up
            pos[0] = max(0, pos[0] - 1)
        elif action == 1:  # Move down
            pos[0] = min(self.grid_size - 1, pos[0] + 1)
        elif action == 2:  # Move left
            pos[1] = max(0, pos[1] - 1)
        elif action == 3:  # Move right
            pos[1] = min(self.grid_size - 1, pos[1] + 1)
        self.positions[agent_id]['position'] = pos

    def capture_check(self):
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
                print(f"Agent {i + 1} eliminated")

        if self.config.capture:
            if max(agent_count) == self.total_agents:
                self.done = True
                self.winner = agent_count.index(max(agent_count))
                print(f"Agent {self.winner} wins")
        else:
            for i in range(self.n_agents):
                if self.eliminated_record[i]:
                    self.done = True
                    self.winner = self.prey_predator_combo[i]
                    print(f"Agent {self.winner} wins")
                    
    def reward_update(self):
        """Update the rewards for each agent."""
        self.reward = [0] * self.n_agents
        if self.done:
            for i in range(self.n_agents):
                if i == self.winner:
                    self.reward[i] = self.rewards['win']
                else:
                    self.reward[i] = self.rewards['lose']
        else:
            self.reward = (np.array(self.capture_record[0]) * self.rewards['captured'] -
                           np.array(self.capture_record[1]) * self.rewards['eliminated'] -
                           np.array([self.rewards['step']] * self.n_agents))

    def render_entire_episode(self):
        """Render the progression of the episode as an animation and save it as a video."""
        if not self.config.animation or not hasattr(self, "store_positions") or len(self.store_positions) == 0:
            print("No animation data to render. Ensure animation is enabled and steps have been performed.")
            return

        fig, ax = plt.subplots()
        grid_data = np.zeros((self.grid_size, self.grid_size), dtype=int)
        im = ax.imshow(grid_data, cmap="viridis", vmin=0, vmax=self.n_agents + 1)

        def update(frame):
            self.update_grid(self.store_positions[frame], print_grid=False)
            im.set_data(self.grid.copy())
            ax.set_title(f"Step {frame + 1}")
            return [im]

        ani = FuncAnimation(fig, update, frames=len(self.store_positions), interval=100, blit=True)
        output_path = self.config.ani_save_path
        ani.save(output_path, writer="ffmpeg", fps=self.config.fps)
        print(f"Animation saved to {output_path}")

    def plot_grid(self):
        """Plot the final grid at the end of the episode."""
        if not hasattr(self, "grid"):
            print("No grid data to plot. Ensure steps have been performed.")
            return
        cmap = plt.cm.get_cmap('viridis', 4)
        self.update_grid(print_grid=False)
        grid = np.array(self.grid)
        plt.figure(figsize=(self.config.grid_size, self.config.grid_size))
        plt.imshow(grid, cmap=cmap, interpolation='nearest')
        plt.colorbar(ticks=range(self.n_agents + 1), label='Agent Type')
        plt.title('Agent Grid')
        plt.xlabel('X')
        plt.ylabel('Y')
        output_path = "grid_plot.png"
        plt.savefig(output_path)

    def step(self, action_list, timestep):
        """Perform a step in the environment."""
        for i in range(self.total_agents):
            if i not in self.captured_agents:
                self.action_update(action_list[i], i)
        self.capture_check()
        self.reward_update()
        if self.config.animation:
            self.store_positions.append(self.positions[:])
        # if timestep == self.config.time_step:
        #     self.done = True
        if self.config.track_grid is not False and timestep % self.config.track_grid == 0:
            self.update_grid()
        if self.done and self.config.animation:
            self.render_entire_episode()
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