import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
import numpy as np
from sb3_contrib import MaskablePPO


BOARD = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1]
    ]
)

GOAL = np.array([3, 3, 3, 3])
PLAYER_POSITION = (0,0)
SIM_PLAYER_POSITION = (5, 5)
COLORS = ["red", "blue", "green", "pink"]

WINDOW_SIZE = (600, 600)


class MazeGameEnvTwoPlayer(gym.Env):
    metadata = {"render_modes": [ "human", "ansi", "rgb_array"], "render_fps": 1}

    def __init__(self, save_file, board=BOARD, goal=GOAL, pos=PLAYER_POSITION, sim_pos=SIM_PLAYER_POSITION, render_mode=None, max_steps = 40, fresh_start = True, encourage = False):
        super(MazeGameEnvTwoPlayer, self).__init__()

        # Save initial parameters
        self.initial_parameters = {"pos": pos, "sim_pos": sim_pos}

        # Initialize env parameters
        self.board = np.array(board)  # Maze represented as a 2D NumPy array
        self.goal = np.array(goal)  # Goal represented as a 1D NumPy array
        self.pos = (pos[0], pos[1])  # Starting position is current posiiton of agent

        self.encourage = encourage


        self.max_steps = max_steps
        self.curr_steps = 0
        self.total_steps = 0
        self.vis_size = 1

        self.fresh_start = fresh_start
      
        self.num_rows, self.num_cols = self.board.shape
        self.num_distinct_items = np.max(self.board) + 1
        self.total_colors = np.array(
            [np.count_nonzero(self.board == i) for i in range(self.num_distinct_items)]
        )

        self.bag = np.array(
            [0] * self.num_distinct_items
        )  # Bag represented as a 1D NumPy array
        self.sim_bag_estimate = self.bag = np.array(
            [0] * self.num_distinct_items
        ) 



        # Assertion checks
        assert (
            self.goal.size == self.num_distinct_items
        ), f"Goal is not of size {self.num_distinct_items}"
        assert self.num_distinct_items <= len(
            COLORS
        ), f"Not enough colors in {COLORS}: need at least {self.num_distinct_items}"
        assert self.goal.shape == self.bag.shape

        # Simulated agent
        self.save_file = save_file
        self.sim_agent = MaskablePPO.load(self.save_file)
        self.sim_pos = sim_pos
        self.sim_bag = np.array(
            [0] * self.num_distinct_items
        )  # Bag represented as a 1D NumPy array
        self.bag_estimate = np.array(
            [0] * self.num_distinct_items
        ) 

        # 6 possible actions: 0 = Up, 1 = Down, 2 = Left, 3 = Right, 4 = Collect, 5 = Information
        self.action_space = spaces.Discrete(6)

        # Can observe: vision, bag, position
        self.observation_space = spaces.Dict(
            {
                "vision": spaces.Box(
                    low=-1,
                    high=self.num_distinct_items - 1,
                    shape=(self.vis_size*2 + 1, self.vis_size*2 + 1),
                    dtype=int,
                ),
                "bag": spaces.Box(low=0, high=20, shape=(self.num_distinct_items,), dtype=int),
                "pos": spaces.Tuple(
                    (spaces.Discrete(self.num_rows), spaces.Discrete(self.num_cols))
                )
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"], f"{render_mode} is not a valid render mode"
        self.render_mode = render_mode

        # Pygame
        self.window = None
        self.clock = None
        self.cell_size = (
            WINDOW_SIZE[0] / self.num_rows,
            WINDOW_SIZE[1] / self.num_cols,
        )
        self.even_timestep = True

        self.reset()

    def set_render_mode(self, mode):
        self.render_mode = mode
        
    def random_state(self):
        flattened = BOARD.flatten()
        np.random.shuffle(flattened)
        random_board = flattened.reshape((self.num_rows, self.num_cols))

        full_color = np.random.randint(self.num_distinct_items)
        random_bag = np.array([0] * self.num_distinct_items)
        random_bag[full_color] = 3

        sim_full_color = np.random.randint(self.num_distinct_items)
        random_sim_bag = np.array([0] * self.num_distinct_items)
        random_sim_bag[sim_full_color] = 3

        return random_board, random_bag, random_sim_bag

    def reset(self, seed=None, options=None):
        super(MazeGameEnvTwoPlayer, self).reset()
        self.curr_steps = 0
        board, bag, sim_bag = self.random_state()

        self.board = np.array(
            board
            #self.initial_parameters["board"]
        )  # Maze represented as a 2D NumPy array
        self.pos = np.array(
            self.initial_parameters["pos"]
        )  # Starting position is current posiiton of agent

        self.bag = np.array(bag)  # Bag represented as a 1D NumPy array
        self.bag_estimate = np.array(
            [0] * self.num_distinct_items
        )

        self.sim_agent = MaskablePPO.load(self.save_file)
        self.sim_pos = np.array(
            self.initial_parameters["sim_pos"]
        )  # Starting position is current posiiton of simulated agent
        self.sim_bag = np.array(sim_bag)  # Bag represented as a 1D NumPy array
        self.sim_bag_estimate = np.array(
            [0] * self.num_distinct_items
        )

        self.even_timestep = True

        return self._generate_observation(self.pos, self.sim_pos, self.bag, self.sim_bag_estimate), {}
    
    def _update(self, pos, bag, action):
        is_legal = False
        collect = False

        # Move the agent based on the selected action
        new_board = np.array(self.board)
        new_r, new_c = pos
        new_bag = np.array(bag)

        if 0 <= action <= 3:
            if action == 0:  # Up
                new_r = max(pos[0]-1, 0)
            elif action == 1:  # Down
                new_r = min(pos[0] + 1, self.num_rows - 1)
            elif action == 2:  # Left
                new_c = max(pos[1]-1, 0)
            elif action == 3:  # Right
                new_c = min(pos[1]+1, self.num_cols - 1)

            is_legal = pos[0] != new_r or pos[1] != new_c

            # self.pos = (new_r, new_c)

        elif action == 4:
            item = new_board[new_r, new_c]

            if 0 <= item < self.num_distinct_items:
                new_bag[item] += 1
                new_board[new_r, new_c] = -1

                collect = self.bag[item] + self.sim_bag_estimate[item] < self.goal[item]

                # self.bag = new_bag
                # self.board = new_board
                
                is_legal = True
        elif action == 5:
            is_legal = True
        
        return new_board, (new_r, new_c), new_bag, is_legal, collect

    def step(self, action):
        self.curr_steps += 1
        
        self.board, self.pos, self.bag, is_legal, collect = self._update(self.pos, self.bag, action)

        # Reward function
        if np.all(self.bag + self.sim_bag >= self.goal):
            reward = 0 
            if self.fresh_start:
                reward = 500
            done = True
        elif not is_legal:
            reward = -250
            done = True
        elif collect:
            reward = -0.9 #-0.9?
            if self.fresh_start:
                reward = 10
            done = False
        else:
            reward = -1
            done = False
        if self.curr_steps > self.max_steps:
            reward = -250
            done = True
        if action == 5:
            self.sim_bag_estimate = self.sim_bag
            if self.encourage:
                reward += 0.2 + 2*self.fresh_start
        
        if not done:
            sim_obs = self._generate_observation(self.sim_pos, self.pos, self.sim_bag, self.bag_estimate)
            action_masks = self.valid_mask(self.sim_pos, self.board, can_communicate=False)
            sim_ac, _states = self.sim_agent.predict(spaces.utils.flatten(self.observation_space, sim_obs), action_masks=action_masks)
            if sim_ac == 5:
                self.bag_estimate = self.bag
            self.board, self.sim_pos, self.sim_bag, _, _ = self._update(self.sim_pos, self.sim_bag, sim_ac)

        # Action mask
        mask = self.valid_mask(self.pos, self.board)

        truncated = done
        info = {"action_mask": mask} | {"bag" + str(i): self.bag[i] for i in range(self.num_distinct_items)}

        return self._generate_observation(self.pos, self.sim_pos, self.bag, self.sim_bag_estimate), reward, done, truncated, info

    def valid_mask(self, curr_pos, board, can_communicate=True):
        row, col = curr_pos
        mask = np.zeros(6, dtype=bool)

        # If agent goes out of the grid
        if row > 0:
            mask[0] = 1
        if row < self.num_rows - 1:
            mask[1] = 1
        if col > 0:
            mask[2] = 1
        if col < self.num_cols - 1:
            mask[3] = 1
        if board[row, col] != -1:
            mask[4] = 1
<<<<<<< HEAD
        mask[5] = 1
=======
        if can_communicate:
            mask[5] = 1
>>>>>>> 14b11f366593cb9d5e1b583fff1567e54270aacd
        return mask

    def _generate_observation(self, pos, other_pos, bag, other_bag):
        big_board = np.zeros((self.num_rows+2, self.num_cols+2))
        big_board[1:-1, 1:-1] = self.board
        big_board[1 + other_pos[0], 1 + other_pos[1]] = -2 
        vis = big_board[pos[0]:pos[0]+3, pos[1]: pos[1]+3]
        return {"vision": vis, "bag": bag + other_bag, "pos": pos}

    def render(self, mode="human"):
        if mode == "ansi":
            return self._render_text()
        else:
            return self._render_gui(mode)

    def _render_gui(self, mode):
        try:
            import pygame  # dependency to pygame only if rendering with human
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install pygame`"
            )

        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Maze Game")
            if mode == "human":
                self.window = pygame.display.set_mode(WINDOW_SIZE)
            else:  # "rgb_array"
                self.window = pygame.Surface(WINDOW_SIZE)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Clear the screen
        self.window.fill((255, 255, 255))

        # Draw env elements one cell at a time
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell = col * self.cell_size[0], row * self.cell_size[1]

                if self.board[row, col] != -1: # Item
                    pygame.draw.rect(
                        self.window,
                        pygame.Color(COLORS[self.board[row, col]]),
                        pygame.Rect(
                            cell[0], cell[1], self.cell_size[0], self.cell_size[1]
                        ),
                    )

                if np.array_equal(np.array(self.pos), np.array([row, col])):  # Agent
                    if self.even_timestep:
                        pygame.draw.circle(
                            self.window,
                            pygame.Color("black"),
                            (cell[0] + self.cell_size[0] / 2, cell[1] + self.cell_size[1] / 2),
                            30
                        )
                    else:
                        pygame.draw.circle(
                            self.window,
                            pygame.Color("black"),
                            (cell[0] + self.cell_size[0] / 2, cell[1] + self.cell_size[1] / 2),
                            30,
                            width=5
                        )
                
                if np.array_equal(np.array(self.sim_pos), np.array([row, col])):  # Simulated Agent
                    if self.even_timestep:
                        pygame.draw.circle(
                            self.window,
                            pygame.Color("grey"),
                            (cell[0] + self.cell_size[0] / 2, cell[1] + self.cell_size[1] / 2),
                            20
                        )
                    else:
                        pygame.draw.circle(
                            self.window,
                            pygame.Color("grey"),
                            (cell[0] + self.cell_size[0] / 2, cell[1] + self.cell_size[1] / 2),
                            20,
                            width=5
                        )
        
        # Draw grid lines
        for row in range(0, self.num_rows + 1):
            pygame.draw.line(
                self.window,
                pygame.Color("black"),
                (0, row * self.cell_size[1]),
                (WINDOW_SIZE[0], row * self.cell_size[1]),
                width=3
            )
        
        for col in range(0, self.num_cols + 1):
            pygame.draw.line(
                self.window,
                pygame.Color("black"),
                (col * self.cell_size[0], 0),
                (col * self.cell_size[0], WINDOW_SIZE[1]),
                width=2
            )
        
        # Flip timestep boolean
        self.even_timestep = not self.even_timestep

        if mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def _render_text(self):
        raise NotImplementedError("Text rendering has not been implemented")
