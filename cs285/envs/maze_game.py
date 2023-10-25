import gym
from gym import spaces
import numpy as np
import pygame

NUM_DISTINCT_ITEMS = 4
COLORS = ['red', 'blue', 'green', 'pink']

class OneHotEncoding(gym.Space):
    """
    {0,...,1,...,0}

    Example usage:
    self.observation_space = OneHotEncoding(size=4)
    """
    def __init__(self, size=None):
        assert isinstance(size, int) and size > 0
        self.size = size
        gym.Space.__init__(self, (), np.int64)

    def sample(self):
        one_hot_vector = np.zeros(self.size)
        one_hot_vector[np.random.randint(self.size)] = 1
        return one_hot_vector

    def contains(self, x):
        if isinstance(x, (list, tuple, np.ndarray)):
            number_of_zeros = list(x).contains(0)
            number_of_ones = list(x).contains(1)
            return (number_of_zeros == (self.size - 1)) and (number_of_ones == 1)
        else:
            return False

    def __repr__(self):
        return "OneHotEncoding(%d)" % self.size

    def __eq__(self, other):
        return self.size == other.size

class MazeGameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, board, goal, playerPosition, render_mode=None):
        assert goal.size == NUM_DISTINCT_ITEMS, f"Goal is not of size ${NUM_DISTINCT_ITEMS}"

        self.start_params = {
            'board': board,
            'playerPosition': playerPosition,
        }

        super(MazeGameEnv, self).__init__()
        self.board = np.array(board)  # Maze represented as a 2D NumPy array
        self.goal = np.array(goal) # Goal represented as a 1D NumPy array
        self.bag = np.array([0] * NUM_DISTINCT_ITEMS) # Bag represented as a 1D NumPy array
        self.pos = playerPosition # Starting position is current posiiton of agent
        assert self.goal.shape == self.bag.shape

        self.num_rows, self.num_cols = self.board.shape

        # 5 possible actions: 0 = Up, 1 = Down, 2 = Left, 3 = Right, 4 = Collect
        self.action_space = spaces.Discrete(5)

        # Observation space is grid of size:rows x columns
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=-1, high=NUM_DISTINCT_ITEMS-1, shape=board.shape[0:2], dtype=int),
            'bag': spaces.Box(low=0, high=20, shape=(NUM_DISTINCT_ITEMS,)),
            'pos': spaces.Tuple((spaces.Discrete(self.num_rows), spaces.Discrete(self.num_cols))),
        })

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize Pygame
        pygame.init()
        self.cell_size = 125

        # setting display size
        self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

    def reset(self):
        super(MazeGameEnv, self).reset()

        self.board = np.array(self.start_params['board'])  # Maze represented as a 2D NumPy array
        self.bag = np.array([0] * NUM_DISTINCT_ITEMS) # Bag represented as a 1D NumPy array
        self.pos = self.start_params['playerPosition'] # Starting position is current posiiton of agent

        return self._generate_observation()

    def step(self, action):
        is_legal = False

        # Move the agent based on the selected action
        new_board = np.array(self.board)
        new_pos = np.array(self.pos)
        new_bag = np.array(self.bag)

        trigger = False

        if 0 <= action <= 3:
            if action == 0:  # Up
                new_pos[0] -= 1
            elif action == 1:  # Down
                new_pos[0] += 1
            elif action == 2:  # Left
                new_pos[1] -= 1
            elif action == 3:  # Right
                new_pos[1] += 1

            # Check if the new position is valid
            if self._is_valid_position(new_pos):
                self.pos = new_pos
                is_legal = True

        elif action == 4:
            item = new_board[new_pos[0], new_pos[1]]
            
            if 0 <= item <= NUM_DISTINCT_ITEMS:
                new_bag[item] += 1
                trigger = True
                new_board[new_pos[0], new_pos[1]] = -1
                self.bag = new_bag
                self.board = new_board
                is_legal = True

        # Reward function
        if np.array_equal(self.bag, self.goal):
            reward = 1.0
            done = True
        elif not is_legal:
            reward = -1
            done = False
        else:
            reward = -0.01
            done = False
        reward += 0.01*trigger

        return self._generate_observation(), reward, done, {"bag" + str(i): self.bag[i] for i in range(4)}

    def _is_valid_position(self, pos):
        row, col = pos
   
        # If agent goes out of the grid
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False
        return True
    
    def _generate_observation(self):
        return {
            'board': self.board,
            'bag': self.bag,
            'pos': self.pos
        }

    def render(self):
        if self.render_mode == "rgb_array":
            # Clear the screen
            self.screen.fill((255, 255, 255))  

            # Draw env elements one cell at a time
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    cell_left = col * self.cell_size
                    cell_top = row * self.cell_size
                
                    try:
                        print(np.array(self.pos) == np.array([row,col]).reshape(-1,1))
                    except Exception as e:
                        print('Initial state')

                    if self.maze[row, col] == -1:  # Blank
                        pygame.draw.rect(self.screen, pygame.Color('white'), (cell_left, cell_top, self.cell_size, self.cell_size))
                    else: # Item
                        pygame.draw.rect(self.screen, pygame.Color(COLORS[self.maze[row, col]]), (cell_left, cell_top, self.cell_size, self.cell_size))
                    
                    if np.array_equal(np.array(self.pos), np.array([row, col]).reshape(-1,1)): # Agent
                        pygame.draw.rect(self.screen, pygame.Color('black'), (cell_left + 20, cell_top + 20, self.cell_size - 40, self.cell_size - 40))

            pygame.display.update()  # Update the display