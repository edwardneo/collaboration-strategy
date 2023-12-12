import gymnasium as gym
from gymnasium.wrappers import EnvCompatibility
from stable_baselines3 import PPO

from cs285.envs.maze_game import MazeGameEnv
from cs285.networks.mask import TorchActionMaskModel
import warnings
from gymnasium.wrappers import FlattenObservation


warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == "__main__":
    gym.register(
        id = 'MazeGame-v0',
        entry_point = 'cs285.envs.maze_game:MazeGameEnv'
    )

    env = FlattenObservation(gym.make('MazeGame-v0'))
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_00000)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        #vec_env.render()    

    env.close()
