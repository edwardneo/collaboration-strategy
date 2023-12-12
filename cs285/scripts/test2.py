import gymnasium as gym
from gymnasium.wrappers import EnvCompatibility
from stable_baselines3 import PPO

from cs285.envs.maze_game_hidden import MazeGameEnv
from cs285.networks.mask import TorchActionMaskModel
import warnings
from gymnasium.wrappers import FlattenObservation


warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == "__main__":
    gym.register(
        id = 'MazeGame-v1',
        entry_point = 'cs285.envs.maze_game_hidden:MazeGameEnv'
    )

    load = True
    if load:
        env = FlattenObservation(gym.make('MazeGame-v1', render_mode = "human"))

        model = PPO.load("./trained_model", env = env)
    else:
        env = FlattenObservation(gym.make('MazeGame-v1', render_mode = "human"))
        model = PPO("MlpPolicy", env, verbose=1)
    train = True
    if train:
        model.learn(total_timesteps=100000)
    save = False
    if save:
        model.save("./trained_model")
    rend = False
    if rend:
        vec_env = model.get_env()
        obs = vec_env.reset()
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            vec_env.render()    

    env.close()
