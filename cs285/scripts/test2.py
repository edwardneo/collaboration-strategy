import gymnasium as gym
from gymnasium.wrappers import EnvCompatibility
from stable_baselines3 import PPO
import argparse

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", "-l", type=str, required=False)
    parser.add_argument("--save", "-s", type=str, required=False)
    parser.add_argument("--rend", "-r", action="store_true")


    parser.add_argument("--train", "-t", type=int, default=100000)
    parser.add_argument("--log_interval", type=int, default=1000)
    args = parser.parse_args()

    # create directory for logging
    #logdir_prefix = "hw5_explore_"  # keep for autograder

    #logger = make_logger(logdir_prefix, config)

    ###COMMAND: python scripts/test2.py -l trained_model -s trained_model2  
    if args.load:
        name = args.load
        env = FlattenObservation(gym.make('MazeGame-v1', render_mode = "human"))

        model = PPO.load(name, env = env) #"./trained_model"
    else:
        env = FlattenObservation(gym.make('MazeGame-v1', render_mode = "human"))
        model = PPO("MlpPolicy", env, verbose=1)
    if args.train:
        model.learn(total_timesteps=args.train)
    if args.save:
        model.save(args.save)
    if args.rend:
        vec_env = model.get_env()
        obs = vec_env.reset()
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            vec_env.render()    

    env.close()
