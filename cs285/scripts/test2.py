import gymnasium as gym
from gymnasium.wrappers import EnvCompatibility
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import argparse

from cs285.envs.maze_game_hidden import MazeGameEnv
#from cs285.networks.mask import TorchActionMaskModel
import warnings
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.evaluation import evaluate_policy




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
    parser.add_argument("--eval", "-e", type=int, default=100)

    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--train_name", type=str, default="training")
    parser.add_argument("--fresh", "-f", action="store_true")

    args = parser.parse_args()



    # create directory for logging
    #logdir_prefix = "hw5_explore_"  # keep for autograder

    #logger = make_logger(logdir_prefix, config)

    ###COMMAND1: python scripts/test2.py -s trained_model -f -t 500000
    ###COMMAND2: python scripts/test2.py -l trained_model -s trained_model2 -t 300000
    ###COMMAND3: python scripts/test2.py -l trained_model2  -t 0 -e 0 -r 
    if args.load:
        name = args.load
        env = FlattenObservation(gym.make('MazeGame-v1', render_mode = "human", fresh_start = args.fresh))
        env = ActionMasker(env, lambda env: env.valid_mask(env.pos, env.board))
        model = MaskablePPO.load(name, env = env, tensorboard_log=args.logdir) #"./trained_model"
    else:
        env = FlattenObservation(gym.make('MazeGame-v1', render_mode = "human", fresh_start = args.fresh))
        env = ActionMasker(env, lambda env: env.valid_mask(env.pos, env.board))
        model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log=args.logdir)

    if args.train:
        model.learn(total_timesteps=args.train,  tb_log_name=args.train_name)
    if args.save:
        model.save(args.save)
    if args.eval:
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.eval)

        # Print the results
        print(f"Mean Reward: {mean_reward:.2f}")
        print(f"Std Reward: {std_reward:.2f}")
    if args.rend:
        vec_env = model.get_env()
        obs = vec_env.reset()
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            vec_env.render()

    env.close()
