import gymnasium as gym
from gymnasium.wrappers import EnvCompatibility
from stable_baselines3 import PPO
import argparse

from cs285.envs.maze_game_two_player import MazeGameEnvTwoPlayer
#from cs285.networks.mask import TorchActionMaskModel
import warnings
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.evaluation import evaluate_policy




warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == "__main__":
    gym.register(
        id = 'MazeGame-v2',
        entry_point = 'cs285.envs.maze_game_two_player:MazeGameEnvTwoPlayer'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", "-l", type=str, required=False)
    parser.add_argument("--player2", "-p2", type=str, required=True)
    parser.add_argument("--save", "-s", type=str, required=False)
    parser.add_argument("--rend", "-r", action="store_true")


    parser.add_argument("--train", "-t", type=int, default=100000)
    parser.add_argument("--eval", "-e", type=int, default=100)

    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--train_name", type=str, default="two_player_training")
    parser.add_argument("--fresh", "-f", action="store_true")

    args = parser.parse_args()



    # create directory for logging
    #logdir_prefix = "hw5_explore_"  # keep for autograder

    #logger = make_logger(logdir_prefix, config)

    ###COMMAND1: python scripts/test3.py -p2 trained_model2 -l trained_model2 -s trained_model_p1 
    ### or ###COMMAND1: python scripts/test3.py -p2 trained_model2 -s trained_model_p1 -f 
    ###and COMMAND2: python scripts/test3.py -p2 trained_model_p1 -l trained_model_p1 -s trained_model_p2
    if args.load:
        name = args.load
        env = FlattenObservation(gym.make('MazeGame-v2', save_file = args.player2, render_mode = "human", fresh_start = args.fresh))

        model = PPO.load(name, env = env, tensorboard_log=args.logdir) #"./trained_model"
    else:
        env = FlattenObservation(gym.make('MazeGame-v2', save_file = args.player2, render_mode = "human", fresh_start = args.fresh))
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=args.logdir)
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