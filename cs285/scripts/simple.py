import gymnasium as gym
from gymnasium.wrappers import EnvCompatibility
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import argparse

from cs285.envs.maze_game_simple import MazeGameEnv
import warnings
from gymnasium.wrappers import FlattenObservation
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks




warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == "__main__":
    gym.register(
        id = 'MazeGame-v1',
        entry_point = 'cs285.envs.maze_game_simple:MazeGameEnv'
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

    ###COMMAND1: python scripts/hidden.py -s trained_model -f -t 500000
    ###COMMAND2: python scripts/hidden.py -s trained_model -l trained_model -t 300000

    if args.load:
        name = args.load
        env = FlattenObservation(gym.make('MazeGame-v1', render_mode = "human", fresh_start = args.fresh))
        env = ActionMasker(env, lambda env: env.valid_mask(env.pos, env.board))
        model = MaskablePPO.load(name, env = env, tensorboard_log=args.logdir) 
    else:
        env = FlattenObservation(gym.make('MazeGame-v1', render_mode = "human", fresh_start = args.fresh))
        env = ActionMasker(env, lambda env: env.valid_mask(env.pos, env.board))
        model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log=args.logdir) #cnn policy?  or biggerMLP?

    if args.train:
        model.learn(total_timesteps=args.train,  tb_log_name=args.train_name)
    if args.save:
        model.save(args.save)
    if args.eval:
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.eval) #buggy

        # Print the results
        print(f"Mean Reward: {mean_reward:.2f}")
        print(f"Std Reward: {std_reward:.2f}")
    if args.rend:
        obs, _ = env.reset()
        while True:
            action_masks = get_action_masks(env)
            action, _states = model.predict(obs, action_masks=action_masks)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render() #kinda buggy?
            if terminated:
                obs, _ = env.reset()

    env.close()
