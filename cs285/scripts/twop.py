import gymnasium as gym
from gymnasium.wrappers import EnvCompatibility
import argparse

from cs285.envs.maze_game_two_player import MazeGameEnvTwoPlayer
import warnings
from gymnasium.wrappers import FlattenObservation
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker



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
    parser.add_argument("--encourage", "-enc", action="store_true")
    parser.add_argument("--epochs", "-ep",type=int, default=1)

    args = parser.parse_args()


    ### COMMAND0: python scripts/twop.py -p2 trained_model2 -s trained_model_p1 -enc -f -t 10000
    ### COMMAND1: python scripts/twop.py -p2 trained_model_p1 -l trained_model_p1 -s trained_model_p1 -enc -f -t 30000
    ### COMMAND2: python scripts/twop.py -p2 trained_model_p1 -l trained_model_p1 -s trained_model_p2 -enc -t 30000
    ### COMMAND3: python scripts/twop.py -p2 trained_model_p2 -l trained_model_p2 -s trained_model_p2 -t 10000 -ep 5
    ### run COMMAND3 multiple times
    ### python scripts/twop.py -l trained_model_p2 -p2 trained_model2  -t 0 -e 0 -r 
    ###how to motivate communication and speed? maybe the goals have to be different? idk
    for _ in range(args.epochs):
        if args.load:
            name = args.load
            env = FlattenObservation(gym.make('MazeGame-v2', save_file = args.player2, render_mode = "human", fresh_start = args.fresh, encourage = args.encourage))
            env = ActionMasker(env, lambda env: env.valid_mask(env.pos, env.board))
            model = MaskablePPO.load(name, env = env, tensorboard_log=args.logdir) 
        else:
            env = FlattenObservation(gym.make('MazeGame-v2', save_file = args.player2, render_mode = "human", fresh_start = args.fresh, encourage = args.encourage))
            env = ActionMasker(env, lambda env: env.valid_mask(env.pos, env.board))
            model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log=args.logdir) #cnn policy? 
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
            env.render()
            if terminated:
                obs, _ = env.reset()

    env.close()
