import gymnasium as gym
from gymnasium.wrappers import EnvCompatibility
import argparse

from cs285.envs.maze_game_toy import MazeGameEnvTwoPlayer
import warnings
from gymnasium.wrappers import FlattenObservation
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker, TimeFeatureWrapper



warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == "__main__":
    gym.register(
        id = 'MazeGame-v2',
        entry_point = 'cs285.envs.maze_game_toy:MazeGameEnvTwoPlayer'
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
    parser.add_argument("--cost", "-c", type=float, default=0)
    parser.add_argument("--num_comm_runs", "-n", type=int, default=0)

    args = parser.parse_args()

    print(args.cost)


    ### python scripts/toy.py -p2 trained_model -l trained_model -s trained_model_p1 -t 500 -c 20
    ### python scripts/toy.py -p2 trained_model_p1 -l trained_model_p1 -s trained_model_p2 -t 100000 -c 20
    ### python scripts/toy.py -l trained_model_p2 -t 0 -e 0 -r 
    for _ in range(args.epochs):
        if args.load:
            name = args.load
            env = TimeFeatureWrapper(FlattenObservation(gym.make('MazeGame-v2', save_file = args.player2, render_mode = "human", fresh_start = args.fresh, encourage = args.encourage, cost = args.cost)), max_steps=40)
            env = ActionMasker(env, lambda env: env.valid_mask(env.pos, env.board))
            model = MaskablePPO.load(name, env = env, tensorboard_log=args.logdir) 
        else:
            env = TimeFeatureWrapper(FlattenObservation(gym.make('MazeGame-v2', save_file = args.player2, render_mode = "human", fresh_start = args.fresh, encourage = args.encourage, cost = args.cost)), max_steps=40)
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
        action_list = []
        while True:
            action_masks = get_action_masks(env)
            action, _states = model.predict(obs, action_masks=action_masks)
            if action == 5:
                print(len(action_list))
            action_list.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated:
                obs, _ = env.reset()
                print(action_list)
                action_list = []
    
    if args.num_comm_runs > 0:
        total = 0
        num_comm = 0
        for _ in range(args.num_comm_runs):
            obs, _ = env.reset()
            action_list = []
            did_communicate = False
            terminated = False
            while not terminated:
                action_masks = get_action_masks(env)
                action, _states = model.predict(obs, action_masks=action_masks)
                if action == 5:
                    total += len(action_list)
                    did_communicate = True
                action_list.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                # env.render()
            num_comm += did_communicate
            # print([int(ac) for ac in action_list])
        print(f"Average communication time: {total / num_comm}")
        print(f"Percentage of communication runs: {num_comm / args.num_comm_runs}")

    env.close()
