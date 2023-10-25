import time
import argparse

from cs285.agents.dqn_agent import DQNAgent
import cs285.env_configs

import os
import time

import gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.replay_buffer import MemoryEfficientReplayBuffer, ReplayBuffer
from cs285.envs.maze_game import MazeGameEnv
from scripting_utils import make_logger, make_config
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers import FlattenObservation
from gym import spaces


MAX_NVIDEO = 2


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    gym.register(
        id = 'MazeGame-v0',
        entry_point = 'cs285.envs.maze_game:MazeGameEnv',
        max_episode_steps=75,
        kwargs = {
            'board': None,
            'goal': None,
            'playerPosition': None
        }
    )
    
    # Test the environment
    board = np.array([
        [-1, 0, 0,  3, -1, -1],
        [0,  0,  0,  1,  1, -1],
        [0,  0,  0, -1,  0, -1],
        [ 1, -1,  1, -1,  2, -1],
        [-1,  0, -1, -1, -1,  2],
        [-1,  2,  1,  0, -1, -1],
    ])
    goal = np.array([1,0,0,0])
    playerPosition = np.array([0,0])
    
    env = FlattenObservation(RecordEpisodeStatistics(gym.make('MazeGame-v0', board=board, goal=goal, playerPosition=playerPosition)))

    eval_env = FlattenObservation(RecordEpisodeStatistics(gym.make('MazeGame-v0', board=board, goal=goal, playerPosition=playerPosition)))
    render_env = FlattenObservation(RecordEpisodeStatistics(gym.make('MazeGame-v0', board=board, goal=goal, playerPosition=playerPosition, render_mode="rgb_array")))
    exploration_schedule = config["exploration_schedule"]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    assert discrete, "DQN only supports discrete action spaces"

    print(spaces.utils.flatdim(env.observation_space))

    agent = DQNAgent(
        spaces.utils.flatdim(env.observation_space),
        env.action_space.n,
        **config["agent_kwargs"],
    )

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    elif "render_fps" in env.env.metadata:
        fps = env.env.metadata["render_fps"]
    else:
        fps = 4

    ep_len = env.spec.max_episode_steps

    observation = None

    ##TODO: are we using replay buffer? either way, the shapes should be looked at 

    # Replay buffer
    replay_buffer = ReplayBuffer()
    # if len(env.observation_space.shape) == 3:
    #     stacked_frames = True
    #     frame_history_len = env.observation_space.shape[0]
    #     assert frame_history_len == 4, "only support 4 stacked frames"
    #     replay_buffer = MemoryEfficientReplayBuffer(
    #         frame_history_len=frame_history_len
    #     )
    # elif len(env.observation_space.shape) == 1:
    #     stacked_frames = False
    #     replay_buffer = ReplayBuffer()
    # else:
    #     raise ValueError(
    #         f"Unsupported observation space shape: {env.observation_space.shape}"
    #     )

    def reset_env_training():
        nonlocal observation

        observation = env.reset()

        assert not isinstance(
            observation, tuple
        ), "env.reset() must return np.ndarray - make sure your Gym version uses the old step API"
        observation = np.asarray(observation)

        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            replay_buffer.on_reset(observation=observation[-1, ...])

    reset_env_training()

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        epsilon = exploration_schedule.value(step)
        
        # TODO(student): Compute action
        action = agent.get_action(observation, epsilon)

        # TODO(student): Step the environment
        next_observation, rew, done, info = env.step(action)

        next_observation = np.asarray(next_observation)
        truncated = info.get("TimeLimit.truncated", False)

        # TODO(student): Add the data to the replay buffer
        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            # We're using the memory-efficient replay buffer,
            # so we only insert next_observation (not observation)
            replay_buffer.insert(action, rew, next_observation[-1], done and not truncated)
        else:
            # We're using the regular replay buffer
            replay_buffer.insert(observation, action, rew, next_observation, done and not truncated)

        # Handle episode termination
        if done:
            reset_env_training()

            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
        else:
            observation = next_observation

        # Main DQN training loop
        if step >= config["learning_starts"]:
            # TODO are using a replay buffer or not?
            batch = replay_buffer.sample(config["batch_size"])

            # Convert to PyTorch tensors
            batch = ptu.from_numpy(batch)

            # TODO(student): Train the agent. `batch` is a dictionary of numpy arrays,
            update_info = agent.update(
                obs=batch['observations'],
                action=batch['actions'],
                reward=batch['rewards'],
                next_obs=batch['next_observations'],
                done=batch['dones'],
                step=step
            )

            # Logging code
            update_info["epsilon"] = epsilon
            update_info["lr"] = agent.lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                logger.flush()

        if step % args.eval_interval == 0:
            # Evaluate
            trajectories = utils.sample_n_trajectories(
                eval_env,
                agent,
                args.num_eval_trajectories,
                ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            if args.num_render_trajectories > 0:
                video_trajectories = utils.sample_n_trajectories(
                    render_env,
                    agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    step,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "hw3_dqn_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
