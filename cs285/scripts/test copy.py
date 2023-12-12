import gymnasium as gym
from gymnasium.wrappers import EnvCompatibility
from cs285.envs.maze_game import MazeGameEnv
from cs285.networks.mask import TorchActionMaskModel
import warnings
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from gymnasium.wrappers import FlattenObservation


warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == "__main__":
    gym.register(
        id = 'MazeGame-v0',
        entry_point = 'cs285.envs.maze_game:MazeGameEnv'
    )


    log_path = os.path.join("logs/", "mazegame", "dqn")
    writer = SummaryWriter(log_path)
    writer.add_text("args", "???")
    logger = TensorboardLogger(writer)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = FlattenObservation(gym.make('MazeGame-v0', render_mode = 'human'))
    train_envs = DummyVectorEnv([lambda: env for _ in range(1)])
    test_envs = DummyVectorEnv([lambda: env for _ in range(1)])
    # model & optimizer
    net = Net(env.observation_space.shape, hidden_sizes=[64, 64], device=device)
    actor = Actor(net, env.action_space.n, device=device).to(device)
    critic = Critic(net, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

    # PPO policy
    dist = torch.distributions.Categorical
    policy = PPOPolicy(actor, critic, optim, dist, action_space=env.action_space, deterministic_eval=True)
            
            
    # collector
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=2,
        step_per_epoch=50000,
        repeat_per_collect=10,
        episode_per_test=10,
        batch_size=256,
        step_per_collect=2000,
        stop_fn=lambda mean_reward: mean_reward >= 400,
        logger = logger
    )
    print("\n" + str(result) + "\n")

    policy.eval()
    result = test_collector.collect(n_episode=1, render=True)
    print("\n"+ "Final reward: {}, length: {}".format(result["rews"].mean(), result["lens"].mean()) + "\n")