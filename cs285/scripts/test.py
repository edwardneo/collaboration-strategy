import gymnasium as gym
from gymnasium.wrappers import EnvCompatibility
from ray.rllib.algorithms.ppo import PPOConfig
from cs285.envs.maze_game import MazeGameEnv

if __name__ == "__main__":
    gym.register(
        id = 'MazeGame-v0',
        entry_point = 'cs285.envs.maze_game:MazeGameEnv',
        max_episode_steps=75,
    )

    # env = gym.make('MazeGame-v0')
    config = (  # 1. Configure the algorithm,
        PPOConfig()
        .rollouts(num_rollout_workers=2)
        .environment(MazeGameEnv)
        .framework("torch")
        .training(model={"fcnet_hiddens": [64, 64]})
        .evaluation(evaluation_num_workers=1)
    )

    algo = config.build()  # 2. build the algorithm,

    for _ in range(5):
        print(algo.train())  # 3. train it,

    algo.evaluate()  # 4. and evaluate it.