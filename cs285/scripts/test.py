import gymnasium as gym
from gymnasium.wrappers import EnvCompatibility
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from cs285.envs.maze_game import MazeGameEnv
from cs285.networks.mask import TorchActionMaskModel
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == "__main__":
    gym.register(
        id = 'MazeGame-v0',
        entry_point = 'cs285.envs.maze_game:MazeGameEnv'
    )

    ModelCatalog.register_custom_model("mask_model", TorchActionMaskModel)

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

    for i in range(3):
        print(f'''\n iteration {i}: ''' + str(algo.train()) + "\n")  # 3. train it,

    evaluation_results = algo.evaluate()  # 4. and evaluate it.
    print("Evaluation Results:", evaluation_results)
