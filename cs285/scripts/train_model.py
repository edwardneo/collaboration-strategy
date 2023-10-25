import argparse

import gym

import numpy as np
import torch
from gym.wrappers import FlattenObservation
from gym import spaces


# from cs285.infrastructure.logger import Logger

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

def run_training_loop():
    # set random seeds
    # make the gym environment
    gym.register(
        id = 'MazeGame-v0',
        entry_point = 'cs285.envs.maze_game:MazeGameEnv', 
        kwargs = {
            'board': None,
            'goal': None,
            'playerPosition': None
        }
    )
    
    # Test the environment
    board = np.array([
        [-1, -1, -1,  3, -1, -1],
        [-1,  1,  1,  1,  1, -1],
        [-1,  0,  0, -1,  0, -1],
        [ 1, -1,  1, -1,  2, -1],
        [-1,  0, -1, -1, -1,  2],
        [-1,  2,  1,  0, -1, -1],
    ])
    goal = np.array([4,3,2,1])
    playerPosition = np.array([0,0])
    
    env = FlattenObservation(gym.make('MazeGame-v0', board=board, goal=goal, playerPosition=playerPosition))

    obs_shape = spaces.utils.flatdim(env.observation_space)
    actions = env.action_space.n

    model = build_model(obs_shape, actions)

    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=5e-3), metrics=['mae'])
    dqn.fit(env, nb_steps=60000, visualize=False, verbose=1)

    results = dqn.test(env, nb_episodes=150, visualize=False)
    print(np.mean(results.history['episode_reward']))
def build_model(states, actions):
    model = Sequential()    
    model.add(Dense(24, activation='relu', input_shape=(1,states)))
    model.add(Dense(24, activation='relu'))
    model.add(Flatten())
    model.add(Dense(actions, activation='linear'))
    return model
if __name__ == '__main__':
    run_training_loop()