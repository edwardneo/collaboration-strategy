import argparse

import gym

import numpy as np
import torch
from gym.wrappers import FlattenObservation
from gym import spaces


# from cs285.infrastructure.logger import Logger

import tensorflow
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
    '''tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(
        log_dir="data/", 
        histogram_freq=1,  # How often to log histograms
        write_graph=True,   # Whether to log the graph
        write_images=True,  # Whether to log images
        update_freq='epoch' # Frequency for scalar logs
    )'''

    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps=5000, visualize=False, verbose=1)

    #results = dqn.test(env, nb_episodes=5, visualize=False, callbacks=[tensorboard_callback])
    #print(results.history)
    log_dir = "data/"
    summary_writer = tensorflow.summary.create_file_writer(log_dir)

    # Your training or evaluation loop
    for episode in range(5):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Perform an action and receive the next state and reward
            action = model(state.reshape(1,-1))
            next_state, reward, done, info = env.step(action[0])
            # Log the reward as a custom scalar
            with summary_writer.as_default():
                [tf.summary.scalar("bag" + str(i), info["bag" + str(i)], step=global_step) for i in range(4)]

            state = next_state
            global_step += 1
    summary_writer.close()

            
def build_model(states, actions):
    model = Sequential()    
    model.add(Dense(480, activation='relu', input_shape=(1,states)))
    model.add(Dense(480, activation='relu'))
    model.add(Flatten())
    model.add(Dense(actions, activation='linear'))
    return model

    #other architectures? GNN? CNN? 
if __name__ == '__main__':
    run_training_loop()