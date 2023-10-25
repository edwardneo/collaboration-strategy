import argparse

import gym

import numpy as np
import torch
from gym.wrappers import FlattenObservation
from gym import spaces


# from cs285.infrastructure.logger import Logger



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


    env.reset()
    NUM_ACTIONS = env.action_space.n
    NUM_STATES = spaces.utils.flatdim(env.observation_space)
    Q = np.zeros([NUM_STATES, NUM_ACTIONS]) #You could also make this dynamic if you don't know all games states upfront
    gamma = 0.95 # discount factor
    alpha = 0.01 # learning rate
    epsilon = 0.1 #
    for episode in range(1,500001):
        done = False
        obs = env.reset()
        while done != True:
            if np.random.rand(1) < epsilon:
                # exploration with a new option with probability epsilon, the epsilon greedy approach
                action = env.action_space.sample()
            else:
                # exploitation
                print(obs)
                action = np.argmax(Q[obs])
            obs2, rew, done, info = env.step(action) #take the action
            Q[obs,action] += alpha * (rew + gamma * np.max(Q[obs2]) - Q[obs,action]) #Update Q-marix using Bellman equation
            obs = obs2 
        if episode % 5000 == 0:
        #report every 5000 steps, test 100 games to get avarage point score for statistics and verify if it is solved
            rew_average = 0.
            for i in range(100):
                obs= env.reset()
                done=False
                while done != True: 
                    action = np.argmax(Q[obs])
                    obs, rew, done, info = env.step(action) #take step using selected action
                    rew_average += rew
            rew_average=rew_average/100
            print('Episode {} avarage reward: {}'.format(episode,rew_average))
        
            if rew_average > 0.9:
                # FrozenLake-v0 defines "solving" as getting average reward of 0.78 over 100 consecutive trials.
                # Test it on 0.8 so it is not a one-off lucky shot solving it
                print("Frozen lake solved")
                break
    rew_tot=0.
    obs= env.reset()
    done=False
    while done != True: 
        action = np.argmax(Q[obs])
        obs, rew, done, info = env.step(action) #take step using selected action
        rew_tot += rew
        env.render()

    print("Reward:", rew_tot) 


if __name__ == '__main__':
    run_training_loop()