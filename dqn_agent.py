# import torch 
# import torch.nn as nn
import numpy as np 
import retro

def prepro(i):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    i = i[35:195]  # crop
    i = i[::2, ::2, 0]  # downsample by factor of 2
    i[i == 144] = 0  # erase background (background type 1)
    i[i == 109] = 0  # erase background (background type 2)
    i[i != 0] = 1  # everything else (paddles, ball) just set to 1
    return i.astype(np.float).ravel()

def dqn_agent(game_name, render=True):

    env = retro.make(game=game_name)

    obs = env.reset()

    if render:

        env.render()

    done = False
    while not done: # TODO: change this to True

        obs, rew, done, info = env.step(env.action_space.sample())

        env.render()

