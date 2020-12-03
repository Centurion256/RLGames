import numpy as np
import retro
from utils import sigmoid
import pickle
import os
import time

INF = float('inf')

up = [1, 0, 0, 0, 1, 0, 0, 0]
down = [1, 0, 0, 0, 0, 1, 0, 0]

H = 200
batch_size = 10
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99
resume = False
render = False

D = 80 * 80


def prepro(i):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    i = i[35:195]  # crop
    i = i[::2, ::2, 0]  # downsample by factor of 2
    i[i == 144] = 0  # erase background (background type 1)
    i[i == 109] = 0  # erase background (background type 2)
    i[i != 0] = 1  # everything else (paddles, ball) just set to 1
    return i.astype(np.float).ravel()


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(model, x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h


def policy_backward(model, eph, epdlogp, epx):
    """ backward pass. (eph is array of intermediate hidden states) """
    dw2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0
    dw1 = np.dot(dh.T, epx)
    return {'W1': dw1, 'W2': dw2}


def test_agent_play(game_name, model):
    pass


def test_agent(game_name):
    model = {'W1': np.random.randn(H, D) / np.sqrt(D * H), 'W2': np.random.randn(H) / np.sqrt(H)}
    if os.path.exists('pg_model.pkl'):
        print("model loaded from file")
        f = open('pg_model.pkl', 'rb')
        model = pickle.load(f)
        f.close()

    grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
    rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}

    env = retro.make(game="Pong-Atari2600")
    observation = env.reset()
    prev_x = None
    xs, hs, dlogps, drs = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0
    rounds_won = 0

    while True:
        # env.render()
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        aprob, h = policy_forward(model, x)

        action = up if np.random.uniform() < aprob else down
        xs.append(x.flatten())
        hs.append(h.flatten())
        y = 1 if action == up else 0

        dlogps.append(y - aprob)

        observation, rew, done, info = env.step(action)

        reward_sum += rew
        drs.append(rew)

        if done:
            if episode_number % 50 == 0:
                print(f"Episode {episode_number}; Rounds won: ", rounds_won)
                rounds_won = 0

            episode_number += 1
            if episode_number % 100 == 0:
                f = open('pg_model.pkl', 'wb')
                pickle.dump(model, f)
                f.close()

            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs, hs, dlogps, drs = [], [], [], []  # reset array memory

            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            epdlogp *= discounted_epr
            grad = policy_backward(model, eph, epdlogp, epx)
            for k in model:
                grad_buffer[k] += grad[k]

            if episode_number % batch_size == 0:
                print("New episode ", episode_number)
                for k, v in model.items():
                    g = grad_buffer[k]
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = np.zeros_like(v)
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            reward_sum = 0
            observation = env.reset()
            prev_x = None
