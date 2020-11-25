import retro
import sys
from sklearn.neural_network import MLPRegressor
import random
import numpy as np
import torch
import pickle

INF = float('inf')


def list_games():
    return retro.data.list_games()


up = [1, 0, 0, 0, 1, 0, 0, 0]
down = [1, 0, 0, 0, 0, 1, 0, 0]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * discount + r[t]
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





def test_agent(game_name):
    model = {'W1': np.random.randn(H, D) / np.sqrt(D * H), 'W2': np.random.randn(H) / np.sqrt(H)}
    # f = open('model', 'rb')
    # model = pickle.load(f)
    # f.close()

    grad_buffer = {k: np.zeros_like(v) for k, v in
                   model.items()}
    rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}

    env = retro.make(game=game_name)
    obs = env.reset()
    three_last_obs = np.ndarray(shape=(3, 4), dtype='float64')
    xs, hs, dlogps, drs = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0
    new_episode = True

    while True:
        env.render()

        while INF in three_last_obs:
            three_last_obs[0] = three_last_obs[1]
            three_last_obs[1] = three_last_obs[2]
            three_last_obs[2] = get_data_from_obs(obs) / 150

            obs, rew, done, info = env.step(env.action_space.sample())
            env.render()
        x = three_last_obs.reshape(-1, 1)
        aprob, h = policy_forward(model, x)
        if new_episode:
            new_episode = False

        action = up if np.random.uniform() < aprob else down
        xs.append(x.flatten())
        hs.append(h.flatten())
        y = 1 if action == up else 0

        dlogps.append(y - aprob)

        obs, rew, done, info = env.step(action)
        three_last_obs[0] = three_last_obs[1]
        three_last_obs[1] = three_last_obs[2]
        three_last_obs[2] = get_data_from_obs(obs) / 150

        reward_sum += rew
        drs.append(rew)

        if done:

            episode_number += 1
            if episode_number % 100 == 0:
                f = open('model', 'wb')
                pickle.dump(model, f)
                f.close()

            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs, hs, dlogps, drs = [], [], [], []  # reset array memory

            discounted_epr = discount_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
            grad = policy_backward(model, eph, epdlogp, epx)
            for k in model:
                grad_buffer[k] += grad[k]

            new_episode = True
            if episode_number % batch_size == 0:
                print("New episode ", episode_number)
                for k, v in model.items():
                    g = grad_buffer[k]
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = np.zeros_like(v)
            reward_sum = 0
            env.reset()


H = 8  # number of hidden layer neurons
D = 4 * 3  # input dimension
discount = 0.99
batch_size = 4
learning_rate = 1e-2
decay_rate = 0.99
max_slider_pos = 100


def pg_agent(game_name):
    model = {}
    model['w1'] = np.random.randn(H, D) / np.sqrt(D)
    model['w2'] = np.random.randn(H) / np.sqrt(H)

    grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
    rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}

    model = torch.nn.Sequential(
        torch.nn.Linear(D, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D),
        torch.nn.Sigmoid()
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # y_true * log model_output + (1 - y_true) * log 1 - model_output

    ## Model training
    env = retro.make(game=game_name)
    obs = env.reset()


def get_data_from_obs(obs) -> np.ndarray:
    channel = obs[34:-16, :, 0]

    slider_one_pos = np.argwhere(channel[:, 16] == 213)
    slider_two_pos = np.argwhere(channel[:, 16] == 92)

    if len(slider_one_pos) == 0:
        slider_one_pos = np.array([160])
    else:
        slider_one_pos = slider_one_pos[0]

    if len(slider_two_pos) == 0:
        slider_two_pos = np.array([160])
    else:
        slider_two_pos = slider_two_pos[0]

    ball_pos = np.argwhere(channel == 236)
    if len(ball_pos) == 0:
        ball_pos = np.array([INF, INF])
    else:
        ball_pos = ball_pos[0]

    return np.concatenate((slider_one_pos, slider_two_pos, ball_pos))


def random_agent(gamename):
    # Airstriker-Genesis is just a sample game, included with the library
    env = retro.make(game=gamename)
    obs = env.reset()

    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()

    env.close()


def main(gamename, algorithm):
    return train(gamename, algorithm)


def train(game, algorithm, save=None):
    if algorithm == "random":

        random_agent(game)
        return 0

    elif algorithm == "test":

        print("Warning: algorithm testing mode is highly unstable and subject to change, use for debugging purposes")
        test_agent(game)
        return 0

    elif algorithm == "policy-gradient":

        print("TODO: Implement PG agent training")
        return 0

    else:  # invalid algorithm

        raise NameError(f"No such algorithm: {algorithm}")


if __name__ == "__main__":
    # print(retro.data.list_games())
    main(*sys.argv[1:3])
