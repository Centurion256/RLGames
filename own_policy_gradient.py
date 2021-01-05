import numpy as np
import pickle
import retro
from utils import get_data_from_obs
import time

batch_size = 5
learning_rate = 1e-5
gamma = 0.99
decay_rate = 0.99
resume = True
render = True

up = [1, 0, 0, 0, 1, 0, 0, 0]
down = [1, 0, 0, 0, 0, 1, 0, 0]

D = 6
H = 20

if resume:
    print("Resuming")
    model = pickle.load(open('own_policy_gradient.pkl', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D) / 160
    model['W2'] = np.random.randn(H) / np.sqrt(H) / 160

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, and hidden state


def policy_backward(eph, epdlogp, epx):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


def own_policy_gradient(game_name):
    env = retro.make(game='Pong-Atari2600')
    observation = env.reset()
    prev_x = np.array([50, 50, 50])
    xs, hs, dlogps, drs = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 1
    while True:
        if render:
            env.render()

        # preprocess the observation, set input to network to be difference image
        cur_x = get_data_from_obs(observation)
        x = np.concatenate((prev_x, cur_x))
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(x)
        # print(aprob)
        action = down if np.random.uniform() < aprob else up
	
        xs.append(x)
        hs.append(h)
        y = 1 if action == down else 0
        dlogps.append(y - aprob)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        time.sleep(0.003)
        reward_sum += reward

        drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

        if done:  # an episode finished
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs, hs, dlogps, drs = [], [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
            grad = policy_backward(eph, epdlogp, epx)
            for k in model:
                grad_buffer[k] += grad[k]  # accumulate grad over batch

            # perform rmsprop parameter update every batch_size episodes
            if episode_number % batch_size == 0:
                for k, v in model.items():
                    g = grad_buffer[k]
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

            # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print(
                'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            if episode_number % 100 == 0:
                print("Episode: ", episode_number)
                pickle.dump(model, open('own_policy_gradient.pkl', 'wb'))
            reward_sum = 0
            observation = env.reset()  # reset env
            prev_x = np.array([50, 50, 50])

        # if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        #     print(f'ep {episode_number}: game finished, reward: {reward}' + ('' if reward == -1 else ' !!!!!!!!'))
