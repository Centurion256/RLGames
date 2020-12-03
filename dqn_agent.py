# import torch 
# import torch.nn as nn
import numpy as np
import retro
from tensorflow import keras
import random
from ring_buf import RingBuf

learning_rate = 0.00025


def prepro(i):
    i = i[35:195]  # crop
    i = i[::2, ::2, 0]  # downsample by factor of 2
    i[i == 144] = 0  # erase background (background type 1)
    i[i == 109] = 0  # erase background (background type 2)
    i[i != 0] = 1  # everything else (paddles, ball) just set to 1
    return i.astype(np.float)


def fit_batch(model, gamma, start_states, actions, rewards, next_states, is_terminal):
    next_Q_values = model.predict([next_states, np.ones(actions.shape)])
    next_Q_values[is_terminal] = 0

    Q_values = rewards + gamma * np.max(next_Q_values, axis=1)

    model.fit(
        [start_states, actions], actions * Q_values[:, None],
        epochs=1, batch_size=len(start_states), verbose=0
    )


def atari_model(n_actions):
    # We assume a theano backend here, so the "channels" are first.
    ATARI_SHAPE = (2, 80, 80)

    # With the functional API we need to define the inputs.
    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    actions_input = keras.layers.Input((n_actions,), name='mask')

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1]
    normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
    conv_1 = keras.layers.Convolution2D(
        16, kernel_size=(8, 8), strides=(4, 4), activation='relu', padding='same', name='conv1'
    )(normalized)

    conv_2 = keras.layers.Convolution2D(
        32, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same', name='conv2'
    )(conv_1)
    # Flattening the second convolutional layer.
    conv_flattened = keras.layers.Flatten()(conv_2)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = keras.layers.Dense(n_actions)(hidden)

    # Finally, we multiply the output by the mask!
    filtered_output = keras.layers.Multiply()([output, actions_input])

    model = keras.models.Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    optimizer = keras.optimizers.RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')

    keras.backend.set_image_data_format('channels_first')

    return model


exploration_rate = 0
exploration_decay_rate = 0.999995  # On iteratoion 1000000 is about 0.6


def get_epsilon_from_iteration():
    global exploration_rate
    exploration_rate = exploration_decay_rate * exploration_rate
    return exploration_rate


up = [1, 0, 0, 0, 1, 0, 0, 0]
down = [1, 0, 0, 0, 0, 1, 0, 0]


def get_action_from_output(output):
    return [1, 0] if output[0] > output[1] else [0, 1]


def get_real_action(action):
    return up if action[0] > action[1] else down


def q_iteration(env, model, state, memory, prev_state):
    # Choose epsilon based on the iteration
    epsilon = get_epsilon_from_iteration()

    # Choose the action
    if random.random() < epsilon:
        action = [1, 0] if random.random() < 0.5 else [0, 1]
    else:
        action = get_action_from_output(model.predict([np.asarray([[prev_state, state]]), np.ones((1, 2))])[0])
        action = [1, 0] if action[0] > action[1] else [0, 1]

    # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
    new_frame, reward, done, _ = env.step(get_real_action(action))
    new_frame = prepro(new_frame)
    memory.add(state, action, new_frame, reward, done)

    # Sample and fit
    if done:
        print("Training")
        states, actions, new_states, rewards, is_done = memory.sample_batch(32)

        fit_batch(model, 0.99, np.array(states), np.array(actions), np.array(rewards), np.array(new_states),
                      np.array(is_done))
    return new_frame, done


class Memory:
    def __init__(self, size):
        self.elements = 0
        self.size = size
        self.frames = RingBuf(size)
        self.actions = RingBuf(size)
        self.rewards = RingBuf(size)
        self.is_done = RingBuf(size)

    def add(self, state, action, new_frame, reward, is_done):
        self.elements += 1
        if is_done:
            self.actions.append(action)
            self.frames.append(state)
            self.rewards.append(reward)
            self.is_done.append(False)

            self.frames.append(new_frame)
            self.is_done.append(True)
            self.actions.append(up)
            self.rewards.append(0)
        else:
            self.actions.append(action)
            self.frames.append(state)
            self.rewards.append(reward)
            self.is_done.append(False)

    def get_sample_indexes(self, size=32):
        return [random.randint(2, min(self.elements, self.size) - 2) for _ in range(size)]

    def sample_batch(self, size):
        indices = self.get_sample_indexes(size)
        actions = []
        frames = []
        new_frames = []
        rewards = []
        is_done = []
        for i in indices:
            actions.append(self.actions[i])
            frames.append([self.frames[i - 1], self.frames[i]])
            rewards.append(self.rewards[i])
            is_done.append(self.is_done[i])
            new_frames.append([self.frames[i], self.frames[i + 1]])
        return frames, actions, new_frames, rewards, is_done


def dqn_run(game_name):
    env = retro.make(game_name)
    model = atari_model(2)
    memory = Memory(750000)

    prev_obs = prepro(env.reset())

    for i in range(100):
        if i % 5000 == 0:
            print(f'Round {i}')
        action = [1, 0] if random.random() < 0.5 else [0, 1]
        obs, rew, done, info = env.step(get_real_action(action))
        obs = prepro(obs)

        memory.add(prev_obs, action, obs, rew, done)
        if done:
            env.reset()
        prev_obs = obs

    env.close()
    env = retro.make(game_name)
    obs = prepro(env.reset())
    prev_obs = obs
    while not done:
        obs_tmp = obs
        obs, is_done = q_iteration(env, model, obs, memory, prev_obs)
        prev_obs = obs_tmp
        if is_done:
            obs = preprop(env.reset())
        env.render()
