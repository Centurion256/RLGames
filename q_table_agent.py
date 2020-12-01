import os
import retro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_data_from_obs
import time
import keyboard


class PongQLearningAgent:

    def __init__(self,
                 learning_rate=0.9,
                 discount_factor=0.97,
                 exploration_rate=0.5,
                 exploration_decay_rate=0.99):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.state = None
        self.action = None
        self._num_actions = 2
        self.prev_obs = np.zeros((4,))
        self.q = np.zeros((21 ** 6, 2))

    @staticmethod
    def get_standardized(value, min_bound, max_bound):
        if not min_bound < value < max_bound:
            print('\n\n', value, '\n\n')

        return (value - min_bound) / (max_bound - min_bound)

    def _build_state(self, observation) -> int:
        if np.inf in observation or np.inf in self.prev_obs:
            self.prev_obs = observation
            return 0
        res = 0
        for i in range(3):
            res += (int(self.prev_obs[i] / 8)) * 20 ** i
        for i in range(3, 6):
            res += (int(observation[i - 4] / 8)) * 20 ** i
        self.prev_obs = observation
        return res

    def begin_episode(self, observation):
        self.state = self._build_state(observation)

        # Reduce exploration over time.
        self.exploration_rate *= self.exploration_decay_rate

        self.action = np.argmax(self.q[self.state])
        return self.action

    def act(self, observation, reward):
        next_state = self._build_state(observation)

        # Exploration/exploitation: choose a random action or select the best one.
        enable_exploration = (1 - self.exploration_rate) <= np.random.uniform(0, 1)

        if enable_exploration:
            next_action = np.random.randint(0, self._num_actions)
        else:
            next_action = np.argmax(self.q[next_state])

        self.q[self.state, self.action] = (1 - self.learning_rate) * self.q[self.state, self.action] + \
                                          self.learning_rate * (reward + self.discount_factor * max(self.q[next_state]))

        self.state = next_state
        self.action = next_action
        return next_action


up = [1, 0, 0, 0, 1, 0, 0, 0]
down = [1, 0, 0, 0, 0, 1, 0, 0]


def get_action(action: int):
    return up if action == 1 else down


def log_timestep(index, action, reward, observation):
    pass


def run_agent(env, verbose=False):
    max_episodes_to_run = 20000
    total_time = 0

    agent = PongQLearningAgent(
        learning_rate=0.01,
        discount_factor=0.99,
        exploration_rate=0,
        exploration_decay_rate=0.985,
    )
    # if os.path.exists('q_table_model_new.csv'):
    #     print('Model loading from file...')
    #     agent.q = np.loadtxt('q_table_model_new.csv', delimiter=',')
    #     print('Model have loaded already...')

    for episode_index in range(1, max_episodes_to_run):
        # if episode_index % 1000 == 0:
        # np.savetxt('q_table_model_new.csv', agent.q, delimiter=',')
        print(episode_index)
        observation = get_data_from_obs(env.reset())
        action = agent.begin_episode(observation)

        while True:
            # Perform the action and observe the new state.
            observation, reward, done, info = env.step(get_action(action))

            # Update the display and log the current state.
            if verbose:
                env.render()

            # Get the next action from the agent, given our new state.
            action = agent.act(get_data_from_obs(observation), reward)

            # Record this episode to the history and check if the goal has been reached.
            if done:
                break

    print("Average score: ", total_time / max_episodes_to_run)


def q_table_agent(game_name):
    env = retro.make(game=game_name)
    run_agent(env, verbose=True)


def run_q_table():
    env = retro.make(game='Pong-Atari2600')
    max_episodes_to_run = 20000

    agent = PongQLearningAgent(
        learning_rate=0.00,
        discount_factor=0.99,
        exploration_rate=0,
        exploration_decay_rate=0.985,
    )
    if os.path.exists('q_table_model_new.csv'):
        print('Model loading from file...')
        agent.q = np.loadtxt('q_table_model_new.csv', delimiter=',')
        print('Model have loaded already...')
    else:
        print("No model found")
        return

    for episode_index in range(1, max_episodes_to_run):
        print("Episode: ", episode_index)
        observation = get_data_from_obs(env.reset())
        action = agent.begin_episode(observation)

        while True:
            # Perform the action and observe the new state.
            observation, reward, done, info = env.step(get_action(action))

            # Update the display and log the current state.
            env.render()

            # Get the next action from the agent, given our new state.
            action = agent.act(get_data_from_obs(observation), reward)
            time.sleep(0.02)

            # Record this episode to the history and check if the goal has been reached.
            if done:
                break


global_action = -1


def change_global(value):
    global global_action
    global_action = value


keyboard.add_hotkey('w', change_global, args=(1,))
keyboard.add_hotkey('s', change_global, args=(0,))


def play_against_q_table(random=False):
    global global_action
    left_up = 6
    left_down = 7
    right_up = 4
    right_down = 5
    zero_action = [1] + [0] * 14 + [1]

    env = retro.make(game='Pong-Atari2600', players=2)

    agent = PongQLearningAgent(
        learning_rate=0.00,
        discount_factor=0.99,
        exploration_rate=0,
        exploration_decay_rate=0.985,
    )
    # if os.path.exists('q_table_model_new.csv'):
    #     print('Model loading from file...')
    #     agent.q = np.loadtxt('q_table_model_new.csv', delimiter=',')
    #     print('Model have loaded already...')
    # else:
    #     print("No model found")
    #     return

    while True:
        observation = get_data_from_obs(env.reset())
        action = agent.begin_episode(observation)

        while True:
            # Perform the action and observe the new stat
            new_action = zero_action.copy()
            if random:
                left = int(np.random.uniform(0, 1) + 0.5)
            else:
                left = global_action
                global_action = -1
            if left == 1:
                new_action[left_up] = 1
            elif left == 0:
                new_action[left_down] = 1

            if action == 1:
                new_action[right_up] = 1
            else:
                new_action[right_down] = 1

            observation, reward, done, info = env.step(new_action)

            # Update the display and log the current state.
            env.render()

            # Get the next action from the agent, given our new state.
            action = agent.act(get_data_from_obs(observation), reward[1])
            time.sleep(0.02)

            # Record this episode to the history and check if the goal has been reached.
            if done:
                break


if __name__ == "__main__":
    # q_table_agent('Pong-Atari2600')
    play_against_q_table()
