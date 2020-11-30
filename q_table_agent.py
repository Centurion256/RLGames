import os
import retro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_data_from_obs
import time


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
    max_episodes_to_run = 40
    total_time = 0

    agent = PongQLearningAgent(
        learning_rate=0.09,
        discount_factor=0.92,
        exploration_rate=0.9,
        exploration_decay_rate=0.95
    )
    if os.path.exists('q_table_model.csv'):
        print('Model loaded from file')
        agent.q = np.loadtxt('q_table_model.csv', delimiter=',')

    for episode_index in range(max_episodes_to_run):
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
    np.savetxt('q_table_model.csv', agent.q, delimiter=',')
    print("Goal not reached after {} episodes.".format(max_episodes_to_run))
    # return episode_history


def save_history(history, experiment_dir):
    # Save the episode lengths to CSV.
    filename = os.path.join(experiment_dir, "episode_history.csv")
    dataframe = pd.DataFrame(history.lengths, columns=["length"])
    dataframe.to_csv(filename, header=True, index_label="episode")


def q_table_agent(game_name):
    random_state = 31
    experiment_dir = "pong-qlearning-1"

    env = retro.make(game=game_name)

    episode_history = run_agent(env, verbose=True)
    save_history(episode_history, experiment_dir)
    env.monitor.close()


if __name__ == "__main__":
    q_table_agent('Pong-Atari2600')
