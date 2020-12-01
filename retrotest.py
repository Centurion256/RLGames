import retro
import sys
from q_table_agent import q_table_agent, run_q_table
from policy_gradient import test_agent, test_agent_play

INF = float('inf')


def list_games():
    return retro.data.list_games()


def random_agent(game_name):
    env = retro.make(game=game_name)
    env.reset()

    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            env.reset()


def main(game_name, algorithm):
    return train(game_name, algorithm)


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
    elif algorithm == 'q_table':
        q_table_agent(game)

    else:  # invalid algorithm

        raise NameError(f"No such algorithm: {algorithm}")


def run(game, algorithm, model):
    if algorithm == "random":

        random_agent(algorithm)

    elif algorithm == "test":

        test_agent_play(game, model)

    elif algorithm == "q_table":

        run_q_table()


if __name__ == "__main__":
    # print(retro.data.list_games())
    main(*sys.argv[1:3])
