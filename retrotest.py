import retro
import sys
from sklearn.neural_network import MLPRegressor
import random


def list_games():
    return retro.data.list_games()


up = [1, 0, 0, 0, 1, 0, 0, 0]
down = [1, 0, 0, 0, 0, 1, 0, 0]


def test_agent(game_name):
    network = MLPRegressor(solver='sgd', activation='logistic', alpha=1e-5, hidden_layer_sizes=(10, 7, 2),
                           learning_rate_init=0.01, random_state=31)

    env = retro.make(game=game_name)  # Airstriker-Genesis is just a sample game, included with the library
    obs = env.reset()

    three_last_obs = get_data_from_obs(obs) * 3
    network.fit([[0] * 12], [0.5])
    last_point_observations = []

    while True:
        # Make random moves while the ball is not visible
        while float('inf') in three_last_obs:
            three_last_obs[:8] = three_last_obs[4:]
            three_last_obs[8:] = get_data_from_obs(obs)
            obs, rew, done, info = env.step(env.action_space.sample())
            env.render()

        predicted_move = network.predict([three_last_obs])[0]
        print(predicted_move)
        if random.random() < (1 - abs(predicted_move)) / 3:
            predicted_move = -predicted_move
        print(predicted_move)
        print('-----')
        obs, rew, done, info = env.step(down if predicted_move < 0 else up)
        obs = get_data_from_obs(obs)
        env.render()

        # Update NN weights if some player won a point
        if rew != 0:
            last_point_observations.reverse()
            for i in range(min(len(last_point_observations), 192)):
                network.partial_fit([last_point_observations[i][0]],
                                    [(rew * (1 if last_point_observations[i][1] > 0 else -1)) * 0.97 ** i])
            last_point_observations.clear()
            rew = 1
            last_point_observations = last_point_observations[192:]
            for i in range(len(last_point_observations)):
                network.partial_fit([last_point_observations[i][0]],
                                    [(rew * (1 if last_point_observations[i][1] > 0 else -1)) * 0.97 ** i])
            last_point_observations.clear()

        if float('inf') in obs:
            continue

        # Update latest observations for
        last_point_observations.append((three_last_obs, predicted_move))
        # shift three_last_obs to the left and replace the last element with new observation
        three_last_obs[:8] = three_last_obs[4:]
        three_last_obs[8:] = obs

        if done:
            last_point_observations.clear()
            obs = env.reset()
            three_last_obs = get_data_from_obs(obs) * 3

    env.close()


def get_data_from_obs(obs) -> list:
    slider_one_pos = float('inf')
    slider_two_pos = float('inf')
    ball_pos = [float('inf')] * 2
    channel = obs[34:-16, :, 0]

    for j in range(len(channel)):
        if channel[j][16] == 213:
            slider_one_pos = min(slider_one_pos, j)

    for j in range(len(channel)):
        if channel[j][140] == 92:
            slider_two_pos = min(slider_two_pos, j)

    # Check if slider moved down beyond the seen scope
    slider_two_pos = min(slider_two_pos, 159)
    slider_one_pos = min(slider_one_pos, 159)

    for i in range(len(channel)):
        for j in range(len(channel[i])):
            if channel[i][j] == 236:  # ball
                ball_pos = min(ball_pos, [i, j])
    return [slider_one_pos] + [slider_two_pos] + ball_pos


def random_agent(gamename):
    env = retro.make(game=gamename)  # Airstriker-Genesis is just a sample game, included with the library
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
