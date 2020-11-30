import numpy as np

INF = float('inf')


def get_data_from_obs(obs) -> np.ndarray:
    channel = obs[34:-16, :, 0]

    slider_two_pos = np.argwhere(channel[:, 140] == 92)

    if len(slider_two_pos) == 0:
        slider_two_pos = np.array([160])
    else:
        slider_two_pos = slider_two_pos[0]

    ball_pos = np.argwhere(channel == 236)
    if len(ball_pos) == 0:
        ball_pos = np.array([INF, INF])
    else:
        ball_pos = ball_pos[0]

    return np.concatenate((slider_two_pos, ball_pos))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
