import retro
import sys


def list_games():

    return retro.data.list_games()

# def test_agent(gamename):



def random_agent(gamename):

    env = retro.make(game=gamename) #Airstriker-Genesis is just a sample game, included with the library
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
        # test_agent(game)
        return 0

    elif algorithm == "policy-gradient":

        print("TODO: Implement PG agent training")
        return 0

    else: #invalid algorithm

        raise NameError(f"No such algorithm: {algorithm}")


if __name__ == "__main__":
    # print(retro.data.list_games())
    main(*sys.argv[1:3])

