import argparse
import retro
from retro.examples.interactive import RetroInteractive
import retrotest

games = {"Pong-Atari2600"}
conflater = {
    "pong": "Pong-Atari2600"
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser("game-agent-launcher")
    parser.add_argument("task", choices=["play", "train", "list", "run"])
    parser.add_argument("-g", "--game", default="pong",
                        help="specify a game to be played/trained on. Defaults to 'Pong-Atari2600'")
    parser.add_argument("-a", "--algorithm", default="random",
                        help="specify a algorithm to be used to train the agent. Defaults to 'random'")

    args = parser.parse_args()
    if args.game not in games:
        try:
            args.game = conflater[args.game]
        except KeyError as err:
            print(err, "is not a valid game name")
            exit(-1)

    if args.task == "play":

        ia = RetroInteractive(game=args.game, state=retro.State.DEFAULT, scenario=None)
        ia.run()

    elif args.task == "train":  # train an agent with specified algorithm on a specified game

        print(f"game={args.game}, algorithm={args.algorithm}")
        retrotest.train(game=args.game, algorithm=args.algorithm)

    elif args.task == "list":  # print a list of all copatible games implemented in retro

        print(retrotest.list_games())

    elif args.task == "run":  # run an existing pre-trained agent

        print(f"game={args.game}, algorithm={args.algorithm}")
        retrotest.run(game=args.game, algorithm=args.algorithm, model='pg_model.pkl')
        
    else:
        raise argparse.ArgumentError(f"task '{args.task}' does not exist")
