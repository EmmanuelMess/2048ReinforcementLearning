import numpy as np

import game2048_env
from lookup_table import LUT


def argmax(a):
    return max(range(len(a)), key=lambda x: a[x])


def set_1d_board(game: game2048_env.Game2048Env, board: np.array):
    game.set_board(board.reshape((4, 4)))


def get_1d_board(game: game2048_env.Game2048Env):
    return game.get_board().ravel()


def set_log_board(game: game2048_env.Game2048Env, board: np.array):
    game.set_board((2 ** board).reshape((4, 4)))


_representation = 2 ** (np.arange(16, dtype=int))


def get_log_board(game: game2048_env.Game2048Env):
    x = game.get_board().ravel()

    for i in range(1, 15):
        x = np.where(x == _representation[i], i, x)

    return x


def evaluate(extraGameLog: game2048_env.Game2048Env, extraGame: game2048_env.Game2048Env, board, action, lut: LUT):
    set_1d_board(extraGameLog, board)
    set_log_board(extraGame, board)
    try:
        extraGameLog.move(action)
    except game2048_env.IllegalMove:
        pass

    try:
        reward = extraGame.move(action)
    except game2048_env.IllegalMove:
        reward = -1

    return reward + lut.calculate(get_1d_board(extraGameLog))


def learn_evaluation(extraGameLog: game2048_env.Game2048Env, extraGame: game2048_env.Game2048Env, non_tiled_next_board, next_board, lut: LUT):
    best_next_action = argmax([evaluate(extraGameLog, extraGame, next_board, 0, lut),
                                  evaluate(extraGameLog, extraGame, next_board, 1, lut),
                                  evaluate(extraGameLog, extraGame, next_board, 2, lut),
                                  evaluate(extraGameLog, extraGame, next_board, 3, lut)])
    set_1d_board(extraGameLog, next_board)
    set_log_board(extraGame, next_board)
    try:
        extraGameLog.move(best_next_action)
    except game2048_env.IllegalMove:
        pass

    try:
        reward_next = extraGame.move(best_next_action)
    except game2048_env.IllegalMove:
        reward_next = -1

    non_tiled_next_next_board = get_1d_board(extraGameLog)
    lut.update(non_tiled_next_board, non_tiled_next_next_board, reward_next)


def main():
    tuples = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
              [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15],
              [0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7],
              [4, 5, 8, 9], [5, 6, 9, 10], [6, 7, 10, 11],
              [8, 9, 12, 13], [9, 10, 13, 14], [10, 11, 14, 15]]

    lut = LUT(len(tuples), [4 for i in range(len(tuples))], tuples, 15, 0.0025)

    env = game2048_env.Game2048Env()
    extraGameLog = game2048_env.Game2048Env()
    extraGame = game2048_env.Game2048Env()
    score = 0.0
    print_interval = 20

    for n_epi in range(125000):
        env.reset()
        done = False
        highest = 0

        while not done:
            currBoard = get_log_board(env)
            best_action = argmax([evaluate(extraGameLog, extraGame, currBoard, 0, lut),
                                     evaluate(extraGameLog, extraGame, currBoard, 1, lut),
                                     evaluate(extraGameLog, extraGame, currBoard, 2, lut),
                                     evaluate(extraGameLog, extraGame, currBoard, 3, lut)])

            s_prime, r, done, info = env.step(best_action)

            set_1d_board(extraGame, currBoard)
            non_tiled_extra_board = get_1d_board(extraGame)
            learn_evaluation(extraGameLog, extraGame, non_tiled_extra_board, get_log_board(env), lut)

            if highest < env.highest():
                highest = env.highest()

            score += r

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {}".format(n_epi, score / print_interval))
            print("Highest : {}".format(highest))
            score = 0.0
    env.close()


if __name__ == '__main__':
    main()