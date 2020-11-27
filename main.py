import numpy as np

import game2048_env
from lookup_table import LUT


def argmax(a):
    return max(range(len(a)), key=lambda x: a[x])

_extraGameLog = game2048_env.Game2048Env()
_extraGame = game2048_env.Game2048Env()


def set_1d_board(board: np.array):
    _extraGameLog.set_board(board.copy().reshape((4, 4)))


def get_1d_board():
    return _extraGameLog.get_board().copy().ravel()


def set_log_board(game: game2048_env.Game2048Env, board: np.array):
    board = board.copy()
    game.set_board((np.where(board == 0, board, 2 ** board)).reshape((4, 4)))


_representation = 2 ** (np.arange(16, dtype=int))


def get_log_board(game: game2048_env.Game2048Env):
    x = game.get_board().copy().ravel()

    for i in range(1, 15):
        x = np.where(x == _representation[i], i, x)

    return x


def compute_afterstate(board, action):
    set_1d_board(board)
    set_log_board(_extraGame, board)
    try:
        _extraGameLog.move(action)
    except game2048_env.IllegalMove:
        pass

    try:
        reward = _extraGame.move(action)
    except game2048_env.IllegalMove:
        reward = 0

    return get_1d_board(), reward


def evaluate(board, action, lut: LUT):
    new_board, reward = compute_afterstate(board, action)

    return reward + lut.calculate(new_board)


def learn_evaluation(non_tiled_next_board, next_board, lut: LUT):
    best_next_action = argmax([evaluate(next_board, 0, lut),
                                  evaluate(next_board, 1, lut),
                                  evaluate(next_board, 2, lut),
                                  evaluate(next_board, 3, lut)])

    non_tiled_next_next_board, reward_next = compute_afterstate(next_board, best_next_action)

    lut.update(non_tiled_next_board, non_tiled_next_next_board, reward_next)


def main():
    tuples = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
              [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15],
              [0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7],
              [4, 5, 8, 9], [5, 6, 9, 10], [6, 7, 10, 11],
              [8, 9, 12, 13], [9, 10, 13, 14], [10, 11, 14, 15]]

    lut = LUT(len(tuples), [4 for i in range(len(tuples))], tuples, 15, 0.01)

    env = game2048_env.Game2048Env()
    env.seed(123)
    score = 0.0
    print_interval = 200

    for n_epi in range(125000):
        env.reset()
        done = False
        highest = 0

        while not done:
            currBoard = get_log_board(env)
            best_action = argmax([evaluate(currBoard, 0, lut),
                                     evaluate(currBoard, 1, lut),
                                     evaluate(currBoard, 2, lut),
                                     evaluate(currBoard, 3, lut)])

            s_prime, r, done, info = env.step(best_action)

            set_1d_board(currBoard)

            try:
                _extraGameLog.move(best_action)
            except game2048_env.IllegalMove:
                pass

            non_tiled_extra_board = get_1d_board()
            learn_evaluation(non_tiled_extra_board, get_log_board(env), lut)

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