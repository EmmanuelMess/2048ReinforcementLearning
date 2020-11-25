import numpy as np

import game2048_env
from lookup_table import LUT


def evaluate(extraGame: game2048_env.Game2048Env, board, action, lut: LUT):
    extraGame.set_board(board.reshape((4,4)))
    try:
        reward = extraGame.move(action)
    except game2048_env.IllegalMove:
        reward = 0

    return reward + lut.calculate(extraGame.get_board().ravel())


def learn_evaluation(extraGame: game2048_env.Game2048Env, non_tiled_next_board, next_board, lut: LUT):
    best_next_action = np.argmax([evaluate(extraGame, next_board, 0, lut),
                             evaluate(extraGame, next_board, 1, lut),
                             evaluate(extraGame, next_board, 2, lut),
                             evaluate(extraGame, next_board, 3, lut)])
    extraGame.set_board(next_board.reshape((4,4)))
    try:
        reward_next = extraGame.move(best_next_action)
    except game2048_env.IllegalMove:
        reward_next = 0
    non_tiled_next_next_board = extraGame.get_board().ravel()
    lut.update(non_tiled_next_board, non_tiled_next_next_board, reward_next)


def main():
    tuples = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
              [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15],
              [0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7],
              [4, 5, 8, 9], [5, 6, 9, 10], [6, 7, 10, 11],
              [8, 9, 12, 13], [9, 10, 13, 14], [10, 11, 14, 15]]

    lut = LUT(len(tuples), [4 for i in range(len(tuples))], tuples, 15, 0.0025)

    env = game2048_env.Game2048Env()
    extraGame = game2048_env.Game2048Env()
    score = 0.0
    print_interval = 200

    for n_epi in range(100000):
        env.reset()
        done = False

        while not done:
            currBoard = env.get_board().ravel()
            best_action = np.argmax([evaluate(extraGame, currBoard, 0, lut),
                                     evaluate(extraGame, currBoard, 1, lut),
                                     evaluate(extraGame, currBoard, 2, lut),
                                     evaluate(extraGame, currBoard, 3, lut)])

            s_prime, r, done, info = env.step(best_action)

            extraGame.set_board(currBoard.reshape((4,4)))
            non_tiled_extra_board = extraGame.get_board().ravel()
            learn_evaluation(extraGame, non_tiled_extra_board, env.get_board().ravel(), lut)

            score += r

        #env.render()
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {}".format(n_epi, score / print_interval))
            score = 0.0
    env.close()


if __name__ == '__main__':
    main()