import gym
import torch
import torch.optim as optim
from torch.distributions import Categorical

import game2048_env
from lookup_table import LUT

from reinforce import Policy

def main():
    tuples = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
              [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15],
              [0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7],
              [4, 5, 8, 9], [5, 6, 9, 10], [6, 7, 10, 11],
              [8, 9, 12, 13], [9, 10, 13, 14], [10, 11, 14, 15]]

    lut = LUT(17, [4 for i in range(17)], tuples)

    env = game2048_env.Game2048Env()
    pi = Policy(4 * 4 * 16, 4)
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset().ravel()
        done = False

        while not done:
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            best_action = a.item()
            s_prime, r, done, info = env.step(best_action)
            s_prime = s_prime.ravel()
            pi.put_data((r, prob[a]))
            s = s_prime
            score += r

        pi.train_net()

        env.render()
        if n_epi % print_interval == 0 and n_epi != 0:
            #print("# of episode :{}, avg score : {}".format(n_epi, score / print_interval))
            score = 0.0
    env.close()


if __name__ == '__main__':
    main()