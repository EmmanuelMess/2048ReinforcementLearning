import gym
import torch
import torch.optim as optim
from torch.distributions import Categorical

import game2048_env

from reinforce import Policy

def main():
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
            s_prime, r, done, info = env.step(a.item())
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