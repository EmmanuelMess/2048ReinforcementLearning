import torch
import torch.optim as optim

import game2048_env
from dqn import Qnet, ReplayBuffer, train, learning_rate

def main():
    env = game2048_env.Game2048Env()
    q = Qnet(4 * 4 * 16, 4)
    q_target = Qnet(4 * 4 * 16, 4)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(1000000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        s = env.reset().ravel()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(a)
            s_prime = s_prime.ravel()
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        env.render()
        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            #print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
            #    n_epi, score / print_interval, memory.size(), epsilon * 100))
            score = 0.0
    env.close()


if __name__ == '__main__':
    main()