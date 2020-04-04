import neuralnetwork as nn
import tetris
import numpy as np
import random
import losses
import matplotlib.pyplot as plt
import matplotlib
import copy
import sys

batch_size = 512
gamma = 0.95
eps_start = 1
eps_end = 0.0
eps_decay = 0.002
memory_size = 20000
lr = 0.001 * 0.001
num_episodes = 3000

filename = "TEST_" + str(lr)

def moving_average(a, n=30) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

em = tetris.TetrisApp(10, 20, 750, False, 40, 30*100)
em.pcrun()
policy_net = nn.DQNsimple(em.get_state_size(), 1, losses.MSE_loss)
memory = nn.ReplayMemory(memory_size)
strategy = nn.EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

# fig = plt.figure()
# thismanager = plt.get_current_fig_manager()
# thismanager.window.wm_geometry("+500+0")
# plt.ion()

score = np.zeros(num_episodes ) * np.nan
lossess = np.zeros(num_episodes)
current_step = -1
for episode in range(num_episodes):
    current_step += 1
    em.reset()
    done = False
    state = em._get_board_props(em.board)
    while not done:
        next_state = em.get_next_states()
        rate = strategy.get_exploration_rate(current_step)

        if rate > random.random():
            best_move = random.sample(list(next_state),1)[0]
        else:
            predicted_qs = {}

            for i, (*data,) in enumerate(next_state):
                predicted_qs[(data[0], data[1])] = policy_net.f_pass(np.array([next_state[data[0], data[1]]]).T)[0,0]

            best_move = max(predicted_qs, key=predicted_qs.get)

        reward, done = em.pcplace(best_move[0], best_move[1])

        memory.push(nn.Experience(state, done, next_state[best_move], reward))
        state = next_state[best_move]

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, dones, rewards, next_states = [], [], [], []
            for exp in experiences:
                states.append(exp.state)
                dones.append(exp.done)
                rewards.append(exp.reward)
                next_states.append(exp.next_state)

            target_q_values = np.zeros(batch_size)
            for i in range(batch_size):
                next_q_value = policy_net.f_pass(np.array([next_states[i]]).T)[0,0]
                if dones[i]:
                    target_q_values[i] = rewards[i]
                else:
                    target_q_values[i] = rewards[i] + gamma * next_q_value

            loss = nn.SGD(batch_size, np.array(states).T, np.array([target_q_values]), policy_net, lr=lr)
            if np.isnan(loss):
                break
            lossess[episode] = loss
    score[episode] = em.score
    print(episode, strategy.get_exploration_rate(current_step), em.score)

    # plt.clf()
    # plt.plot(np.linspace(49, num_episodes, num_episodes - 49) , moving_average(score, 50))
    # plt.xlabel("Episodes")
    # plt.ylabel('Average score over 50 episodes')
    # plt.draw()
    # plt.pause(0.0001)


x = np.linspace(1, num_episodes, num_episodes)

plt.xlabel("Episodes")
plt.ylabel('Average score over 30 episodes')
plt.grid()
plt.plot(np.linspace(30, num_episodes, num_episodes - 29) , moving_average(score, 30))
plt.savefig(filename + ".png")

np.savetxt(filename + ".csv", np.asarray([x.astype(int), score.astype(int), lossess]).T, delimiter=',')


em.quit()