import tetris
import neuralnetwork as NN
import losses
import numpy as np
import matplotlib.pyplot as plt

N = 655
N2 = 3
score = np.zeros((N, N2))
for i in range(N):
    em = tetris.TetrisApp(8, 16, 0.01*750, False, 40, 30*100)
    net = NN.DQN(em.get_state_size(), 1, losses.MSE_loss)

    gene = np.loadtxt("data\\evolutionNNstate168\\generation" + str(i) + ".csv", delimiter=',')

    index = 0
    net.L1.W = gene[index:index + net.L1.W.size].reshape(net.L1.W.shape)
    index += net.L1.W.size
    net.L1.B = gene[index:index + net.L1.B.size].reshape(net.L1.B.shape)
    index += net.L1.B.size
    net.L2.W = gene[index:index + net.L2.W.size].reshape(net.L2.W.shape)
    index += net.L2.W.size
    net.L2.B = gene[index:index + net.L2.B.size].reshape(net.L2.B.shape)
    index += net.L2.B.size
    net.L3.W = gene[index:index + net.L3.W.size].reshape(net.L3.W.shape)
    index += net.L3.W.size
    net.L3.B = gene[index:index + net.L3.B.size].reshape(net.L3.B.shape)

    for j in range(N2):
        em.pcrun()
        em.reset()
        done = False
        while not done:
            next_state = em.get_next_states()
            predicted_qs = {}

            for ir, (*data,) in enumerate(next_state):
                predicted_qs[(data[0], data[1])] = net.f_pass(np.array([next_state[data[0], data[1]]]).T)[0,0]

            best_move = max(predicted_qs, key=predicted_qs.get)

            reward, done = em.pcplace(best_move[0], best_move[1])
        print(i, j, em.score)
        score[i, j] = em.score

plt.figure()
x = np.linspace(1, N, N)
plt.plot(x, np.sum(score, axis=1))
plt.grid()
plt.xlabel("Generation")
plt.ylabel("Average score over 3 games")
plt.show()