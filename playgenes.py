import tetris
import neuralnetwork as NN
import losses
import numpy as np

em = tetris.TetrisApp(10, 20, 750, True, 40, 30*100)
net = NN.DQN(em.get_state_size(), 1, losses.MSE_loss)
em.pcrun()
em.reset()
done = False

gene = np.loadtxt("evolution\\generation16.csv", delimiter=',')

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

while not done:
    next_state = em.get_next_states()
    predicted_qs = {}

    for i, (*data,) in enumerate(next_state):
        predicted_qs[(data[0], data[1])] = net.f_pass(np.array([next_state[data[0], data[1]]]).T)[0,0]

    best_move = max(predicted_qs, key=predicted_qs.get)

    reward, done = em.pcplace(best_move[0], best_move[1])

