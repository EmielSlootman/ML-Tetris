import neuralnetwork as nn
import tetris
import numpy as np
import random
import losses

batch_size = 4
gamma = 0.95
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 20000
lr = 0.001
num_episodes = 2

# em = tetris.TetrisApp(8, 16, 750, True, 40, 30)
# strategy = nn.EpsilonGreedyStrategt(eps_start, eps_end, eps_decay)
# agent = nn.Agens(strategy, em.num_actions_available())
# memory = nn.ReplayMemory(memory_size)

# policy_net = nn.DQN(em.get_state_size(), 1, nn.CE_Loss)
# target_net = nn.DQN(em.get_state_size(), 1, nn.CE_Loss)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()
# optimizer = optim.Adam(params=policy_net.parameters(),policy_net, lr=lr)

# episode_durations = []

# for episode in range(num_episodes):
#     em.reset()
#     state = em.get_state()
#     for timestep in count():
#         action = agent.select_action(state, policy_net)
#         reward = em.take_action(action)
#         next_state = em.get_state()
#         memory.push(nn.Experience(state, action, next_state, reward))
#         state = next_state
#         if memory.can_provide_sample(batch_size):
#             experiences = memory.sample(batch_size)
#             states, actions, rewards, next_states = extract_tensors(experiences)
            
#             current_q_values = QValues.get_current(policy_net, states, actions)
#             next_q_values = QValues.get_next(target_net, next_states)
#             target_q_values = (next_q_values * gamma) + rewards

#             loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         if em.done:
#             episode_durations.append(timestep)
#             plot(episode_durations, 100)
#             break
        
#     if episode % target_update == 0:
#         target_net.load_state_dict(policy_net.state_dict())

# em.close()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

em = tetris.TetrisApp(8, 16, 750, True, 40, 30)
em.pcrun()
policy_net = nn.DQNsimple(em.get_state_size(), 1, losses.MSE_loss)
memory = nn.ReplayMemory(memory_size)
strategy = nn.EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

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



em.quit()