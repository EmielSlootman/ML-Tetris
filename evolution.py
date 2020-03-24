import numpy as np
import random
import tetris
import neuralnetwork as NN
import losses

def cross(A, B):
    C = A
    ind = np.random.choice(B.shape[0], int(np.floor(len(B)/2)), replace=False)
    C[ind] = B[ind]
    ## take a random pick of dimension / 2 from A, the others from B
    return(C)

def mutate(C):
    dimension = len(C)
    a = 1
    b = 0.05
    D = (np.random.rand(dimension) >= (1-b)) * a * random.uniform(-1,1)
    ## for every gen take a b% chance to change the gen with a random value with amplitude a 
    return(np.add(C,D))

def cross_and_mutate(pop, pop_size):
    ## pop describes the complete population: 
    ## numbers of individuals is
    size_population = pop_size[0]
    N = int(size_population/4)
    dimension = pop_size[1] 
    offspring = pop
    k = N
    while k < size_population:
        if (k < 2 * N):
            offspring[k] = cross(pop[k], pop[k+1])
        if (k >= 2 * N) and (k < 3*N):
            offspring[k] = cross(pop[k], pop[random.randrange(0, 4*N)])
        if (k >= 3 * N):
            offspring[k] = cross(pop[random.randrange(0, 4*N)], pop[random.randrange(0, 4*N)])
        ## next mutate
        offspring[-2:-1] = pop[0]
        offspring[k] = mutate(offspring[k])
        k = k + 1
    return offspring

def run(N = 6, num_generations = 10000):
    em = tetris.TetrisApp(8, 16, 750, False, 40, 30*100)
    em.pcrun()
    net = NN.DQN(em.get_state_size(), 1, losses.MSE_loss)

    dimension = net.L1.W.size + net.L1.B.size + net.L2.W.size + net.L2.B.size + net.L3.W.size + net.L3.B.size
    size_population = 4 * N
    pop_size = (size_population, dimension) 
    new_population = np.random.rand(size_population, dimension)
    fitness = np.ndarray(size_population)

    generations = np.linspace(1, num_generations, num_generations)
    maxscore = np.zeros(num_generations)

    for generation in range(num_generations):
        ## compute the fitness of each individual
        for it, row in enumerate(new_population):
            index = 0
            net.L1.W = row[index:index + net.L1.W.size].reshape(net.L1.W.shape)
            index += net.L1.W.size
            net.L1.B = row[index:index + net.L1.B.size].reshape(net.L1.B.shape)
            index += net.L1.B.size
            net.L2.W = row[index:index + net.L2.W.size].reshape(net.L2.W.shape)
            index += net.L2.W.size
            net.L2.B = row[index:index + net.L2.B.size].reshape(net.L2.B.shape)
            index += net.L2.B.size
            net.L3.W = row[index:index + net.L3.W.size].reshape(net.L3.W.shape)
            index += net.L3.W.size
            net.L3.B = row[index:index + net.L3.B.size].reshape(net.L3.B.shape)
            em.reset()
            done = False
            while not done:
                next_state = em.get_next_states()
                predicted_qs = {}

                for i, (*data,) in enumerate(next_state):
                    predicted_qs[(data[0], data[1])] = net.f_pass(np.array([next_state[data[0], data[1]]]).T)[0,0]

                best_move = max(predicted_qs, key=predicted_qs.get)

                reward, done = em.pcplace(best_move[0], best_move[1])
                if em.get_game_score() > 20000:
                    break

            fitness[it] = em.get_game_score()

        ## sort this such that the best is on top, etc 
        new_population = new_population[fitness.argsort()[::-1]]
        ## help: argsort
        maxscore[generation] = max(fitness)
        print(generation, max(fitness))
        if max(fitness) > 20000:
            break
        np.savetxt("evolution\\generation" + str(generation) + ".csv", new_population[0], delimiter=',')
        offspring_crossover = cross_and_mutate(new_population, pop_size)
        new_population = offspring_crossover
    np.savetxt("evolution\\scores.csv", np.array([generations, maxscore]).T, delimiter=',')
    return(fitness[0], new_population[0])

fit, sol = run(50, 1000)

np.savetxt("evolution\\" + str(fit) + "gene.csv", sol, delimiter=',')
print(fit)
