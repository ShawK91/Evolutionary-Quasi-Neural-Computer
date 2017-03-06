import numpy as np, os
import mod_eqnc as mod, sys
from numpy import linalg as LA
from random import randint
import random

save_foldername = 'R_EQNC'
class tracker(): #Tracker
    def __init__(self, parameters, foldername = save_foldername):
        self.foldername = foldername
        self.fitnesses = []; self.avg_fitness = 0; self.tr_avg_fit = []
        self.hof_fitnesses = []; self.hof_avg_fitness = 0; self.hof_tr_avg_fit = []
        if not os.path.exists(foldername):
            os.makedirs(foldername)
            self.file_save = 'ECM.csv'


    def add_fitness(self, fitness, generation):
        self.fitnesses.append(fitness)
        if len(self.fitnesses) > 100:
            self.fitnesses.pop(0)
        self.avg_fitness = sum(self.fitnesses)/len(self.fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/' + 'rough_' + self.file_save
            self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
            np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

    def add_hof_fitness(self, hof_fitness, generation):
        self.hof_fitnesses.append(hof_fitness)
        if len(self.hof_fitnesses) > 100:
            self.hof_fitnesses.pop(0)
        self.hof_avg_fitness = sum(self.hof_fitnesses)/len(self.hof_fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/' + 'hof_' + self.file_save
            self.hof_tr_avg_fit.append(np.array([generation, self.hof_avg_fitness]))
            np.savetxt(filename, np.array(self.hof_tr_avg_fit), fmt='%.3f', delimiter=',')

    def save_csv(self, generation, filename):
        self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
        np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')


class Parameters:
    def __init__(self):
            self.population_size = 100
            self.total_gens = 10000

            #Reward scheme
            self.reward_scheme = 6

            #ECM param
            self.heap_x = 50
            self.heap_y = 16
            self.graph_dim = 4
            self.hop_limit = 6
            self.num_input = self.graph_dim * self.graph_dim
            self.num_func = 3

            #SSNE Stuff
            self.elite_fraction = 0.1
            self.crossover_prob = 0.1
            self.mutation_prob = 0.9
            self.weight_magnitude_limit = 1000000000000
            self.mut_distribution = 3  # 1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s




parameters = Parameters() #Create the Parameters class
tracker = tracker(parameters) #Initiate tracker



#Get training data
#train_x = np.ones((10, parameters.graph_dim**2 * 4, parameters.graph_dim**2 * 2 + 1))
train_y = np.ones((1, parameters.graph_dim * 2 - 2, parameters.graph_dim**2))

l = []
for i in range(parameters.graph_dim**2):
    l.append(np.zeros(parameters.graph_dim**2 * 2 + 1)+i)
    l.append(np.zeros(parameters.graph_dim ** 2 * 2 + 1) + i)
    l.append(np.zeros(parameters.graph_dim ** 2 * 2 + 1) + i)
    l.append(np.zeros(parameters.graph_dim ** 2 * 2 + 1) + i)

train_x = np.array(l)
train_x = np.reshape(train_x, (1, parameters.graph_dim**2 * 4, parameters.graph_dim**2 * 2 + 1))






class Path_finder:
    def __init__(self, parameters):
        self.parameters = parameters
        self.agent = mod.SSNE(self.parameters)



    def get_reward(self, output, target):
        reward = 0
        for i in range(len(target)):
            reward -= abs(np.sum(output[i] - target[i]))

        #Special penalty for not reaching goal
        reward -= 10 * abs(np.sum(output[-1] - target[-1]))
        return reward

    def run_simulation(self, graph_input, target_path, individual):
        output_path = individual.feedforward(graph_input)
        reward = self.get_reward(output_path, target_path)
        return reward

    def evolve(self, gen):
        best_epoch_reward = -1000000000
        rand_map_choice = np.random.choice(len(train_x))


        for i in range(self.parameters.population_size): #Test all genomes/individuals
            reward = self.run_simulation(train_x[rand_map_choice], train_y[rand_map_choice], self.agent.pop[i])
            self.agent.fitness_evals[i] = reward
            #print reward
            if reward > best_epoch_reward: best_epoch_reward = reward

        # #HOF test net
        # hof_index = self.agent.fitness_evals.index(max(self.agent.fitness_evals))
        # hof_score = self.test_net(hof_index)

        #Save population and HOF
        if (gen + 1) % 1 == 0:
            mod.pickle_object(self.agent.pop, save_foldername + '/seq_recall_pop')
            #mod.pickle_object(self.agent.pop[hof_index], save_foldername + '/seq_recall_hof')
            np.savetxt(save_foldername + '/gen_tag', np.array([gen + 1]), fmt='%.3f', delimiter=',')

        self.agent.epoch()
        return best_epoch_reward



if __name__ == "__main__":
    print 'Running ECM'
    task = Path_finder(parameters)
    for gen in range(parameters.total_gens):

        #task.agent.pop[0].feedforward(train_x[0])

        epoch_reward = task.evolve(gen)
        print 'Gen:', gen, ' Ep_rew:', "%0.2f" % epoch_reward#, ' Cml_Hof_rew:', "%0.2f" % tracker.hof_avg_fitness
        #tracker.add_fitness(epoch_reward, gen)  # Add average global performance to tracker
        #tracker.add_hof_fitness(hof_score, gen)  # Add average global performance to tracker













