import numpy as np, os
import mod_eqnc as mod, sys
from copy import deepcopy



class Parameters:
    def __init__(self):
            self.population_size = 500
            self.total_gens = 10000

            #ECM param
            self.heap_x = 9
            self.heap_y = 16
            self.graph_dim = 4
            self.hop_limit = 9
            self.num_input = self.graph_dim * self.graph_dim
            self.num_func = 3
            self.fitness_trials = 5

            #SSNE Stuff
            self.elite_fraction = 0.1
            self.crossover_prob = 0.1
            self.mutation_prob = 0.9
            self.weight_magnitude_limit = 1000000000000
            self.mut_distribution = 3  # 1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s

parameters = Parameters() #Create the Parameters class

#Get training data
train_upper_bound = 2000
train_data_folder = 'Train_Data/'
train_x = []; train_y = []
for i in range(1, train_upper_bound+1):
    x_filename = train_data_folder + 'maps/' + 'map' + str(i) +'.csv'
    y_filename = train_data_folder + 'paths/' + 'path' + str(i) + '.csv'

    try:
        x_raw = np.loadtxt(x_filename, delimiter=',')
        y_raw = np.loadtxt(y_filename, delimiter=',')
    except:
        continue

    x, y = mod.generate_training_data(parameters.graph_dim, x_raw, y_raw)
    train_x.append(x)
    train_y.append(y)

#Array format
train_x = np.array(train_x)
train_y = np.array(train_y)

#Train-Test split
split = int(0.2*len(train_x))
ig = np.split(train_x, [split, len(train_x)])
train_x = ig[0]; test_x = ig[1]
ig = np.split(train_y, [split, len(train_y)])
train_y = ig[0]; test_y = ig[1]
print train_x.shape, test_x.shape, train_y.shape, test_y.shape



class Path_finder:
    def __init__(self, parameters):
        self.parameters = parameters

    def test_all(self, hof):
        hof_reward = 0.0; hof_reward_test = 0.0
        reward = 0.0; reward_test = 0.0
        all_choices = np.arange(len(train_x))
        for choice in all_choices:
            hof_reward += self.hof_test(train_x[choice], train_y[choice], hof)
            reward += self.run_simulation(train_x[choice], train_y[choice], hof)
            hof_reward_test += self.hof_test(test_x[choice], test_y[choice], hof)
            reward_test += self.run_simulation(test_x[choice], test_y[choice], hof)
        hof_reward /= len(train_x)
        reward /= len(train_x)
        hof_reward_test /= len(test_x)
        reward_test /= len(test_x)

        return hof_reward, reward, hof_reward_test, reward_test

    def test_complete(self, output, target):
        is_complete = 1
        for i in range(len(target)):
            if (output[i] == target[i]).all():
                continue
            else:
                is_complete = 0
                break
        if is_complete: reward = 1.0
        else: reward = 0.0
        return reward

    def hof_test(self, graph_input, target_path, individual):
        output_path = individual.feedforward(graph_input)
        reward = self.test_complete(output_path, target_path)
        return reward

    def get_reward(self, output, target):
        reward = 0.0
        for i in range(len(target)):
            if (output[i] == target[i]).all():
                reward += 1.0

        #Normalize
        reward /= len(target)
        return reward

    def run_simulation(self, graph_input, target_path, individual):
        output_path = individual.feedforward(graph_input)
        reward = self.get_reward(output_path, target_path)
        return reward

    def evolve(self, gen):
        best_epoch_reward = -1000000000
        rand_map_choices = np.random.choice(len(train_x), parameters.fitness_trials, replace=False)

        for i in range(self.parameters.population_size): #Test all genomes/individuals
            reward = 0
            for choice in rand_map_choices:
                reward += self.run_simulation(train_x[choice], train_y[choice], self.agent.pop[i])
            reward /= parameters.fitness_trials
            self.agent.fitness_evals[i] = reward
            if reward > best_epoch_reward: best_epoch_reward = reward

        #HOF test net
        hof_index = self.agent.fitness_evals.index(max(self.agent.fitness_evals))
        hof_reward = 0.0
        for choice in rand_map_choices:
            hof_reward += self.hof_test(train_x[choice], train_y[choice], self.agent.pop[hof_index])
        hof_reward /= len(rand_map_choices)


        #self.agent.hof_net = deepcopy(self.agent.pop[hof_index])

        #Save population and HOF
        if (gen + 1) % 1 == 0:
            mod.pickle_object(self.agent.pop, save_foldername + '/seq_recall_pop')
            mod.pickle_object(self.agent.pop[hof_index], save_foldername + '/seq_recall_hof')
            np.savetxt(save_foldername + '/gen_tag', np.array([gen + 1]), fmt='%.3f', delimiter=',')

        self.agent.epoch()
        return best_epoch_reward, hof_reward

if __name__ == "__main__":
    print 'Running ECM TEST'

    hof = mod.unpickle('R_EQNC/seq_recall_hof')
    task = Path_finder(parameters)
    hof_reward, reward, hof_reward_test, reward_test = task.test_all(hof)
    print 'HOF REWARD:', hof_reward, ' HOF REWARD TEST:', hof_reward_test
    print 'REWARD:', reward, ' REWARD TEST:', reward_test














