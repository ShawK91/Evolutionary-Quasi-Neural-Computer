import numpy as np, os
import mod_eqnc as mod, sys
from copy import deepcopy

graph_dim = 4

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
            filename = self.foldername + '/ECM_fitness.csv'
            self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
            np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

    def add_hof_fitness(self, hof_fitness, generation):
        self.hof_fitnesses.append(hof_fitness)
        if len(self.hof_fitnesses) > 100:
            self.hof_fitnesses.pop(0)
        self.hof_avg_fitness = sum(self.hof_fitnesses)/len(self.hof_fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/' + 'hof_fitness.csv'
            self.hof_tr_avg_fit.append(np.array([generation, self.hof_avg_fitness]))
            np.savetxt(filename, np.array(self.hof_tr_avg_fit), fmt='%.3f', delimiter=',')

    def save_csv(self, generation, filename):
        self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
        np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

class Parameters:
    def __init__(self):
            self.population_size = 100
            self.total_gens = 10000
            self.reward_scheme = 1 #1:Exact measure (Test + coarse feedback) #2 Raw performance in graph

            #ECM param
            self.heap_x = graph_dim * 2 + 5
            self.heap_y = graph_dim * graph_dim
            self.graph_dim = graph_dim
            self.hop_limit = graph_dim * 2 + 5
            self.num_input = self.graph_dim * self.graph_dim
            self.num_func = 3
            self.fitness_trials = 5

            #SSNE Stuff
            self.elite_fraction = 0.1
            self.crossover_prob = 0.1
            self.mutation_prob = 0.9
            self.weight_magnitude_limit = 100000000
            self.mut_distribution = 3  # 1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s

parameters = Parameters() #Create the Parameters class
tracker = tracker(parameters) #Initiate tracker


#Get training data
train_upper_bound = 5000
if graph_dim == 4: train_data_folder = 'Train_Data/'
elif graph_dim == 5: train_data_folder = 'Train_Data/5by5/'
elif graph_dim == 7: train_data_folder = 'Train_Data/7by7/'
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
split = int(0.8*len(train_x))
ig = np.split(train_x, [split, len(train_x)])
train_x = ig[0]; test_x = ig[1]
ig = np.split(train_y, [split, len(train_y)])
train_y = ig[0]; test_y = ig[1]
print train_x.shape, test_x.shape, train_y.shape, test_y.shape



class Path_finder:
    def __init__(self, parameters):
        self.parameters = parameters
        self.agent = mod.SSNE(self.parameters)

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

    def get_raw_reward(self, output, target, graph_input):
        output_cost = 0.0
        target_cost = 0.0
        label_length = len(output[0])

        is_match = 1
        for i in range(len(output)-1):
            if not is_match: output_cost -= 100
            else: is_match = 0

            for entry in graph_input:
                #Output computation
                if np.sum(abs(entry[0:label_length] - output[i])) == 0: #from
                    if np.sum(abs(entry[(label_length+1):] - output[i+1])) == 0: #to
                        output_cost -= entry[label_length]
                        is_match = 1
                        break

        return output_cost

    def hof_test(self, graph_input, target_path, individual):
        output_path = individual.feedforward(graph_input)
        reward = self.test_complete(output_path, target_path)
        return reward

    def get_reward(self, output, target, graph_input):

        if self.parameters.reward_scheme == 1:
            reward = 0.0
            for i in range(len(target)):
                if (output[i] == target[i]).all():
                    reward += 1.0

            #Normalize
            reward /= len(target)
            return reward

        elif self.parameters.reward_scheme == 2:
            return self.get_raw_reward(output, target, graph_input)

    def run_simulation(self, graph_input, target_path, individual):
        output_path = individual.feedforward(graph_input)
        reward = self.get_reward(output_path, target_path, graph_input)
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
    print 'Running ECM'
    task = Path_finder(parameters)
    for gen in range(parameters.total_gens):
        epoch_reward, hof_reward = task.evolve(gen)
        print 'Gen:', gen, ' Ep_rew:', "%0.2f" % epoch_reward, ' Hof_rew:', "%0.2f" % hof_reward
        tracker.add_fitness(epoch_reward, gen)  # Add average global performance to tracker
        tracker.add_hof_fitness(hof_reward, gen)  # Add average global performance to tracker













