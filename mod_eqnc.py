from random import randint
import math
import numpy as np
import random
import copy
from copy import deepcopy
#from scipy.special import expit
import sys,os, cPickle


def generate_training_data(graph_dim, x_raw, y_raw):
    train_x = []; train_y = []



    #One hot representation
    decorator_one_hot = [0]*graph_dim**2
    for i in range(graph_dim):
        for j in range(graph_dim):
            row_x = []
            #Left
            row_x += decorator_one_hot
            row_x[i * 4 + j] = 1 #Self label
            if j == 0: #Edge
                row_x.append(100)
                row_x += decorator_one_hot
            else:
                if x_raw[i][j-1] == -1: row_x.append(100)
                else: row_x.append(1)
                ig = decorator_one_hot[:]
                ig[i * 4 + j - 1] = 1
                row_x += ig

            if i == 0 and j == 0: x = np.reshape(np.array(row_x), (1, len(row_x)))
            else: x = np.concatenate((x, np.reshape(np.array(row_x), (1, len(row_x)))), axis=0)
            row_x = []

            #Right
            row_x += decorator_one_hot
            row_x[i * 4 + j] = 1 #Self label
            if j == graph_dim-1: #Edge
                row_x.append(100)
                row_x += decorator_one_hot
            else:
                if x_raw[i][j+1] == -1: row_x.append(100)
                else: row_x.append(1)
                ig = decorator_one_hot[:]
                ig[i * 4 + j + 1] = 1
                row_x += ig

            x = np.concatenate((x, np.reshape(np.array(row_x), (1, len(row_x)))), axis=0)
            row_x = []

            #Up
            row_x += decorator_one_hot
            row_x[i * 4 + j] = 1 #Self label
            if i == 0: #Edge
                row_x.append(100)
                row_x += decorator_one_hot
            else:
                if x_raw[i-1][j] == -1: row_x.append(100)
                else: row_x.append(1)
                ig = decorator_one_hot[:]
                ig[(i-1) * 4 + j] = 1
                row_x += ig

            x = np.concatenate((x, np.reshape(np.array(row_x), (1, len(row_x)))), axis=0)
            row_x = []

            #Down
            row_x += decorator_one_hot
            row_x[i * 4 + j] = 1 #Self label
            if i == graph_dim-1: #Edge
                row_x.append(100)
                row_x += decorator_one_hot
            else:
                if x_raw[i+1][j] == -1: row_x.append(100)
                else: row_x.append(1)
                ig = decorator_one_hot[:]
                ig[(i+1) * 4 + j] = 1
                row_x += ig

            x = np.concatenate((x, np.reshape(np.array(row_x), (1, len(row_x)))), axis=0)
            row_x = []



    #Target path
    y = []
    for label in y_raw:
        ig = decorator_one_hot[:]
        ig[int((label[0]-1) * 4) + int(label[1]-1)] = 1
        y.append(ig)


    y = np.array(y)
    return x,y

class ROM: #Deal with input graph (store it in a Read Only Memory)
    def __init__(self):
        self.graph = None
        self.label_length = None

    def get_graph(self, graph):
        self.graph = copy.deepcopy(graph)
        self.label_length = (graph.shape[1]-1)/2
        start = graph[-1][0:self.label_length]
        end = graph[0][self.label_length+1:]
        return start, end

    def associative_recall(self, query):
        #Sharpen
        query = (query > 0.5).astype(np.int)

        result = []
        for entry in self.graph:
            if np.sum(abs(entry[0:self.label_length] - query)) == 0:
                result.append(entry[self.label_length:])

        while len(result) != 4:
            result.append([0]*(self.label_length+1))

        return np.mat(np.array(result))

class Memory_Bank:
    def __init__(self, heap_x, heap_y):
        self.heap_x = heap_x; self.heap_y = heap_y
        self.heap = np.zeros((heap_x, heap_y))+100
        self.write_ptr = 0
        self.read_ptr = 0

    def addressing(self, w_c):
        #Location based
        return int(round(w_c))

        #TODO Content based addressing

    def read(self, w_c):
        self.read_ptr += self.addressing(w_c)
        self.write_ptr = self.write_ptr % self.heap_x
        return self.heap[self.read_ptr,:]

    def write(self, w_c, mem_new):
        self.write_ptr += self.addressing(w_c)
        self.write_ptr = self.write_ptr % self.heap_x
        #self.write_ptr += 1
        self.heap[self.write_ptr, :] = mem_new
        #print self.write_ptr

    def reset(self):
        self.heap = np.zeros((self.heap_x, self.heap_y))+100
        self.write_ptr = 0
        self.read_ptr = 0

    def get_output(self, path_len):
        # TODO NEURALIZE
        return self.heap[0:path_len, :]

class Primitive_fnc_mod:
    def __init__(self):
        self.key = None



    def func_operation(self, func_keys, *args):
        self.key = np.argmax(func_keys)
        if self.key == 2:
            return self.sort_ascending(args)

        if self.key == 1:
            #return self.random_sort(args)
            return self.similarity(args)

        if self.key == 0:
            return self.one_hot_min(args)

    def similarity(self, args):
        return np.sum((args[0] == args[1]).astype(int), axis=1)

    def one_hot_min(self, args):
        ig = args[0]
        if ig.shape[1] != 1: ig = np.sum(ig, axis=1)
        min = np.min(ig)
        return (ig == min).astype(int)

    def addition(self, i, j):
        return i + j

    def substraction(self, i, j):
        return i - j

    def multiply(self, i, j):
        return i * j

    def divide(self, i, j):
        return i/j

    def is_greater(self, i, j):
        return i>j

    def is_equal(self, i, j):
        return i == j

    def l2(self, i, j):
        return LA.norm(i - j)

    def sort_ascending(self, args):
        ig = np.argsort(args[0])
        if ig.shape[1] != 1: ig = np.sum(ig, axis=1)
        return ig

    def random_sort(self, args):
        ig = np.arange(args[0].shape[0])
        np.random.shuffle(ig)
        return np.mat(ig).transpose()

class Controller:
    def __init__(self, num_input, num_function, mean = 0, std = 1):
        #self.arch_type = 'quasi_ntm'
        self.num_input = num_input
        self.mem_size = num_input * 2 + 1
        self.num_function = num_function

        #Adaptive components (plastic with network running)
        #self.last_output = np.mat(np.zeros(num_output)).transpose()
        self.memory_cell = np.mat(np.zeros(self.mem_size)) #Memory Cell

        #Write key generator
        self.w_write_key = np.mat(np.random.normal(mean, std, 1 * (self.mem_size + self.num_input)))
        self.w_write_key = np.mat(np.reshape(self.w_write_key, ((self.mem_size + self.num_input), 1)))

        #Post Query - preprocess to functional operator (PARSE QUERY)
        self.w_pre_func_1 = np.mat(np.random.normal(mean, std, (self.num_input+1)*self.num_input))
        self.w_pre_func_1 = np.mat(np.reshape(self.w_pre_func_1, ((self.num_input + 1), self.num_input)))
        self.w_pre_func_2 = np.mat(np.random.normal(mean, std, (self.num_input+1)))
        self.w_pre_func_2 = np.mat(np.reshape(self.w_pre_func_2, ((self.num_input + 1), 1)))

        #Shape functiona arguments
        self.w_shape_args = np.mat(np.random.normal(mean, std, 2 * (1)))
        self.w_shape_args = np.mat(np.reshape(self.w_shape_args, (2, 1)))

        #Choice of function operator
        self.w_func_choice_1 = np.mat(np.random.normal(mean, std,(self.num_input*4) * (self.num_function)))
        self.w_func_choice_1 = np.mat(np.reshape(self.w_func_choice_1, ((self.num_input*4), self.num_function)))
        self.w_func_choice_2 = np.mat(np.random.normal(mean, std,(self.mem_size+self.num_function+1) * (self.num_function)))
        self.w_func_choice_2 = np.mat(np.reshape(self.w_func_choice_2, ((self.mem_size+self.num_function+1), self.num_function)))

        #Process h to next labels
        self.w_next_labels = np.mat(np.random.normal(mean, std,(self.num_input + 1) * (self.num_input)))
        self.w_next_labels = np.mat(np.reshape(self.w_next_labels, ((self.num_input + 1), self.num_input)))

        #Compute new input
        self.w_update_inp = np.mat(np.random.normal(mean, std,(self.num_input + 1) * (self.num_input)))
        self.w_update_inp = np.mat(np.reshape(self.w_update_inp, ((self.num_input + 1), self.num_input)))


    def get_func_args(self, h):
        return self.linear_combination(h, self.w_pre_func_1), self.linear_combination(h, self.w_pre_func_2)

    def shape_f_args(self, f_out_1, f_args_2):
        ig = np.concatenate((f_out_1, f_args_2), axis=1)
        return self.linear_combination(ig, self.w_shape_args)



    def get_write_key(self, input):
        ig = np.concatenate((input, self.memory_cell), axis = 1)
        return np.tanh(self.linear_combination(ig, self.w_write_key))

    def get_func_key(self, args_1, args_2):
        ig = np.reshape(args_1, (1, args_1.shape[0]*args_1.shape[1]))
        keys_1 = self.linear_combination(ig, self.w_func_choice_1)
        ig = np.concatenate((args_2.transpose(), self.memory_cell), axis=1)
        keys_2 = self.linear_combination(ig, self.w_func_choice_2)
        return keys_1, keys_2


    def update_input(self, h, post_func):
        h_prime = np.delete(h, 0, 1)
        return self.linear_combination(post_func.transpose(), h_prime)


    def linear_combination(self, w_matrix, layer_input): #Linear combine weights with inputs
        return np.dot(w_matrix, layer_input) #Linear combination of weights and inputs

    # def relu(self, layer_input):    #Relu transformation function
    #     for x in range(len(layer_input)):
    #         if layer_input[x] < 0:
    #             layer_input[x] = 0
    #     return layer_input
    #
    # def fast_sigmoid(self, layer_input): #Sigmoid transform
    #     layer_input = expit(layer_input)
    #     return layer_input
    #
    # def softmax(self, layer_input): #Softmax transform
    #     layer_input = np.exp(layer_input)
    #     layer_input = layer_input / np.sum(layer_input)
    #     return layer_input

    # def format_input(self, input, add_bias = True): #Formats and adds bias term to given input at the end
    #     if add_bias:
    #         input = np.concatenate((input, [1.0]))
    #     return np.mat(input)
    #
    # def format_memory(self, memory):
    #     ig = np.mat([1])
    #     return np.concatenate((memory, ig))
    #
    # #Memory_write gate
    # def feedforward(self, input): #Feedforwards the input and computes the forward pass of the network
    #     self.input = self.format_input(input).transpose()  # Format and add bias term at the end
    #     last_memory = self.format_memory(self.memory_cell)
    #     last_output = self.format_memory(self.last_output)
    #
    #     #Input gate
    #     ig_1 = self.linear_combination(self.w_inpgate, self.input)
    #     ig_2 = self.linear_combination(self.w_rec_inpgate, last_output)
    #     ig_3 = self.linear_combination(self.w_mem_inpgate, last_memory)
    #     input_gate_out = ig_1 + ig_2 + ig_3
    #     input_gate_out = self.fast_sigmoid(input_gate_out)
    #
    #     #Input processing
    #     ig_1 = self.linear_combination(self.w_inp, self.input)
    #     ig_2 = self.linear_combination(self.w_rec_inp, last_output)
    #     block_input_out = ig_1 + ig_2
    #     block_input_out = self.fast_sigmoid(block_input_out)
    #
    #     #Gate the Block Input and compute the final input out
    #     input_out = np.multiply(input_gate_out, block_input_out)
    #
    #     #Forget Gate
    #     ig_1 = self.linear_combination(self.w_forgetgate, self.input)
    #     ig_2 = self.linear_combination(self.w_rec_forgetgate, last_output)
    #     ig_3 = self.linear_combination(self.w_mem_forgetgate, last_memory)
    #     forget_gate_out = ig_1 + ig_2 + ig_3
    #     forget_gate_out = self.fast_sigmoid(forget_gate_out)
    #
    #     #Memory Output
    #     memory_output = np.multiply(forget_gate_out, self.memory_cell)
    #
    #     #Compute hidden activation - processing hidden output for this iteration of net run
    #     hidden_act = memory_output + input_out
    #
    #     #Write gate (memory cell)
    #     ig_1 = self.linear_combination(self.w_writegate, self.input)
    #     ig_2 = self.linear_combination(self.w_rec_writegate, last_output)
    #     ig_3 = self.linear_combination(self.w_mem_writegate, last_memory)
    #     write_gate_out = ig_1 + ig_2 + ig_3
    #     write_gate_out = self.fast_sigmoid(write_gate_out)
    #
    #     #Write to memory Cell - Update memory
    #     self.memory_cell += np.multiply(write_gate_out, np.tanh(hidden_act))
    #
    #
    #     # if input != 0:
    #     #     ig_read_gate = [forget_gate_out[2,0], forget_gate_out[4,0]]
    #     #     ig_write_gate = [write_gate_out[2,0], write_gate_out[4,0]]
    #     #     ig_hidden = [hidden_act[2,0], hidden_act[4,0]]
    #     #     lalal = 1
    #     #temp = np.multiply(write_gate_out, hidden_act)
    #     #temp = self.fast_sigmoid(temp)
    #     #self.memory_cell += temp
    #
    #     #Compute final output
    #     hidden_act = self.format_memory(hidden_act)
    #     self.last_output = self.linear_combination(self.w_output, hidden_act)
    #     self.last_output = self.fast_sigmoid(self.last_output)
    #     #print self.last_output
    #     return np.array(self.last_output).tolist()

    def reset(self):
        self.memory_cell = np.mat(np.zeros(self.mem_size))  # Memory Cell

class ECM:
    def __init__(self, parameters):
        self.rom = ROM()
        self.prim_fnc = Primitive_fnc_mod()
        self.mem_bank = Memory_Bank(parameters.heap_x, parameters.heap_y)
        self.controller = Controller(parameters.num_input, parameters.num_func)
        self.hop_limit = parameters.hop_limit


    def feedforward(self, input_graph):

        # Load ROM (Get graph to plan a path on)
        start, end = self.rom.get_graph(input_graph)
        input = np.mat(np.array(start))
        last_input = input[:]

        #Reset memory bank and controller memory
        self.mem_bank.reset()
        self.controller.reset()

        #Loop
        for hop in range(self.hop_limit): #TODO NEURALIZE
            #Write stuff
            write_key = self.controller.get_write_key(input) #Get write key
            self.mem_bank.write(write_key, input) #Write to memory

            #Query phase
            h = self.rom.associative_recall(input)

            #print h.shape

            #Primitive Function Phase
            f_args_1, f_args_2 = self.controller.get_func_args(h) #Process/Parse Query
            func_keys_1, func_keys_2 = self.controller.get_func_key(f_args_1, f_args_2) #Get Functional head keys

            f_out_1 = self.prim_fnc.func_operation(func_keys_1, f_args_1, last_input)
            final_args = self.controller.shape_f_args(f_out_1, f_args_2)

            post_func = self.prim_fnc.func_operation(func_keys_1, final_args, last_input)



            #print args.shape, func_keys.shape, post_func.shape

            #Update input for next hop
            last_input = input[:]
            input = self.controller.update_input(h, post_func)

            #Update memory
            #TODO MEMORY UPDATE

        #Get output
        output = self.mem_bank.get_output(self.hop_limit)
        return output

class SSNE:
        def __init__(self, parameters):
            self.current_gen = 0
            self.parameters = parameters;
            self.population_size = self.parameters.population_size;
            self.num_elitists = int(self.parameters.elite_fraction * parameters.population_size)
            if self.num_elitists < 1: self.num_elitists = 1

            self.fitness_evals = [[] for x in xrange(parameters.population_size)]  # Fitness eval list
            # Create population
            self.pop = []
            for i in range(self.population_size):
                self.pop.append(ECM(self.parameters))
            self.hof_net = ECM(self.parameters)
            self.num_substructures = 8

        def selection_tournament(self, index_rank, num_offsprings, tournament_size):
            total_choices = len(index_rank)
            offsprings = []
            for i in range(num_offsprings):
                winner = np.min(np.random.randint(total_choices, size=tournament_size))
                offsprings.append(index_rank[winner])

            offsprings = list(set(offsprings))  # Find unique offsprings
            if len(offsprings) % 2 != 0:  # Number of offsprings should be even
                offsprings.append(offsprings[randint(0, len(offsprings) - 1)])
            return offsprings

        def list_argsort(self, seq):
            return sorted(range(len(seq)), key=seq.__getitem__)

        def crossover_inplace(self, gene1, gene2):
            gene1 = gene1.controller
            gene2 = gene2.controller
            # INPUT GATES
            # Layer 1
            num_cross_overs = randint(1, len(gene1.w_write_key))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_write_key) - 1)
                    gene1.w_write_key[ind_cr, :] = gene2.w_write_key[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_write_key) - 1)
                    gene2.w_write_key[ind_cr, :] = gene1.w_write_key[ind_cr, :]
                else:
                    continue

            # Layer 2
            num_cross_overs = randint(1, len(gene1.w_pre_func_1))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_pre_func_1) - 1)
                    gene1.w_pre_func_1[ind_cr, :] = gene2.w_pre_func_1[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_pre_func_1) - 1)
                    gene2.w_pre_func_1[ind_cr, :] = gene1.w_pre_func_1[ind_cr, :]
                else:
                    continue

            # Layer 2
            num_cross_overs = randint(1, len(gene1.w_pre_func_2))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_pre_func_2) - 1)
                    gene1.w_pre_func_2[ind_cr, :] = gene2.w_pre_func_2[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_pre_func_2) - 1)
                    gene2.w_pre_func_2[ind_cr, :] = gene1.w_pre_func_2[ind_cr, :]
                else:
                    continue

            # Layer 3
            num_cross_overs = randint(1, len(gene1.w_func_choice_1))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_func_choice_1) - 1)
                    gene1.w_func_choice_1[ind_cr, :] = gene2.w_func_choice_1[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_func_choice_1) - 1)
                    gene2.w_func_choice_1[ind_cr, :] = gene1.w_func_choice_1[ind_cr, :]
                else:
                    continue

            # Layer 3
            num_cross_overs = randint(1, len(gene1.w_func_choice_2))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_func_choice_2) - 1)
                    gene1.w_func_choice_2[ind_cr, :] = gene2.w_func_choice_2[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_func_choice_2) - 1)
                    gene2.w_func_choice_2[ind_cr, :] = gene1.w_func_choice_2[ind_cr, :]
                else:
                    continue

            # BLOCK INPUTS
            # Layer 1
            num_cross_overs = randint(1, len(gene1.w_next_labels))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_next_labels) - 1)
                    gene1.w_next_labels[ind_cr, :] = gene2.w_next_labels[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_next_labels) - 1)
                    gene2.w_next_labels[ind_cr, :] = gene1.w_next_labels[ind_cr, :]
                else:
                    continue

            # BLOCK INPUTS
            # Layer 1
            num_cross_overs = randint(1, len(gene1.w_shape_args))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_shape_args) - 1)
                    gene1.w_shape_args[ind_cr, :] = gene2.w_shape_args[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_shape_args) - 1)
                    gene2.w_shape_args[ind_cr, :] = gene1.w_shape_args[ind_cr, :]
                else:
                    continue

            # Layer 2
            num_cross_overs = randint(1, len(gene1.w_update_inp))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_update_inp) - 1)
                    gene1.w_update_inp[ind_cr, :] = gene2.w_update_inp[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_update_inp) - 1)
                    gene2.w_update_inp[ind_cr, :] = gene1.w_update_inp[ind_cr, :]
                else:
                    continue

        def regularize_weight(self, weight):
            if weight > self.parameters.weight_magnitude_limit:
                weight = self.parameters.weight_magnitude_limit
            if weight < -self.parameters.weight_magnitude_limit:
                weight = -self.parameters.weight_magnitude_limit
            return weight

        def mutate_inplace(self, gene):
            gene = gene.controller
            mut_strength = 0.2
            num_mutation_frac = 0.2
            super_mut_strength = 10
            super_mut_prob = 0.05

            # Initiate distribution
            if self.parameters.mut_distribution == 1:  # Gaussian
                ss_mut_dist = np.random.normal(random.random(), random.random() / 2, self.num_substructures)
            elif self.parameters.mut_distribution == 2:  # Laplace
                ss_mut_dist = np.random.normal(random.random(), random.random() / 2, self.num_substructures)
            elif self.parameters.mut_distribution == 3:  # Uniform
                ss_mut_dist = np.random.uniform(0, 1, self.num_substructures)
            else:
                ss_mut_dist = [1] * self.num_substructures


            # INPUT GATES
            # Layer 1
            if random.random() < ss_mut_dist[0]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_write_key.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_write_key.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_write_key.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_write_key[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength * gene.w_write_key[
                            ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_write_key[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_write_key[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_write_key[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_write_key[ind_dim1, ind_dim2])

            # Layer 2
            if random.random() < ss_mut_dist[1]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_pre_func_1.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_pre_func_1.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_pre_func_1.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_pre_func_1[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength *
                                                                               gene.w_pre_func_1[
                                                                                   ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_pre_func_1[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_pre_func_1[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_pre_func_1[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_pre_func_1[ind_dim1, ind_dim2])

            # Layer 2
            if random.random() < ss_mut_dist[1]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_pre_func_2.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_pre_func_2.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_pre_func_2.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_pre_func_2[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength *
                                                                              gene.w_pre_func_2[
                                                                                  ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_pre_func_2[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_pre_func_2[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_pre_func_2[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_pre_func_2[ind_dim1, ind_dim2])

            # Layer 3
            if random.random() < ss_mut_dist[2]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_func_choice_1.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_func_choice_1.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_func_choice_1.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_func_choice_1[ind_dim1, ind_dim2] += random.gauss(0,
                                                                               super_mut_strength *
                                                                               gene.w_func_choice_1[
                                                                                   ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_func_choice_1[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_func_choice_1[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_func_choice_1[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_func_choice_1[ind_dim1, ind_dim2])

            # Layer 3
            if random.random() < ss_mut_dist[2]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_func_choice_2.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_func_choice_2.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_func_choice_2.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_func_choice_2[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                 super_mut_strength *
                                                                                 gene.w_func_choice_2[
                                                                                     ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_func_choice_2[ind_dim1, ind_dim2] += random.gauss(0, mut_strength *
                                                                                 gene.w_func_choice_2[
                                                                                     ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_func_choice_2[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_func_choice_2[ind_dim1, ind_dim2])

            # Layer 3
            if random.random() < ss_mut_dist[2]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_shape_args.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_shape_args.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_shape_args.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_shape_args[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                 super_mut_strength *
                                                                                 gene.w_shape_args[
                                                                                     ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_shape_args[ind_dim1, ind_dim2] += random.gauss(0, mut_strength *
                                                                                 gene.w_shape_args[
                                                                                     ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_shape_args[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_shape_args[ind_dim1, ind_dim2])

            # BLOCK INPUTS
            # Layer 1
            if random.random() < ss_mut_dist[3]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_next_labels.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_next_labels.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_next_labels.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_next_labels[ind_dim1, ind_dim2] += random.gauss(0,
                                                                       super_mut_strength * gene.w_next_labels[
                                                                           ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_next_labels[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_next_labels[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_next_labels[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_next_labels[ind_dim1, ind_dim2])

            # Layer 2
            if random.random() < ss_mut_dist[4]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_update_inp.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_update_inp.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_update_inp.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_update_inp[ind_dim1, ind_dim2] += random.gauss(0,
                                                                           super_mut_strength * gene.w_update_inp[
                                                                               ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_update_inp[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_update_inp[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_update_inp[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_update_inp[ind_dim1, ind_dim2])

        def epoch(self):

            self.current_gen += 1
            # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
            index_rank = self.list_argsort(self.fitness_evals);
            index_rank.reverse()
            elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

            # Selection step
            offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                                   tournament_size=3)

            # Figure out unselected candidates
            unselects = [];
            new_elitists = []
            for i in range(self.population_size):
                if i in offsprings or i in elitist_index:
                    continue
                else:
                    unselects.append(i)
            random.shuffle(unselects)

            # Elitism step, assigning eleitst candidates to some unselects
            for i in elitist_index:
                replacee = unselects.pop(0)
                new_elitists.append(replacee)
                self.pop[replacee] = copy.deepcopy(self.pop[i])

            # Crossover for unselected genes with 100 percent probability
            if len(unselects) % 2 != 0:  # Number of unselects left should be even
                unselects.append(unselects[randint(0, len(unselects) - 1)])
            for i, j in zip(unselects[0::2], unselects[1::2]):
                off_i = random.choice(new_elitists);
                off_j = random.choice(offsprings)
                self.pop[i] = copy.deepcopy(self.pop[off_i])
                self.pop[j] = copy.deepcopy(self.pop[off_j])
                self.crossover_inplace(self.pop[i], self.pop[j])

            # Crossover for selected offsprings
            for i, j in zip(offsprings[0::2], offsprings[1::2]):
                if random.random() < self.parameters.crossover_prob: self.crossover_inplace(self.pop[i], self.pop[j])

            # Mutate all genes in the population except the new elitists
            for i in range(self.population_size):
                if i not in new_elitists:  # Spare the new elitists
                    if random.random() < self.parameters.mutation_prob:
                        self.mutate_inplace(self.pop[i])


        def save_pop(self, filename='Pop'):
            filename = str(self.current_gen) + '_' + filename
            pickle_object(self.pop, filename)

def unpickle(filename = 'def.pickle'):
    import pickle
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def pickle_object(obj, filename):
    with open(filename, 'wb') as output:
        cPickle.dump(obj, output, -1)