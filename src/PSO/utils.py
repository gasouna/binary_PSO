from random import random, randint
from math import exp

class Utils:

    def __init__(self, number_of_bits, min_value, max_value):

        self.number_of_bits = number_of_bits
        self.min_value = min_value
        self.max_value = max_value

    def decToBin(self, real_input):

        converted_value = ((2 ** self.number_of_bits) - 1) * (real_input - self.min_value) / (self.max_value - self.min_value)

        return '{0:0{1}b}'.format(int(converted_value), self.number_of_bits)
    
    def binToDec(self, binary_input):

        return round(((int(binary_input, 2) * (self.max_value - self.min_value)) / ((2 ** self.number_of_bits) - 1)) + self.min_value, 4)


    def selectBit(self, x):

        individuals = len(x)
        variables = len(x[0])
        new_individuals = []

        for i in range(individuals):

            new_individual = []

            for j in range(variables):

                sig = 1/(1+exp(-int(x[i][j])))

                if random() >= sig:

                    new_individual.append(0)

                else:

                    new_individual.append(1)

            new_individuals.append(''.join(map(str, new_individual)))

        return new_individuals
    
    def mutate(self, x, mutation_rate):

        mutated_x = ""

        for b in x:

            if mutation_rate > random():
        
                if b == '1':
                    mutated_x += '0'
                else:
                    mutated_x += '1'
    
            else:

                mutated_x += b
        
        return mutated_x
    
    def binaryTournamentSelection(self, population, fitness, number_of_children):

        selected_particles = []

        for i in range(number_of_children):
        
            particle_1 = randint(0,len(population)-1)
            particle_2 = randint(0,len(population)-1)

            if fitness[particle_1] <= fitness[particle_2]:
                selected_particles.append(population[particle_1])
            else:
                selected_particles.append(population[particle_2])

        return selected_particles
    
    def singlePointCrossover(self, first_parent, second_parent):

        # Because Python uses len(x) - 1 as the vector's last position we need to use len(x) - 2 to garantee that the break point will split at least one bit from the parents
        break_point = randint(1, len(first_parent) - 2) 

        return first_parent[:break_point] + second_parent[break_point:], second_parent[:break_point] + first_parent[break_point:]
