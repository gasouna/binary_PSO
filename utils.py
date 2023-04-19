from random import random
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