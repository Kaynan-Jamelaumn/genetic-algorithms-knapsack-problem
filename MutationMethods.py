import random


class MutationMethods:
    @staticmethod
    def bit_flip(individual, mutation_chance):
        """
        Perform bit-flip mutation on an individual's chromosome.
        """
        individual.bit_flip(mutation_chance)

    @staticmethod
    def swap_mutation(individual):
        """
        Perform swap mutation by swapping two genes in the chromosome.
        """
        individual.swap_mutation()

    @staticmethod
    def scramble_mutation(individual):
        """
        Perform scramble mutation by shuffling a subsection of the chromosome.
        """
        individual.scramble_mutation()

    @staticmethod
    def random_mutation(individual, mutation_chance):
        """
        Apply a random mutation method to the individual.
        """
        methods = [
            lambda ind: MutationMethods.mutation(ind, mutation_chance),
            MutationMethods.swap_mutation,
            MutationMethods.scramble_mutation
        ]
        random.choice(methods)(individual)
