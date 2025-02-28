import random
from core.Individual import Individual

class MutationMethods:
    @staticmethod
    def bit_flip(individual: Individual, mutation_chance: float) -> None:
        """
        Perform bit-flip mutation on an individual's chromosome.
        """
        individual.bit_flip(mutation_chance)

    @staticmethod
    def swap_mutation(individual: Individual) -> None:
        """
        Perform swap mutation by swapping two genes in the chromosome.
        """
        individual.swap_mutation()

    @staticmethod
    def scramble_mutation(individual: Individual) -> None:
        """
        Perform scramble mutation by shuffling a subsection of the chromosome.
        """
        individual.scramble_mutation()

    @staticmethod
    def random_mutation(individual: Individual, mutation_chance: float) -> None:
        """
        Apply a random mutation method to the individual.
        """
        methods = [
            lambda ind: MutationMethods.mutation(ind, mutation_chance),
            MutationMethods.swap_mutation,
            MutationMethods.scramble_mutation
        ]
        random.choice(methods)(individual)

    @staticmethod
    def inversion_mutation(individual: Individual) -> None:
        """
        Perform inversion mutation by reversing a random subsection of the chromosome.
        """
        individual.inversion_mutation()