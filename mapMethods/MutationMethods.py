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
            lambda ind: MutationMethods.bit_flip(ind, mutation_chance),
            MutationMethods.swap_mutation,
            MutationMethods.scramble_mutation,
            MutationMethods.inversion_mutation,
            MutationMethods.duplicate_mutation,
            MutationMethods.insertion_mutation
        ]
        random.choice(methods)(individual)

    @staticmethod
    def inversion_mutation(individual: Individual) -> None:
        """
        Perform inversion mutation by reversing a random subsection of the chromosome.
        """
        individual.inversion_mutation()



    @staticmethod
    def duplicate_mutation(individual: Individual) -> None:
        """
        Apply a random duplication gene mutation method to the individual.
        """
        individual.duplicate_mutation()



    @staticmethod
    def insertion_mutation(individual: Individual) -> None:
        """
        Apply a insertion mutation to the individual.
        """
        individual.insertion_mutation()

