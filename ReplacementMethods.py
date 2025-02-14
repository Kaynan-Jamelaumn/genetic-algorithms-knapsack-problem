
import random
from Individual import Individual

class ReplacementMethods:

    @staticmethod
    def random_individual_replacement(island: list[Individual], number_of_individuals: int = None) -> int | list[int]:
        """
        Selects a random individual(s) from the island for replacement.
        
        :param island: The destination island population.
        :param number_of_individuals: Number of individuals to replace. If None, replaces one individual.
        :return: Index or list of indexes of the individual(s) to replace.
        """
        if number_of_individuals is None or number_of_individuals == 0:
            return random.randint(0, len(island) - 1)
        else:
            return random.sample(range(len(island)), number_of_individuals)

    @staticmethod
    def worst_individual_replacement(island: list[Individual], number_of_individuals: int = None) -> int | list[int]:
        """
        Selects the worst individual(s) from the island for replacement.
        
        :param island: The destination island population.
        :param number_of_individuals: Number of individuals to replace. If None, replaces one individual.
        :return: Index or list of indexes of the worst individual(s).
        """
        sorted_island = sorted(island, key=lambda x: x.evaluation_score)
        if number_of_individuals is None or number_of_individuals == 0:
            return island.index(sorted_island[0])
        else:
            return [island.index(individual) for individual in sorted_island[:number_of_individuals]]

    @staticmethod
    def best_individual_replacement(island: list[Individual], number_of_individuals: int = None) -> int | list[int]:
        """
        Selects the best individual(s) from the island for replacement.
        
        :param island: The destination island population.
        :param number_of_individuals: Number of individuals to replace. If None, replaces one individual.
        :return: Index or list of indexes of the best individual(s).
        """
        sorted_island = sorted(island, key=lambda x: x.evaluation_score, reverse=True)
        if number_of_individuals is None or number_of_individuals == 0:
            return island.index(sorted_island[0])
        else:
            return [island.index(individual) for individual in sorted_island[:number_of_individuals]]