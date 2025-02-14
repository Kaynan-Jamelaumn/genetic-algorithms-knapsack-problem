from Individual import Individual
from ReplacementMethods import *
from typing import Callable

class MigrationMethods:
    @staticmethod
    def ring_migration(islands: list[list[Individual]], num_migrants: int, replacement_methods: dict[str, Callable[[list["Individual"], int], int | list[int]]], primary_replacement_method: str = "best", secondary_replacement_method: str = "random") -> None:
        """
        Performs migration of individuals between islands to introduce genetic diversity.

        Steps:
        1. Selects the top num_migrants individuals from each island.
        2. Migrates individuals to the next island in a circular pattern.
        3. Replaces random individuals in the destination island with migrants.
        4. Evaluates the updated population.

        :param num_migrants: The number of individuals to migrate per island.
        """
        migrants = []
        for island in islands:
            # Use the best_individual_replacement method to select the top migrants
            migrant_indexes = MigrationMethods.apply_replacement_method(replacement_methods, primary_replacement_method, island, num_migrants)
            # Retrieve the actual migrants using the indexes
            migrants.append([island[idx] for idx in migrant_indexes])

        for i in range(len(islands)):
            dest_idx = (i + 1) % len(islands)  # Determine the target island
            for migrant in migrants[i]:
                idx = MigrationMethods.apply_replacement_method(replacement_methods, secondary_replacement_method, islands[dest_idx], 1)  # Replace one individual at a time
                if isinstance(idx, list):
                    idx = idx[0]  # If a list is returned, take the first index
                islands[dest_idx][idx] = migrant  # Replace with migrant
                islands[dest_idx][idx].evaluate()  # Re-evaluate the replaced individual

    @staticmethod
    def random_migration(islands: list[list[Individual]], num_migrants: int, replacement_methods: dict[str, Callable[[list["Individual"], int], int | list[int]]], primary_replacement_method: str = "random", secondary_replacement_method: str = "random") -> None:
        """
        Performs random migration of individuals between islands to introduce genetic diversity.

        Steps:
        1. Selects random individuals from each island.
        2. Migrates individuals to a randomly chosen island.
        3. Replaces random individuals in the destination island with migrants.
        4. Evaluates the updated population.

        :param num_migrants: The number of individuals to migrate per island.
        """
        migrants = []
        for island in islands:
            # Use the random_individual_replacement method to select random migrants
            migrant_indexes = MigrationMethods.apply_replacement_method(replacement_methods, primary_replacement_method, island, num_migrants)
            # Retrieve the actual migrants using the indexes
            migrants.append([island[idx] for idx in migrant_indexes])

        for i in range(len(islands)):
            for migrant in migrants[i]:
                dest_idx = random.randint(0, len(islands) - 1)  # Randomly choose a destination island
                idx = MigrationMethods.apply_replacement_method(replacement_methods, secondary_replacement_method, islands[dest_idx], 1)  # Replace one individual at a time
                if isinstance(idx, list):
                    idx = idx[0]  # If a list is returned, take the first index
                islands[dest_idx][idx] = migrant  # Replace with migrant
                islands[dest_idx][idx].evaluate()  # Re-evaluate the replaced individual


    @staticmethod
    def apply_replacement_method(
        replacement_methods: dict[str, Callable[[list["Individual"], int], int | list[int]]],
        replacement_target: str, island: list[Individual],
        num_migrants: int = None) -> int | list[int]:
        """
        Apply the specified replacement method to the individual.
        """
        if replacement_target not in replacement_methods:
            raise ValueError(f"Invalid replacement method: {replacement_target}")
        replacement_function = replacement_methods[replacement_target]
        return replacement_function(island, num_migrants)