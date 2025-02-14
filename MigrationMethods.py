from Individual import Individual
from ReplacementMethods import *
from typing import Callable

class MigrationMethods:
    @staticmethod
    def ring_migration(islands: list[list[Individual]], num_migrants: int, replacement_methods: dict[str, Callable[[list["Individual"], int], int | list[int]]], primary_replacement_method: str = "best", secondary_replacement_method: str = "random") -> None:
        """
        Performs migration of individuals between islands in a circular pattern to introduce genetic diversity.

        This method selects a specified number of top individuals from each island and migrates them to the next island in a ring structure.
        The migrated individuals replace existing individuals in the destination island based on a chosen replacement method.

        Steps:
        1. Selects top num_migrants individuals from each island using the primary replacement method.
        2. Migrates individuals to the next island in a circular order.
        3. Replaces individuals in the destination island using the secondary replacement method.
        4. Evaluates the updated individuals after replacement.

        :param islands: A list of islands, where each island is a list of individuals.
        :param num_migrants: The number of individuals to migrate per island.
        :param replacement_methods: A dictionary mapping method names to callable functions for selecting replacement indices.
        :param primary_replacement_method: The method used to select migrants from the source island.
        :param secondary_replacement_method: The method used to select individuals for replacement in the destination island.
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

        This method randomly selects a specified number of individuals from each island and migrates them to another
        randomly chosen island. The migrated individuals replace existing individuals in the destination island,
        following a specified replacement method.

        Steps:
        1. Selects random individuals from each island using the primary replacement method.
        2. Migrates individuals to randomly chosen destination islands.
        3. Replaces individuals in the destination islands using the secondary replacement method.
        4. Evaluates the updated individuals after replacement.

        :param islands: A list of islands, where each island is a list of individuals.
        :param num_migrants: The number of individuals to migrate per island.
        :param replacement_methods: A dictionary mapping method names to callable functions for selecting replacement indices.
        :param primary_replacement_method: The method used to select migrants from the source island.
        :param secondary_replacement_method: The method used to select individuals for replacement in the destination island.
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
    def adaptive_migration(islands: list[list[Individual]], num_migrants: int, 
                            replacement_methods: dict[str, Callable[[list["Individual"], int], int | list[int]]], 
                            primary_replacement_method: str = "best", secondary_replacement_method: str = "random") -> None:
        """
        Performs adaptive migration based on genetic diversity to balance diversity across islands.

        This method calculates the genetic diversity of each island and adjusts the number of migrants accordingly.
        Islands with lower diversity send more individuals, while those with higher diversity send fewer.
        Migrants are sent to the next island in a circular fashion.

        Steps:
        1. Calculate genetic diversity for each island.
        2. Determine the number of migrants adaptively based on diversity.
        3. Select migrants using the primary replacement method.
        4. Send migrants to the next island in a circular order.
        5. Replace individuals in the destination island using the secondary replacement method.
        6. Evaluate the updated individuals after replacement.

        :param islands: A list of islands, where each island is a list of individuals.
        :param num_migrants: The base number of individuals to migrate per island.
        :param replacement_methods: A dictionary mapping method names to callable functions for selecting replacement indices.
        :param primary_replacement_method: The method used to select migrants from the source island.
        :param secondary_replacement_method: The method used to select individuals for replacement in the destination island.
        """
        diversities = [MigrationMethods.calculate_diversity(island) for island in islands]
        avg_diversity = sum(diversities) / len(islands)
        
        migrants = []
        for i, island in enumerate(islands):
            num_migrants_adaptive = max(1, int(num_migrants * (1 - diversities[i] / avg_diversity)))
            migrant_indexes = MigrationMethods.apply_replacement_method(replacement_methods, primary_replacement_method, island, num_migrants_adaptive)
            migrants.append([island[idx] for idx in migrant_indexes])
        
        for i in range(len(islands)):
            dest_idx = (i + 1) % len(islands)
            for migrant in migrants[i]:
                idx = MigrationMethods.apply_replacement_method(replacement_methods, secondary_replacement_method, islands[dest_idx], 1)
                if isinstance(idx, list):
                    idx = idx[0]
                islands[dest_idx][idx] = migrant
                islands[dest_idx][idx].evaluate()

    @staticmethod
    def calculate_diversity(island: list[Individual]) -> float:
        """
        Calculates the genetic diversity of an island.

        The diversity is computed as the average number of differing genes (Hamming distance)
        between all pairs of individuals in the island.

        :param island: A list of individuals representing an island.
        :return: The calculated genetic diversity value.
        """
        if len(island) < 2:
            return 0
        diversity = sum(sum(c1 != c2 for c1, c2 in zip(ind1.chromosome, ind2.chromosome)) 
                        for i, ind1 in enumerate(island) for ind2 in island[i+1:])
        return diversity / (len(island) * (len(island) - 1) / 2)



    @staticmethod
    def apply_replacement_method(
        replacement_methods: dict[str, Callable[[list["Individual"], int], int | list[int]]],
        replacement_target: str, island: list[Individual],
        num_migrants: int = None) -> int | list[int]:
        """
        Applies a specified replacement method to determine the indices of individuals to be replaced or migrated.

        This method retrieves the appropriate replacement function from the provided dictionary and applies it to the given island population.

        :param replacement_methods: A dictionary mapping method names to callable functions for selecting replacement indices.
        :param replacement_target: The name of the replacement method to apply.
        :param island: A list of individuals representing an island.
        :param num_migrants: The number of individuals to replace or migrate (if applicable).
        :return: A single index or a list of indices representing the individuals selected by the replacement method.
        :raises ValueError: If the specified replacement method is not found in the dictionary.
        """
        if replacement_target not in replacement_methods:
            raise ValueError(f"Invalid replacement method: {replacement_target}")
        replacement_function = replacement_methods[replacement_target]
        return replacement_function(island, num_migrants)