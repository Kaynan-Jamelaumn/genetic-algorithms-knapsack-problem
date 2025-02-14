from Individual import Individual
class CrossoverMethods:
    @staticmethod
    def single_point_crossover(num_migrants: int, islands: list[list[Individual]]) -> None:
        pass

    @staticmethod
    def uniform_crossover(parent1: Individual , parent2: Individual) -> tuple[Individual, Individual]:
        return parent1.uniform_crossover(parent2)

  