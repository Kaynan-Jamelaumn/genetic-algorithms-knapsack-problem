from Individual import Individual
class CrossoverMethods:
    @staticmethod
    def single_point_crossover(parent1: Individual , parent2: Individual) -> tuple[Individual, Individual]:
        return parent1.crossover(parent2)

    @staticmethod
    def uniform_crossover(parent1: Individual , parent2: Individual) -> tuple[Individual, Individual]:
        return parent1.uniform_crossover(parent2)

    @staticmethod
    def two_point_crossover(parent1: Individual , parent2: Individual) -> tuple[Individual, Individual]:
        return parent1.two_point_crossover(parent2)

    @staticmethod
    def arithmetic_crossover(parent1: Individual , parent2: Individual) -> tuple[Individual, Individual]:
        return parent1.arithmetic_crossover(parent2)

    @staticmethod
    def half_uniform_crossover(parent1: Individual , parent2: Individual) -> tuple[Individual, Individual]:
        return parent1.half_uniform_crossover(parent2)