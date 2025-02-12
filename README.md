# Genetic Algorithm

This project implements a genetic algorithm to solve optimization problems, such as the knapsack problem. The `GeneticAlgorithm` class is configurable with different selection, crossover, and mutation methods, and supports both standard execution and the island model.

## Class `GeneticAlgorithm`

The `GeneticAlgorithm` class is responsible for managing the population of individuals, applying genetic operations (selection, crossover, mutation), and evolving the population through generations.

### Method `__init__`

Initializes the genetic algorithm with the provided parameters.

#### Parameters:
- **population_size** (`int`): The number of individuals in the population.
- **selection_method** (`str`, optional): The parent selection method. Default: `"roulette"`.
- **crossover_method** (`str`, optional): The crossover method. Default: `"single_point"`.
- **mutation_method** (`str`, optional): The mutation method. Default: `"bit_flip"`.
- **standard_execution** (`bool`, optional): Determines if the algorithm will run in the standard model or the island model. Default: `False`.

#### Attributes:
- **population_size** (`int`): The size of the population.
- **selection_method** (`str`): The selection method used.
- **crossover_method** (`str`): The crossover method used.
- **mutation_method** (`str`): The mutation method used.
- **standard_execution** (`bool`): Indicates if the island model will be used.
- **population** (`list[Individual]`): The list of individuals in the population.
- **generation** (`int`): The current generation number.
- **best_solution** (`Individual`): The best solution found so far.
- **solution_list** (`list[float]`): A list of the best solution scores over generations.
- **islands** (`list[list[Individual]]`): A list of populations for the island model.

#### Selection Methods:
- `"roulette"`: Roulette wheel selection.
- `"tournament"`: Tournament selection.
- `"rank"`: Rank selection.
- `"truncation"`: Truncation selection.
- `"sus"`: Stochastic universal sampling.
- `"steady_state"`: Steady-state selection.
- `"random"`: Random selection.
- `"boltzmann"`: Boltzmann selection.
- `"linear_ranking"`: Linear ranking selection.
- `"exponential_ranking"`: Exponential ranking selection.
- `"mu_lambda"`: (μ, λ) selection.
- `"metropolis_hastings"`: Metropolis-Hastings selection.
- `"rss"`: Remainder stochastic sampling.

#### Crossover Methods:
- `"single_point"`: Single-point crossover.
- `"uniform"`: Uniform crossover.
- `"two_point"`: Two-point crossover.
- `"arithmetic"`: Arithmetic crossover.
- `"half_uniform"`: Half-uniform crossover.

#### Mutation Methods:
- `"bit_flip"`: Bit-flip mutation.
- `"swap_mutation"`: Swap mutation.
- `"scramble_mutation"`: Scramble mutation.
- `"random"`: Random mutation.

### Method `solve`

Executes the genetic algorithm to solve an optimization problem, such as the knapsack problem.

#### Parameters:
- **mutation_rate** (`float`): The mutation probability for each individual.
- **num_generations** (`int`): The number of generations to run.
- **spaces** (`list[float]`): A list of the space required for each item.
- **values** (`list[float]`): A list of the values for each item.
- **space_limit** (`float`): The maximum space limit for the knapsack problem.
- **generate_graphic** (`bool`, optional): Determines if a visualization of the results will be generated. Default: `True`.
- **adaptative_mutation** (`bool`, optional): Determines if adaptive mutation rates will be used. Default: `True`.
- **elitism_chance** (`float | int`, optional): The percentage of the best individuals to be retained for the next generation. Default: `0.05` (5%).
- **num_islands** (`int`, optional): The number of islands (sub-populations) for the island model. Default: `4`.
- **migration_interval** (`int`, optional): The number of generations between migrations. Default: `5`.
- **num_migrants** (`int`, optional): The number of individuals to migrate between islands. Default: `2`.

#### Return:
- **tuple[Individual, list[int]]**: A tuple containing the Individual of the best solution and the list of scores over generations.

#### Example Usage:
```python
ga = GeneticAlgorithm(population_size=100, selection_method="tournament", crossover_method="uniform", mutation_method="bit_flip")
best_solution, generation_scores = ga.solve(mutation_rate=0.01, num_generations=50, spaces=[1, 2, 3], values=[10, 20, 30], space_limit=5)
```
# Genetic Algorithm Selection Methods

This table compares different selection methods used in Genetic Algorithms based on their exploration (diversity), exploitation (focus on the best individuals), and best use cases.

| Selection Method                  | Exploration (Diversity) | Exploitation (Focus on Best) | Best for                               |
|------------------------------------|-------------------------|-----------------------------|----------------------------------------|
| **Roulette Selection**             | ✅ Medium               | ❌ Can be random           | Balanced problems                      |
| **Rank Selection**                 | ✅ Medium               | ✅ Medium                  | Avoiding fitness scaling issues        |
| **Tournament Selection**           | ✅ High                 | ✅ High                    | Strong selective pressure              |
| **Truncation Selection**           | ❌ Low                  | ✅ Very High               | Fast convergence                        |
| **Stochastic Universal Sampling**  | ✅ High                 | ✅ Medium                  | More fairness in selection             |
| **Boltzmann Selection**            | ✅ High                 | ✅ Medium                  | Simulated Annealing-style adaptation   |
| **Steady-State Selection**         | ✅ Medium               | ✅ High                    | Gradual evolution                      |
| **Linear Ranking Selection**       | ✅ Medium               | ✅ Medium                  | Balanced selection pressure            |
| **Exponential Ranking Selection**  | ✅ Low                  | ✅ Very High               | Favoring top-ranked individuals        |
| **(μ, λ) Selection**               | ❌ Low                  | ✅ Very High               | Evolutionary strategies                |
| **Metropolis-Hastings Selection**  | ✅ High                 | ✅ Medium                  | Simulated Annealing-style adaptation   |
| **Remainder Stochastic Sampling**  | ✅ High                 | ✅ Medium                  | Hybrid of deterministic & probabilistic selection |

## Explanation
- **Exploration (Diversity):** How well the method maintains genetic diversity.
- **Exploitation (Focus on Best):** How strongly the method favors the fittest individuals.
- **Best for:** The scenarios where the method is most effective.


## Explanation of Selection Methods

### **1. Roulette Selection**
- Individuals are selected with a probability proportional to their fitness.
- Higher fitness increases the chance of being selected, but randomness can lead to premature convergence.
- **Best for:** Balanced problems where maintaining diversity is crucial.

### **2. Rank Selection**
- Individuals are ranked based on fitness, and selection probability is assigned based on rank rather than raw fitness.
- Helps avoid issues where a few dominant individuals take over early.
- **Best for:** Situations with large fitness differences to prevent premature convergence.

### **3. Tournament Selection**
- A small subset of individuals competes, and the best one is selected.
- Strength of selection pressure depends on tournament size.
- **Best for:** Problems that benefit from strong selective pressure.

### **4. Truncation Selection**
- Only the top percentage of individuals (e.g., top 50%) are selected.
- Fast convergence but can lead to loss of diversity.
- **Best for:** Situations where quick optimization is needed.

### **5. Stochastic Universal Sampling (SUS)**
- Ensures proportional selection while avoiding excessive randomness.
- Multiple selections are made at evenly spaced intervals over the fitness range.
- **Best for:** More fairness in selection while maintaining diversity.

### **6. Boltzmann Selection**
- Inspired by simulated annealing; selection probability changes dynamically over time.
- Initially promotes exploration, then shifts towards exploitation.
- **Best for:** Problems needing gradual adaptation over generations.

### **7. Steady-State Selection**
- A few individuals are replaced each generation instead of a full population reset.
- Promotes gradual evolution with less disruption.
- **Best for:** Long-term refinement and stability.

### **8. Linear Ranking Selection**
- Assigns selection probability linearly based on rank.
- Reduces dominance of high-fitness individuals while maintaining competition.
- **Best for:** Balanced selection pressure.

### **9. Exponential Ranking Selection**
- Similar to Linear Ranking but with an exponential bias toward top individuals.
- Strong exploitation of best individuals, reducing diversity.
- **Best for:** When favoring top-ranked solutions is critical.

### **10. (μ, λ) Selection**
- Evolutionary strategy where **μ** parents generate **λ** offspring.
- Only the best offspring survive to the next generation.
- **Best for:** Evolutionary strategies needing strong elitism.

### **11. Metropolis-Hastings Selection**
- Inspired by **Markov Chain Monte Carlo (MCMC)**.
- Compares a randomly chosen individual to another; if the second is worse, it may still be chosen with a probability based on fitness.
- **Best for:** Simulated annealing-style adaptation.

### **12. Remainder Stochastic Sampling (RSS)**
- Combines **deterministic selection** (direct selection of strong individuals) with **probabilistic selection** for the remaining slots.
- Ensures fair selection while reducing excessive randomness.
- **Best for:** A hybrid approach to deterministic and stochastic selection.

---

## Summary
Each selection method has trade-offs between **exploration (diversity)** and **exploitation (focus on best solutions)**. Choosing the right method depends on problem constraints, convergence speed, and the need to balance diversity.

# Chromosome Options in Genetic Algorithms

This document explains various chromosome crossover options implemented in the `Individual` class and provides a comparison table to highlight their key characteristics.

## Overview

Chromosomes represent possible solutions in genetic algorithms. The crossover methods determine how parent chromosomes combine to produce offspring. Each method has specific advantages depending on the problem and the desired balance between exploration (diversity) and exploitation (focus on optimal solutions).

## Comparison Table

| Crossover Method        | Exploration (Diversity) | Exploitation (Preserving Traits) | Best for                                |
|--------------------------|-------------------------|-----------------------------------|-----------------------------------------|
| **Single-Point**         | ✅ Medium                 | ✅ High                              | Preserving large sequences              |
| **Uniform**              | ✅ High                   | ❌    Low                  | High variability in offspring           |
| **Two-Point**            | ✅ Medium                 | ✅ Medium                            | Partial sequence preservation           |
| **Arithmetic**           | ✅ Medium                 | ✅ Medium                            | Blending solutions                      |
| **Half-Uniform (HUX)**   | ✅ Medium                 | ✅ High                              | Slight variation between similar parents|

## Mutation Option

In addition to crossover, mutations introduce random changes to chromosomes with a defined probability. This helps maintain genetic diversity and prevents premature convergence.

---


## Crossover Options

### **1. Single-Point Crossover**
- **Description:** A random split point is chosen, and chromosomes are exchanged after the split.
- **Best for:** Problems where preserving larger gene sequences is important.
- **Example:**
  - Parent 1: `[1, 0, 1 | 0, 1, 1]`
  - Parent 2: `[0, 1, 0 | 1, 0, 0]`
  - Offspring 1: `[1, 0, 1 | 1, 0, 0]`
  - Offspring 2: `[0, 1, 0 | 0, 1, 1]`

---

### **2. Uniform Crossover**
- **Description:** Each gene is selected randomly from one parent with equal probability.
- **Best for:** Problems requiring high variability in offspring.
- **Example:**
  - Parent 1: `[1, 0, 1, 1, 0, 1]`
  - Parent 2: `[0, 1, 0, 0, 1, 0]`
  - Offspring 1: `[1, 1, 1, 0, 1, 0]`
  - Offspring 2: `[0, 0, 0, 1, 0, 1]`

---

### **3. Two-Point Crossover**
- **Description:** Two random split points are chosen, and the segment between them is swapped.
- **Best for:** Problems benefiting from partial sequence preservation.
- **Example:**
  - Parent 1: `[1, 0 | 1, 0, 1 | 1, 0]`
  - Parent 2: `[0, 1 | 0, 1, 0 | 1, 1]`
  - Offspring 1: `[1, 0 | 0, 1, 0 | 1, 0]`
  - Offspring 2: `[0, 1 | 1, 0, 1 | 1, 1]`

---

### **4. Arithmetic Crossover**
- **Description:** Genes are blended using a weighted average (alpha parameter). Resulting genes are rounded to binary.
- **Best for:** Problems requiring gradual blending of traits.
- **Example:**
  - Parent 1: `[1, 0, 1, 1, 0]`
  - Parent 2: `[0, 1, 0, 0, 1]`
  - With `alpha = 0.5`: `[0.5, 0.5, 0.5, 0.5, 0.5]`
  - Rounded to binary:
    - Offspring 1: `[1, 0, 1, 1, 0]`
    - Offspring 2: `[0, 1, 0, 0, 1]`

---

### **5. Half-Uniform Crossover (HUX)**
- **Description:** Only half of the differing genes are swapped between parents.
- **Best for:** Maintaining high similarity between parents while introducing slight diversity.
- **Example:**
  - Parent 1: `[1, 0, 1, 0, 1]`
  - Parent 2: `[0, 1, 0, 1, 0]`
  - Offspring:
    - Child 1: `[1, 0, 0, 0, 1]`
    - Child 2: `[0, 1, 1, 1, 0]`

---



## Summary

The choice of crossover method depends on the problem and desired balance between exploration and exploitation. Combine these methods with mutation to enhance performance and adaptability in genetic algorithms.

# Mutation Methods in Genetic Algorithms

Mutation is a key genetic algorithm (GA) operation that helps maintain diversity (random changes) in a population and prevents premature convergence to a local optimum. Different mutation methods introduce various types of randomness into the genetic code of an individual. Below are common mutation techniques and their implementations.

## 1. Bit-Flip Mutation

### Description:
- Each gene in the chromosome has a probability of flipping (0 → 1 or 1 → 0).
- Helps maintain genetic diversity while making small, localized changes.

## 2. Swap Mutation

### Description:
- Two randomly chosen genes in the chromosome swap positions.
- Useful for problems where order matters (e.g., scheduling, traveling salesman problem).

## 3. Scramble Mutation

### Description:
- A random subsection of the chromosome is shuffled.
- Preserves the number of selected genes while introducing variation.
- Useful for permutation-based problems like ordering tasks in a schedule.

## Summary

| Mutation Type     | Effect                                        | Best Use Case                  |
|------------------|--------------------------------|--------------------------------|
| **Bit-Flip Mutation**  | Randomly flips bits (0 ↔ 1) in binary chromosomes | Binary representation problems |
| **Swap Mutation**      | Swaps two genes' positions in a chromosome | Order-based problems |
| **Scramble Mutation**  | Randomly shuffles a portion of the chromosome | Permutation-based problems |


## Island Models in Genetic Algorithms

Island models are a variation of the standard genetic algorithm where the population is divided into sub-populations (or islands). Each island evolves independently using its own genetic algorithm, and periodically, individuals from different islands migrate to exchange genetic information. This process is intended to introduce diversity into the population, potentially preventing premature convergence to suboptimal solutions and allowing the algorithm to explore different regions of the solution space simultaneously.

### Key Concepts of Island Models

- **Sub-populations (Islands):** The total population is divided into smaller groups called islands. Each island evolves separately with its own genetic operations (selection, crossover, mutation).
- **Migration:** Periodically, individuals from different islands are selected to migrate to other islands. This allows for genetic exchange and promotes diversity among the islands.
- **Isolation:** Initially, the islands are isolated from each other, meaning they do not share individuals unless a migration occurs. The isolation creates a dynamic where different genetic strategies can develop on different islands.
- **Global Optimization:** Although each island evolves independently, the migration process allows information to flow between islands, ultimately leading to a more robust search for optimal solutions across the entire population.

### Benefits of Island Models

- **Diversity Preservation:** Island models help preserve diversity in the population by preventing the entire population from converging to a single solution too quickly. Different islands may evolve in different directions, providing a broader exploration of the solution space.
- **Faster Convergence:** By exchanging genetic material between islands, the population can converge to high-quality solutions faster than in traditional single-population genetic algorithms.
- **Avoiding Local Optima:** The migration of individuals between islands can help avoid local optima by introducing new genetic material that could lead to the discovery of better solutions.

### Migration Process

Migration is the process of transferring individuals between islands. The frequency and the number of individuals migrating can be configured, and there are different strategies for selecting which individuals will migrate, but currently this project only uses the best Individual migration strategy with random replacement:

- **Interval-based Migration:** Individuals migrate after a fixed number of generations.
- **Random Migration:** Random individuals are selected for migration.
- **Best Individual Migration:** The best individual from one island is selected to migrate to another island.
