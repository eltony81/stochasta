# Genetic Algorithm Example
#
# This example demonstrates how to use the Genetic Algorithm (GA) Engine
# to solve a simple "OneMax" problem: maximizing the number of 1s in a bitstring.

require "../src/stochasta"

# Define the fitness function: the score of a genome is the sum of its bits.
# Higher fitness means a better individual.
fitness_fn = ->(genome : Array(Int32)) {
  genome.sum.to_f
}

# Define the crossover function: 1-point crossover.
# It splits both parent genomes at a random point and combines them.
crossover_fn = ->(parent1 : Array(Int32), parent2 : Array(Int32)) {
  split_point = Random.rand(parent1.size)
  parent1[0...split_point] + parent2[split_point...]
}

# Define the mutation function: randomly flips a single bit in the genome.
mutate_fn = ->(genome : Array(Int32)) {
  mutated = genome.dup
  random_index = Random.rand(genome.size)
  # Flip 0 to 1, or 1 to 0
  mutated[random_index] = mutated[random_index] == 0 ? 1 : 0
  mutated
}

# Initialize a starting population of 20 individuals, each with 10 random bits.
initial_population = Array(Array(Int32)).new(20) do
  Array(Int32).new(10) { Random.rand(2) }
end

# Instantiate the Genetic Algorithm Engine.
# We set population size to 20, mutation rate to 10%, and crossover rate to 80%.
engine = Stochasta::Genetic::Engine(Array(Int32)).new(
  population_size: 20,
  mutation_rate: 0.1,
  crossover_rate: 0.8,
  fitness_fn: fitness_fn,
  crossover_fn: crossover_fn,
  mutate_fn: mutate_fn,
  initial_genomes: initial_population
)

puts "Starting evolution..."

# Evolve the population for 50 generations.
# We pass a block to print the best individual found at each generation step.
best_individual = engine.evolve(50) do |generation, best|
  puts "Gen #{generation}: Best Fitness = #{best.fitness} | Genome = #{best.genome}"
end

puts "\nFinal Best Genome: #{best_individual.genome}"
puts "Final Best Fitness: #{best_individual.fitness}"
