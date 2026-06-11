module Stochasta
  module Genetic
    class Individual(T)
      property genome : T
      property fitness : Float64

      def initialize(@genome : T, @fitness : Float64 = 0.0)
      end
    end

    class Engine(T)
      property population : Array(Individual(T))
      property fitness_fn : T -> Float64
      property crossover_fn : (T, T) -> T
      property mutate_fn : T -> T
      property population_size : Int32
      property mutation_rate : Float64
      property crossover_rate : Float64
      property tournament_size : Int32
      property elitism : Bool

      def initialize(
        @population_size : Int32,
        @mutation_rate : Float64,
        @crossover_rate : Float64,
        @fitness_fn : T -> Float64,
        @crossover_fn : (T, T) -> T,
        @mutate_fn : T -> T,
        initial_genomes : Array(T),
        @tournament_size : Int32 = 3,
        @elitism : Bool = true
      )
        raise ArgumentError.new("Initial population size must match population_size") if initial_genomes.size != @population_size
        @population = initial_genomes.map { |g| Individual(T).new(g, @fitness_fn.call(g)) }
      end

      # Selects an individual using tournament selection
      def select_tournament : Individual(T)
        best = nil
        @tournament_size.times do
          candidate = @population.sample
          if best.nil? || candidate.fitness > best.fitness
            best = candidate
          end
        end
        best.not_nil!
      end

      # Evolves the population by one generation
      def evolve_generation
        new_pop_genomes = Array(T).new(@population_size)

        # Keep the best individual (elitism)
        if @elitism
          best_ind = best_individual
          new_pop_genomes << best_ind.genome
        end

        while new_pop_genomes.size < @population_size
          parent1 = select_tournament
          parent2 = select_tournament

          child_genome = if Random.rand < @crossover_rate
                           @crossover_fn.call(parent1.genome, parent2.genome)
                         else
                           Random.rand < 0.5 ? parent1.genome : parent2.genome
                         end

          if Random.rand < @mutation_rate
            child_genome = @mutate_fn.call(child_genome)
          end

          new_pop_genomes << child_genome
        end

        # Trim to exact size if elitism + pairs exceeded it (should not happen with single child logic, but good practice)
        new_pop_genomes = new_pop_genomes[0...@population_size]

        @population = new_pop_genomes.map do |g|
          Individual(T).new(g, @fitness_fn.call(g))
        end
      end

      def best_individual : Individual(T)
        @population.max_by { |ind| ind.fitness }
      end

      # Evolves for the specified number of generations
      def evolve(generations : Int32) : Individual(T)
        generations.times do
          evolve_generation
        end
        best_individual
      end

      # Evolves for the specified number of generations and yields progress
      def evolve(generations : Int32, &block : Int32, Individual(T) ->) : Individual(T)
        generations.times do |gen|
          evolve_generation
          block.call(gen + 1, best_individual)
        end
        best_individual
      end
    end
  end
end
