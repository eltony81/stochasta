module Stochasta
  module DifferentialEvolution
    class Optimizer
      property population : Array(Array(Float64))
      property costs : Array(Float64)
      property bounds : Array(Tuple(Float64, Float64))
      property f : Float64  # Scale factor (mutation weight)
      property cr : Float64 # Crossover probability
      property best_position : Array(Float64)
      property best_cost : Float64

      def initialize(
        pop_size : Int32,
        @bounds : Array(Tuple(Float64, Float64)),
        @f : Float64 = 0.8,
        @cr : Float64 = 0.9,
        &cost_fn : Array(Float64) -> Float64
      )
        dimensions = @bounds.size
        @population = Array(Array(Float64)).new(pop_size)
        @costs = Array(Float64).new(pop_size)
        @best_cost = Float64::MAX
        @best_position = Array(Float64).new(dimensions, 0.0)

        # Initialize population spread out within bounds
        pop_size.times do
          agent = @bounds.map { |min, max| min + (max - min) * Random.rand }
          cost = cost_fn.call(agent)
          @population << agent
          @costs << cost

          if cost < @best_cost
            @best_cost = cost
            @best_position = agent.dup
          end
        end
      end

      # Performs one iteration of the DE optimization
      def step(&cost_fn : Array(Float64) -> Float64)
        pop_size = @population.size
        dimensions = @bounds.size

        new_population = Array(Array(Float64)).new(pop_size)
        new_costs = Array(Float64).new(pop_size)

        pop_size.times do |i|
          # Select 3 distinct agents from population (excluding target i)
          r1, r2, r3 = select_three_distinct(pop_size, i)

          # Generate mutant vector using DE/rand/1 strategy
          mutant = Array(Float64).new(dimensions)
          dimensions.times do |d|
            val = @population[r1][d] + @f * (@population[r2][d] - @population[r3][d])
            
            # Clamp to bounds
            min, max = @bounds[d]
            val = min if val < min
            val = max if val > max
            mutant << val
          end

          # Crossover: binomial recombination
          trial = Array(Float64).new(dimensions)
          rand_d = Random.rand(dimensions)
          dimensions.times do |d|
            if Random.rand < @cr || d == rand_d
              trial << mutant[d]
            else
              trial << @population[i][d]
            end
          end

          # Selection: compare trial vector cost with target vector cost
          trial_cost = cost_fn.call(trial)
          if trial_cost <= @costs[i]
            new_population << trial
            new_costs << trial_cost

            # Keep track of global best
            if trial_cost < @best_cost
              @best_cost = trial_cost
              @best_position = trial.dup
            end
          else
            new_population << @population[i].dup
            new_costs << @costs[i]
          end
        end

        @population = new_population
        @costs = new_costs
      end

      # Run optimization for multiple generations
      def optimize(generations : Int32, &cost_fn : Array(Float64) -> Float64) : Tuple(Array(Float64), Float64)
        generations.times do
          step(&cost_fn)
        end
        {@best_position, @best_cost}
      end

      private def select_three_distinct(pop_size : Int32, exclude : Int32) : Tuple(Int32, Int32, Int32)
        candidates = (0...pop_size).to_a
        candidates.delete(exclude)
        sampled = candidates.sample(3)
        {sampled[0], sampled[1], sampled[2]}
      end
    end
  end
end
