module Stochasta
  module ArtificialBeeColony
    class Optimizer
      property food_sources : Array(Array(Float64))
      property costs : Array(Float64)
      property fitnesses : Array(Float64)
      property trials : Array(Int32)
      property bounds : Array(Tuple(Float64, Float64))
      property limit : Int32 # Limit for food source abandonment
      property best_position : Array(Float64)
      property best_cost : Float64
      property n_sources : Int32

      def initialize(
        swarm_size : Int32, # Total bees (employed + onlookers = swarm_size)
        @bounds : Array(Tuple(Float64, Float64)),
        @limit : Int32 = 100,
        &cost_fn : Array(Float64) -> Float64
      )
        dimensions = @bounds.size
        # S is the number of food sources, which is swarm_size / 2
        @n_sources = swarm_size // 2
        raise ArgumentError.new("Swarm size must be at least 4") if @n_sources < 2

        @food_sources = Array(Array(Float64)).new(@n_sources)
        @costs = Array(Float64).new(@n_sources)
        @fitnesses = Array(Float64).new(@n_sources)
        @trials = Array(Int32).new(@n_sources, 0)

        @best_cost = Float64::MAX
        @best_position = Array(Float64).new(dimensions, 0.0)

        # Initialize food sources randomly within bounds
        @n_sources.times do
          source = @bounds.map { |min, max| min + (max - min) * Random.rand }
          cost = cost_fn.call(source)
          
          @food_sources << source
          @costs << cost
          @fitnesses << calculate_fitness(cost)

          if cost < @best_cost
            @best_cost = cost
            @best_position = source.dup
          end
        end
      end

      # Fits the costs to fitness values
      private def calculate_fitness(cost : Float64) : Float64
        if cost >= 0.0
          1.0 / (1.0 + cost)
        else
          1.0 + cost.abs
        end
      end

      # Performs one iteration of the ABC optimization
      def step(&cost_fn : Array(Float64) -> Float64)
        dimensions = @bounds.size

        # 1. Employed Bees Phase
        @n_sources.times do |i|
          # Choose a random parameter index and a random partner food source (k != i)
          param_idx = Random.rand(dimensions)
          partner_idx = choose_partner(i)

          phi = Random.rand * 2.0 - 1.0
          candidate = @food_sources[i].dup
          candidate[param_idx] = @food_sources[i][param_idx] + phi * (@food_sources[i][param_idx] - @food_sources[partner_idx][param_idx])

          # Clamp candidate to bounds
          min, max = @bounds[param_idx]
          candidate[param_idx] = min if candidate[param_idx] < min
          candidate[param_idx] = max if candidate[param_idx] > max

          # Evaluate candidate
          cand_cost = cost_fn.call(candidate)
          if cand_cost < @costs[i]
            @food_sources[i] = candidate
            @costs[i] = cand_cost
            @fitnesses[i] = calculate_fitness(cand_cost)
            @trials[i] = 0

            if cand_cost < @best_cost
              @best_cost = cand_cost
              @best_position = candidate.dup
            end
          else
            @trials[i] += 1
          end
        end

        # 2. Onlooker Bees Phase
        # Calculate selection probabilities
        sum_fit = @fitnesses.sum
        probs = @fitnesses.map { |f| sum_fit > 0.0 ? f / sum_fit : 1.0 / @n_sources }

        # For each onlooker bee (same count as food sources)
        @n_sources.times do
          # Select a food source based on probabilities
          i = select_by_probability(probs)
          
          param_idx = Random.rand(dimensions)
          partner_idx = choose_partner(i)

          phi = Random.rand * 2.0 - 1.0
          candidate = @food_sources[i].dup
          candidate[param_idx] = @food_sources[i][param_idx] + phi * (@food_sources[i][param_idx] - @food_sources[partner_idx][param_idx])

          min, max = @bounds[param_idx]
          candidate[param_idx] = min if candidate[param_idx] < min
          candidate[param_idx] = max if candidate[param_idx] > max

          cand_cost = cost_fn.call(candidate)
          if cand_cost < @costs[i]
            @food_sources[i] = candidate
            @costs[i] = cand_cost
            @fitnesses[i] = calculate_fitness(cand_cost)
            @trials[i] = 0

            if cand_cost < @best_cost
              @best_cost = cand_cost
              @best_position = candidate.dup
            end
          else
            @trials[i] += 1
          end
        end

        # 3. Scout Bees Phase (Abandon food sources with trials > limit)
        @n_sources.times do |i|
          if @trials[i] > @limit
            new_source = @bounds.map { |min, max| min + (max - min) * Random.rand }
            new_cost = cost_fn.call(new_source)

            @food_sources[i] = new_source
            @costs[i] = new_cost
            @fitnesses[i] = calculate_fitness(new_cost)
            @trials[i] = 0

            if new_cost < @best_cost
              @best_cost = new_cost
              @best_position = new_source.dup
            end
          end
        end
      end

      # Optimizes for `iterations` steps
      def optimize(iterations : Int32, &cost_fn : Array(Float64) -> Float64) : Tuple(Array(Float64), Float64)
        iterations.times do
          step(&cost_fn)
        end
        {@best_position, @best_cost}
      end

      private def choose_partner(exclude : Int32) : Int32
        partner = Random.rand(@n_sources)
        while partner == exclude
          partner = Random.rand(@n_sources)
        end
        partner
      end

      private def select_by_probability(probs : Array(Float64)) : Int32
        r = Random.rand
        running_sum = 0.0
        @n_sources.times do |idx|
          running_sum += probs[idx]
          return idx if r <= running_sum
        end
        @n_sources - 1
      end
    end
  end
end
