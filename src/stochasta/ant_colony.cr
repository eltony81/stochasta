module Stochasta
  module AntColony
    class TSPSolver
      property distance_matrix : Array(Array(Float64))
      property pheromone_matrix : Array(Array(Float64))
      property n_cities : Int32
      property n_ants : Int32
      property alpha : Float64 # Pheromone importance factor
      property beta : Float64  # Heuristic (closeness) importance factor
      property evaporation_rate : Float64 # Evaporation rate (rho)
      property q : Float64 # Pheromone strength constant
      property best_tour : Array(Int32)
      property best_tour_length : Float64

      def initialize(
        @distance_matrix : Array(Array(Float64)),
        @n_ants : Int32 = 10,
        @alpha : Float64 = 1.0,
        @beta : Float64 = 2.0,
        @evaporation_rate : Float64 = 0.5,
        @q : Float64 = 100.0
      )
        @n_cities = @distance_matrix.size
        raise ArgumentError.new("Distance matrix must be square") if @distance_matrix.any? { |row| row.size != @n_cities }
        
        # Initialize pheromone matrix to a default initial value
        @pheromone_matrix = Array(Array(Float64)).new(@n_cities) { Array(Float64).new(@n_cities, 1.0) }
        
        @best_tour = [] of Int32
        @best_tour_length = Float64::MAX
      end

      # Performs one iteration of the ant colony optimization
      def step
        # 1. Let all ants build a tour
        tours = Array(Array(Int32)).new(@n_ants)
        lengths = Array(Float64).new(@n_ants)

        @n_ants.times do
          tour = build_tour
          len = calculate_tour_length(tour)
          tours << tour
          lengths << len

          if len < @best_tour_length
            @best_tour_length = len
            @best_tour = tour.dup
          end
        end

        # 2. Evaporate pheromones
        @n_cities.times do |i|
          @n_cities.times do |j|
            @pheromone_matrix[i][j] *= (1.0 - @evaporation_rate)
            # Clip minimum pheromone level to prevent mathematical collapse
            @pheromone_matrix[i][j] = 1e-4 if @pheromone_matrix[i][j] < 1e-4
          end
        end

        # 3. Deposit new pheromones
        @n_ants.times do |k|
          tour = tours[k]
          length = lengths[k]
          pheromone_to_deposit = @q / length

          @n_cities.times do |step_idx|
            from_city = tour[step_idx]
            to_city = tour[(step_idx + 1) % @n_cities]
            @pheromone_matrix[from_city][to_city] += pheromone_to_deposit
            @pheromone_matrix[to_city][from_city] += pheromone_to_deposit # Undirected edges
          end
        end
      end

      # Solves TSP for multiple iterations
      def solve(iterations : Int32) : Tuple(Array(Int32), Float64)
        iterations.times { step }
        {@best_tour, @best_tour_length}
      end

      # Helper to build a complete tour for a single ant
      private def build_tour : Array(Int32)
        tour = Array(Int32).new(@n_cities)
        visited = Set(Int32).new

        # Place ant in a random starting city
        current_city = Random.rand(@n_cities)
        tour << current_city
        visited << current_city

        while tour.size < @n_cities
          next_city = choose_next_city(current_city, visited)
          tour << next_city
          visited << next_city
          current_city = next_city
        end

        tour
      end

      # Probabilistically choose the next city to visit using roulette wheel selection
      private def choose_next_city(current : Int32, visited : Set(Int32)) : Int32
        probabilities = Array(Float64).new(@n_cities, 0.0)
        sum = 0.0

        @n_cities.times do |city|
          unless visited.includes?(city)
            tau = @pheromone_matrix[current][city]
            dist = @distance_matrix[current][city]
            # Avoid division by zero: if distance is zero, we give it high priority
            eta = dist > 0.0 ? 1.0 / dist : 1e4
            
            p_val = (tau ** @alpha) * (eta ** @beta)
            probabilities[city] = p_val
            sum += p_val
          end
        end

        # If sum is zero (should not happen unless all features are zero), choose randomly
        return (@n_cities.times.to_a - visited.to_a).sample if sum < 1e-12

        # Roulette selection
        r = Random.rand * sum
        running_sum = 0.0
        @n_cities.times do |city|
          unless visited.includes?(city)
            running_sum += probabilities[city]
            return city if r <= running_sum
          end
        end

        # Fallback (rounding error safety)
        (@n_cities.times.to_a - visited.to_a).first
      end

      private def calculate_tour_length(tour : Array(Int32)) : Float64
        len = 0.0
        @n_cities.times do |i|
          from = tour[i]
          to = tour[(i + 1) % @n_cities]
          len += @distance_matrix[from][to]
        end
        len
      end
    end
  end
end
