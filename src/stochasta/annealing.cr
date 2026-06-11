module Stochasta
  module SimulatedAnnealing
    # Runs the Simulated Annealing optimization process
    # T is the state type
    def self.optimize(
      initial_state : T,
      initial_temp : Float64,
      cooling_rate : Float64,
      cost_fn : T -> Float64,
      min_temp : Float64 = 1e-4,
      &neighbor_fn : T -> T
    ) : Tuple(T, Float64) forall T
      current_state = initial_state
      current_cost = cost_fn.call(current_state)
      
      best_state = current_state
      best_cost = current_cost

      temp = initial_temp

      while temp > min_temp
        # Yield the current state to the block to get a neighbor
        # We need a neighbor generator function. We can yield to the block or pass a lambda.
        # Let's yield to get the neighbor: yield(current_state)
        neighbor = neighbor_fn.call(current_state)
        neighbor_cost = cost_fn.call(neighbor)

        cost_diff = neighbor_cost - current_cost

        # If neighbor is better, or accepted by probability
        if cost_diff < 0.0 || Random.rand < Math.exp(-cost_diff / temp)
          current_state = neighbor
          current_cost = neighbor_cost

          if current_cost < best_cost
            best_state = current_state
            best_cost = current_cost
          end
        end

        temp *= cooling_rate
      end

      {best_state, best_cost}
    end
  end
end
