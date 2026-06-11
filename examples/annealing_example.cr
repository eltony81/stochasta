# Simulated Annealing Example
#
# This example demonstrates how to find the minimum of a 1D function
# f(x) = x^4 - 5*x^2 + 3*x (which has local and global minima)
# using Simulated Annealing.

require "../src/stochasta"

# Define the cost/energy function we want to minimize
cost_fn = ->(x : Float64) {
  x**4 - 5.0 * (x**2) + 3.0 * x
}

# Define the neighbor generator function
# It takes the current state (x) and slightly perturbs it within a step size.
neighbor_fn = ->(x : Float64) {
  step = (Random.rand * 2.0 - 1.0) * 0.5
  x + step
}

# Initialize Simulated Annealing parameters
initial_state = 10.0      # Start searching from x = 10.0
initial_temp = 100.0      # High starting temperature
cooling_rate = 0.98       # Temperature decreases by 2% at each iteration
min_temp = 0.0001         # Minimum temperature threshold to stop

puts "Starting Simulated Annealing..."

# Run the optimization
best_state, best_cost = Stochasta::SimulatedAnnealing.optimize(
  initial_state: initial_state,
  initial_temp: initial_temp,
  cooling_rate: cooling_rate,
  cost_fn: cost_fn,
  min_temp: min_temp
) do |current|
  # This block generates a neighbor state
  neighbor_fn.call(current)
end

puts "\nOptimization Complete!"
puts "Best X (state) found: #{best_state.round(6)}"
puts "Minimum Cost (energy) found: #{best_cost.round(6)}"
