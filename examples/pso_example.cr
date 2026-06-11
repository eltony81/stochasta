# Particle Swarm Optimization (PSO) Example
#
# This example demonstrates how to find the global minimum of the Rosenbrock function
# using Particle Swarm Optimization.
#
# The Rosenbrock function is a classic optimization test function:
# f(x, y) = (a - x)^2 + b * (y - x^2)^2, where a=1 and b=100.
# The global minimum is at (1, 1) where f(x, y) = 0.

require "../src/stochasta"

# Define the Rosenbrock fitness function
rosenbrock = ->(pos : Array(Float64)) {
  x = pos[0]
  y = pos[1]
  (1.0 - x)**2 + 100.0 * (y - x**2)**2
}

# Define the search space bounds for a 2D problem:
# x in [-5.0, 5.0], y in [-5.0, 5.0]
bounds = [
  {-5.0, 5.0}, # X search bounds
  {-5.0, 5.0}  # Y search bounds
]

# Instantiate the PSO Optimizer
# Parameters:
# - swarm_size: 40 particles
# - bounds: defined above
# - w: inertia weight (controls velocity conservation)
# - c1: cognitive parameter (influence of particle's personal best)
# - c2: social parameter (influence of the global best)
# - minimize: true (we are finding the minimum)
optimizer = Stochasta::PSO::Optimizer.new(
  swarm_size: 40,
  bounds: bounds,
  w: 0.5,
  c1: 1.5,
  c2: 1.5,
  minimize: true,
  &rosenbrock
)

puts "Starting PSO optimization..."

# Perform 100 iterations step by step to track progress
100.times do |i|
  optimizer.step(&rosenbrock)
  if (i + 1) % 10 == 0
    puts "Iteration #{i + 1}: Global Best Position = #{optimizer.global_best_position.map(&.round(4))} | Best Fitness = #{optimizer.global_best_fitness.round(6)}"
  end
end

puts "\nOptimization Complete!"
puts "Best Position Found: #{optimizer.global_best_position}"
puts "Best Fitness Found: #{optimizer.global_best_fitness} (Expected close to 0.0)"
