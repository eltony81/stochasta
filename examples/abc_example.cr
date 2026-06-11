# Artificial Bee Colony (ABC) Example
#
# This example demonstrates how to find the global minimum of the Sphere function
# using the Artificial Bee Colony optimizer.
# The Sphere function is defined as: f(x, y) = x^2 + y^2
# The global minimum is at [0, 0] where f(x, y) = 0.

require "../src/stochasta"

# Define the Sphere cost function
sphere_fn = ->(pos : Array(Float64)) {
  pos[0]**2 + pos[1]**2
}

# Define the bounds for 2 dimensions:
# x, y in [-5.0, 5.0]
bounds = [
  {-5.0, 5.0}, # X bounds
  {-5.0, 5.0}  # Y bounds
]

# Instantiate the Artificial Bee Colony Optimizer
# Parameters:
# - swarm_size: 20 bees (10 employed bees + 10 onlooker bees)
# - bounds: defined above
# - limit: 20 (trial limit for food source abandonment by scout bees)
optimizer = Stochasta::ArtificialBeeColony::Optimizer.new(
  swarm_size: 20,
  bounds: bounds,
  limit: 20,
  &sphere_fn
)

puts "Starting Artificial Bee Colony optimization..."

# Optimize for 100 iterations
50.times do |iter|
  optimizer.step(&sphere_fn)
  if (iter + 1) % 10 == 0
    puts "Iteration #{iter + 1}: Best Cost = #{optimizer.best_cost.round(6)} | Best Position = #{optimizer.best_position.map(&.round(4))}"
  end
end

puts "\nOptimization Complete!"
puts "Best Position Found: #{optimizer.best_position}"
puts "Best Cost Found: #{optimizer.best_cost} (Expected close to 0.0)"
