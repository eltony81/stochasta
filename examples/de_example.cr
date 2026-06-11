# Differential Evolution (DE) Example
#
# This example demonstrates how to find the global minimum of the Sphere function
# using the Differential Evolution optimizer.
# The Sphere function is defined as: f(x, y, z) = x^2 + y^2 + z^2
# The global minimum is at [0, 0, 0] where f(x, y, z) = 0.

require "../src/stochasta"

# Define the Sphere cost function
sphere_fn = ->(pos : Array(Float64)) {
  pos[0]**2 + pos[1]**2 + pos[2]**2
}

# Define the search bounds for 3 dimensions:
# x, y, z in [-5.12, 5.12]
bounds = [
  {-5.12, 5.12}, # X bounds
  {-5.12, 5.12}, # Y bounds
  {-5.12, 5.12}  # Z bounds
]

# Instantiate the Differential Evolution Optimizer
# Parameters:
# - pop_size: 20 agents
# - bounds: defined above
# - f: scaling factor (0.8)
# - cr: crossover rate (0.9)
optimizer = Stochasta::DifferentialEvolution::Optimizer.new(
  pop_size: 20,
  bounds: bounds,
  f: 0.8,
  cr: 0.9,
  &sphere_fn
)

puts "Starting Differential Evolution optimization..."

# Optimize for 100 generations
50.times do |gen|
  optimizer.step(&sphere_fn)
  if (gen + 1) % 10 == 0
    puts "Generation #{gen + 1}: Best Cost = #{optimizer.best_cost.round(6)} | Best Position = #{optimizer.best_position.map(&.round(4))}"
  end
end

puts "\nOptimization Complete!"
puts "Best Position: #{optimizer.best_position}"
puts "Best Cost: #{optimizer.best_cost} (Expected close to 0.0)"
