# Ant Colony Optimization (ACO) Example
#
# This example shows how to solve the Traveling Salesperson Problem (TSP)
# for a set of 4 cities using Ant Colony Optimization.
#
# The distances between the cities are defined in a symmetric matrix.

require "../src/stochasta"

# Define city names for output
city_names = ["Rome", "Milan", "Venice", "Florence"]

# Define a 4x4 distance matrix
# distance_matrix[i][j] represents the distance from city i to city j
distance_matrix = [
  [0.0, 570.0, 530.0, 280.0],  # Rome
  [570.0, 0.0, 270.0, 300.0],  # Milan
  [530.0, 270.0, 0.0, 260.0],  # Venice
  [280.0, 300.0, 260.0, 0.0]   # Florence
]

# Instantiate the ACO TSP Solver
# Parameters:
# - distance_matrix: defined above
# - n_ants: 8 ants
# - alpha: 1.0 (importance of pheromones)
# - beta: 2.0 (importance of distance heuristic)
# - evaporation_rate: 0.2 (evaporation rate rho)
# - q: 100.0 (pheromone strength constant)
solver = Stochasta::AntColony::TSPSolver.new(
  distance_matrix: distance_matrix,
  n_ants: 8,
  alpha: 1.0,
  beta: 2.0,
  evaporation_rate: 0.2,
  q: 100.0
)

puts "Starting Ant Colony optimization..."

# Run the solver for 30 iterations
30.times do |iter|
  solver.step
  if (iter + 1) % 5 == 0
    readable_tour = solver.best_tour.map { |idx| city_names[idx] }
    puts "Iteration #{iter + 1}: Best Tour Length = #{solver.best_tour_length} km | Tour = #{readable_tour.join(" -> ")} -> #{readable_tour.first}"
  end
end

puts "\nOptimization Complete!"
readable_best_tour = solver.best_tour.map { |idx| city_names[idx] }
puts "Best Tour Length: #{solver.best_tour_length} km"
puts "Best Tour Path: #{readable_best_tour.join(" -> ")} -> #{readable_best_tour.first}"
