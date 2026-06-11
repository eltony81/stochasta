# Monte Carlo Simulation & Integration Example
#
# This example demonstrates:
# 1. Estimating the value of Pi using random sampling inside a unit quadrant.
# 2. Calculating the integral of f(x) = x^2 from x=0 to x=1.
# 3. Double integration of f(x, y) = x * y over [0, 1] x [0, 1].

require "../src/stochasta"

# --- 1. Pi Estimation via Simulation ---
puts "--- 1. Pi Estimation ---"

# Set total trials for the simulation
total_samples = 1_000_000

# Run a Monte Carlo simulation. We sample random points (x, y) in a unit square.
# The proportion of points inside the unit circle quadrant (x^2 + y^2 <= 1) times 4 estimates Pi.
results = Stochasta::MonteCarlo.simulate(total_samples) do
  x = Random.rand
  y = Random.rand
  (x*x + y*y) <= 1.0 ? 1 : 0
end

# Calculate Pi estimate
points_inside_circle = results.sum
pi_estimate = (points_inside_circle.to_f / total_samples) * 4.0
puts "Estimated Pi with #{total_samples} samples: #{pi_estimate}"


# --- 2. 1D Numerical Integration ---
puts "\n--- 2. 1D Numerical Integration ---"

# Integrate f(x) = x^2 from a=0.0 to b=2.0. (Analytical result is 8/3 = 2.6667)
result_1d = Stochasta::MonteCarlo.integrate(0.0, 2.0, samples: 100_000) do |x|
  x * x
end

puts "Integral of x^2 from 0 to 2: #{result_1d} (Expected ~2.6667)"


# --- 3. Multi-dimensional Numerical Integration ---
puts "\n--- 3. Multi-dimensional Integration ---"

# Integrate f(x, y) = x * y over x in [0, 2] and y in [0, 3].
# Bounding box is defined by [{x_min, x_max}, {y_min, y_max}]
# Analytical result: (2^2 / 2) * (3^2 / 2) = 2 * 4.5 = 9.0
bounds = [
  {0.0, 2.0}, # X boundaries
  {0.0, 3.0}  # Y boundaries
]

result_multi = Stochasta::MonteCarlo.integrate_multi(bounds, samples: 100_000) do |point|
  x = point[0]
  y = point[1]
  x * y
end

puts "Integral of x * y over [0,2]x[0,3]: #{result_multi} (Expected ~9.0)"
