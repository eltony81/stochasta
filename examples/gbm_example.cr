# Geometric Brownian Motion (GBM) Simulation Example
#
# This example shows how to simulate future price paths for a stock
# using Geometric Brownian Motion (GBM) stochastics.

require "../src/stochasta"

# Define initial conditions for stock
s0 = 100.0     # Initial stock price ($100)
mu = 0.10      # 10% expected annual return (drift)
sigma = 0.20   # 20% annual volatility (standard deviation)
t = 1.0        # 1 year simulation horizon
steps = 252    # 252 trading days in a year

puts "--- Stock Parameters ---"
puts "Initial Price: $#{s0}"
puts "Expected annual drift (mu): #{mu * 100}%"
puts "Expected annual volatility (sigma): #{sigma * 100}%"

# --- 1. Simulate a single price path ---
puts "\n--- 1. Single Price Path Simulation ---"
path = Stochasta::Portfolio::GBM.simulate_path(s0, mu, sigma, t, steps)
puts "Path steps simulated: #{path.size - 1}"
puts "Final price at end of year: $#{path.last.round(2)}"
puts "Sample path quotes (first 5 steps):"
path[0...5].each_with_index do |price, step|
  puts "  Day #{step}: $#{price.round(2)}"
end

# --- 2. Multi-path Simulation ---
puts "\n--- 2. Multi-Path Simulation (10,000 runs) ---"
paths = Stochasta::Portfolio::GBM.simulate_paths(s0, mu, sigma, t, steps, n_paths: 10_000)
final_prices = paths.map(&.last)

# Statistics
mean_final = final_prices.sum / final_prices.size
max_final = final_prices.max
min_final = final_prices.min

puts "Simulated paths: #{paths.size}"
puts "Mean final price: $#{mean_final.round(2)}"
puts "Max final price:  $#{max_final.round(2)}"
puts "Min final price:  $#{min_final.round(2)}"
