# Portfolio Optimization Example
#
# This example shows how to perform Modern Portfolio Theory (MPT) optimization
# (Minimum Variance and Maximum Sharpe Ratio) using Stochasta's built-in
# Particle Swarm Optimization (PSO) to find optimal asset allocation weights.

require "../src/stochasta"

# Define 3 assets (e.g. Stock A, Stock B, Stock C)
# Expected annual returns for each asset
expected_returns = [0.12, 0.18, 0.10] # 12%, 18%, 10%

# Covariance matrix between the assets
covariance_matrix = [
  [0.04, 0.02, 0.01],  # Asset A (variance = 4%)
  [0.02, 0.09, 0.015], # Asset B (variance = 9%)
  [0.01, 0.015, 0.03]  # Asset C (variance = 3%)
]

# Instantiate the Portfolio Optimizer
optimizer = Stochasta::Portfolio::Optimizer.new(expected_returns, covariance_matrix)

puts "--- Assets ---"
puts "Expected Returns: #{expected_returns.map { |r| "#{(r*100).round(1)}%" }}"

# --- 1. Minimum Variance Portfolio ---
puts "\n--- 1. Optimizing for Minimum Variance ---"
min_var_w = optimizer.min_variance_weights(iterations: 150)
ret_min = optimizer.portfolio_return(min_var_w)
var_min = optimizer.portfolio_variance(min_var_w)

puts "Optimal Weights:  #{min_var_w.map { |w| "#{(w*100).round(2)}%" }}"
puts "Expected Return:  #{(ret_min*100).round(2)}%"
puts "Expected Volatility: #{(Math.sqrt(var_min)*100).round(2)}%"


# --- 2. Maximum Sharpe Ratio Portfolio ---
puts "\n--- 2. Optimizing for Maximum Sharpe Ratio ---"
risk_free_rate = 0.03 # 3% Risk-free rate
max_sharpe_w = optimizer.max_sharpe_weights(risk_free_rate: risk_free_rate, iterations: 150)
ret_sharpe = optimizer.portfolio_return(max_sharpe_w)
var_sharpe = optimizer.portfolio_variance(max_sharpe_w)
vol_sharpe = Math.sqrt(var_sharpe)
sharpe_ratio = (ret_sharpe - risk_free_rate) / vol_sharpe

puts "Optimal Weights:  #{max_sharpe_w.map { |w| "#{(w*100).round(2)}%" }}"
puts "Expected Return:  #{(ret_sharpe*100).round(2)}%"
puts "Expected Volatility: #{(vol_sharpe*100).round(2)}%"
puts "Sharpe Ratio:     #{sharpe_ratio.round(4)}"
