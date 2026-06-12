# Portfolio Risk Metrics Example
#
# This example shows how to calculate Value at Risk (VaR) and Conditional Value
# at Risk (CVaR) using both Parametric (Normal Distribution) and Monte Carlo methods.

require "../src/stochasta"

# Define expected annual portfolio statistics
mean_return = 0.08  # 8% expected return
std_dev = 0.15     # 15% standard deviation (volatility)

# --- 1. Parametric VaR & CVaR ---
puts "--- 1. Parametric Risk Metrics (95% Confidence) ---"
var_95 = Stochasta::Portfolio::Risk.parametric_var(mean_return, std_dev, 0.95)
cvar_95 = Stochasta::Portfolio::Risk.parametric_cvar(mean_return, std_dev, 0.95)

puts "Parametric VaR  (95%): #{(var_95 * 100).round(2)}% loss"
puts "Parametric CVaR (95%): #{(cvar_95 * 100).round(2)}% loss"


# --- 2. Monte Carlo VaR & CVaR ---
puts "\n--- 2. Monte Carlo Risk Metrics (95% Confidence) ---"

# Simulate 50,000 portfolio returns using normal distribution
simulated_returns = Array(Float64).new(50_000)
50_000.times do
  # Generate returns using Box-Muller transform scaled to our mean/std_dev
  z = Stochasta::Portfolio::GBM.random_normal
  simulated_returns << (mean_return + z * std_dev)
end

mc_var = Stochasta::Portfolio::Risk.monte_carlo_var(simulated_returns, 0.95)
mc_cvar = Stochasta::Portfolio::Risk.monte_carlo_cvar(simulated_returns, 0.95)

puts "Monte Carlo VaR  (95%): #{(mc_var * 100).round(2)}% loss"
puts "Monte Carlo CVaR (95%): #{(mc_cvar * 100).round(2)}% loss"
