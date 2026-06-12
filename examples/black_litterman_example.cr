# Black-Litterman Portfolio Returns Estimation Example
#
# This example demonstrates how the Black-Litterman model combines prior expected
# returns (market equilibrium) with investor views and uncertainty to compute
# an updated expected returns vector.

require "../src/stochasta"

# Prior expected returns (implied equilibrium returns from market cap weights)
prior_returns = [0.08, 0.12, 0.05] # Stock A, Stock B, Stock C

# Covariance matrix (historical or implied)
covariance = [
  [0.04, 0.02, 0.01],
  [0.02, 0.09, 0.015],
  [0.01, 0.015, 0.03]
]

# Investor Views
# Let's say we have two views:
# View 1: Stock B will return 15% (Absolute view)
# View 2: Stock A will outperform Stock C by 4% (Relative view)
q_vector = [0.15, 0.04]

# View Picker Matrix P (size K views x N assets)
# Row 0: View 1 (Stock B) -> [0, 1, 0]
# Row 1: View 2 (Stock A - Stock C) -> [1, 0, -1]
p_matrix = [
  [0.0, 1.0, 0.0],
  [1.0, 0.0, -1.0]
]

# Tau scale factor (usually 0.025 to 0.05)
tau = 0.025

# Estimate returns using Black-Litterman
# (Omega covariance of views is estimated automatically from P, covariance, and tau if omitted)
adjusted_returns = Stochasta::Portfolio::BlackLitterman.estimate_returns(
  prior_returns: prior_returns,
  covariance: covariance,
  p_matrix: p_matrix,
  q_vector: q_vector,
  tau: tau
)

puts "--- Black-Litterman Returns Estimation ---"
puts "Asset A prior: #{(prior_returns[0]*100).round(2)}% | Adjusted: #{(adjusted_returns[0]*100).round(2)}%"
puts "Asset B prior: #{(prior_returns[1]*100).round(2)}% | Adjusted: #{(adjusted_returns[1]*100).round(2)}%"
puts "Asset C prior: #{(prior_returns[2]*100).round(2)}% | Adjusted: #{(adjusted_returns[2]*100).round(2)}%"
puts "\nNotice how Asset B adjusted returns moved towards our view of 15%."
puts "Also, Asset A adjusted returns are higher than Asset C, reflecting the relative view."
