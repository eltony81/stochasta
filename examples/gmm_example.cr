# Gaussian Mixture Model (GMM) Example
#
# This example shows how to use GMM with Expectation-Maximization to perform
# "soft clustering", where each data point has a probability of belonging
# to each cluster.

require "../src/stochasta"

# Define 2D dataset with two distinct clusters
data = [
  # Cluster 0
  [1.0, 1.2],
  [1.1, 0.9],
  [0.9, 1.1],

  # Cluster 1
  [10.0, 9.8],
  [9.9, 10.1],
  [10.1, 10.0]
]

# Instantiate GMM model with K = 2 clusters
model = Stochasta::GMM::Model.new(k: 2).fit(data)

puts "GMM Training Complete."
puts "\n--- Model Parameters ---"

model.components.each_with_index do |comp, idx|
  puts "Component #{idx}:"
  puts "  Weight:   #{comp.weight.round(4)}"
  puts "  Mean:     #{comp.mean.map(&.round(4))}"
  puts "  Variance: #{comp.variance.map(&.round(4))}"
end

puts "\n--- Soft Assignments (Probabilities) ---"

# Let's predict probabilities for each point in our training set
data.each_with_index do |point, idx|
  probs = model.predict_probabilities(point)
  hard_label = model.predict(point)
  puts "Point #{idx} #{point} -> P(C0)=#{(probs[0]*100).round(1)}% | P(C1)=#{(probs[1]*100).round(1)}% -> Hard Label: #{hard_label}"
end
