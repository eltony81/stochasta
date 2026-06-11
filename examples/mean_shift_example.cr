# Mean Shift Clustering Example
#
# This example shows how Mean Shift dynamically uncovers cluster centroids
# based on sample density, without requiring the user to specify K (the number of clusters).

require "../src/stochasta"

# Define 2D dataset with two dense clusters and a single noise point
data = [
  # Dense cluster A
  [1.0, 1.0],
  [1.1, 0.9],
  [0.9, 1.1],

  # Dense cluster B
  [20.0, 20.0],
  [20.1, 19.9],
  [19.9, 20.1]
]

# Instantiate Mean Shift model
# We set bandwidth = 3.0. This acts as the search radius.
model = Stochasta::MeanShift::Model.new(bandwidth: 3.0).fit(data)

puts "Mean Shift Clustering Complete."
puts "\n--- Found Centroids (K determined automatically) ---"
puts "Number of clusters found: #{model.centroids.size}"

model.centroids.each_with_index do |centroid, idx|
  puts "  Centroid #{idx}: #{centroid.map(&.round(4))}"
end

puts "\n--- Assignments ---"
data.each_with_index do |point, idx|
  cluster_idx = model.predict(point)
  puts "  Point #{idx} #{point} -> Assigned to Cluster #{cluster_idx}"
end
