# K-Medoids Clustering Example
#
# This example shows how K-Medoids partitions a dataset into K clusters.
# Unlike K-Means, K-Medoids enforces that the cluster centers (medoids)
# must be actual data points from the dataset.

require "../src/stochasta"

# Define a dataset with obvious clusters and one clear outlier
data = [
  # Cluster A
  [1.0, 1.0],
  [1.2, 0.9],
  [0.8, 1.1],

  # Cluster B
  [10.0, 10.0],
  [9.8, 10.2],
  [10.2, 9.8],

  # Outlier
  [100.0, 100.0]
]

# Fit K-Medoids model with K=2 clusters
model = Stochasta::KMedoids::Model.new(k: 2).fit(data)

puts "K-Medoids Clustering Complete."
puts "\n--- Medoids (Cluster Centers) ---"

model.medoids.each_with_index do |medoid, idx|
  # Print the medoid, which corresponds exactly to a point in the data array
  puts "Medoid #{idx}: #{medoid} (Is in dataset: #{data.includes?(medoid)})"
end

puts "\n--- Assignments ---"
data.each_with_index do |point, idx|
  cluster_idx = model.predict(point)
  puts "Point #{idx} #{point} -> Cluster #{cluster_idx}"
end
