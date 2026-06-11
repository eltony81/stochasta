# Hierarchical Agglomerative Clustering Example
#
# This example demonstrates bottom-up hierarchical clustering on a 2D dataset
# using Average Linkage distance metrics.

require "../src/stochasta"

# Define 2D dataset containing three clusters
data = [
  # Group A
  [1.0, 1.0],
  [1.2, 0.8],

  # Group B
  [5.0, 5.0],
  [4.8, 5.2],

  # Group C
  [10.0, 10.0],
  [9.8, 10.2]
]

# Instantiate Agglomerative Hierarchical Model with K=3 clusters
# We use Linkage::Average to calculate distances between clusters
model = Stochasta::Hierarchical::Model.new(
  k: 3, 
  linkage: Stochasta::Hierarchical::Linkage::Average
).fit(data)

puts "Hierarchical Clustering Complete."
puts "\n--- Points and Assigned Clusters ---"

data.each_with_index do |point, idx|
  cluster_id = model.labels[idx]
  puts "Point #{idx} #{point} -> Assigned to Cluster #{cluster_id}"
end
