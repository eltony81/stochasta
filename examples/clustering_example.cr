# Clustering & Dimensionality Reduction Example
#
# This example shows:
# 1. Reducing 3D correlated data to 2D using PCA.
# 2. Applying K-Means clustering to partition the dataset.
# 3. Applying DBSCAN clustering to find core groups and noise.

require "../src/stochasta"

# Define a 3D dataset containing three distinct regions and one outlier (noise point)
data = [
  # Group A (Near [0, 0, 0])
  [0.1, 0.2, 0.15],
  [0.2, 0.1, 0.05],
  [0.05, 0.15, 0.1],
  
  # Group B (Near [10, 10, 10])
  [10.1, 9.9, 10.05],
  [9.8, 10.2, 10.0],
  [10.2, 9.8, 9.95],

  # Group C (Near [20, 20, 20])
  [20.0, 20.1, 19.9],
  [20.2, 19.8, 20.0],
  [19.9, 20.2, 20.1],

  # Outlier / Noise point
  [50.0, 50.0, 50.0]
]

puts "Original Data Points (3D): #{data.size}"


# --- 1. Principal Component Analysis (PCA) ---
puts "\n--- 1. Principal Component Analysis (PCA) ---"
# Reduce the 3D data into 2D
pca = Stochasta::PCA.new(n_components: 2).fit(data)

puts "Explained Variance Ratio per Component:"
pca.explained_variance_ratio.each_with_index do |ratio, idx|
  puts "  PC #{idx + 1}: #{(ratio * 100).round(2)}%"
end

# Project original data points to the 2D space
reduced_data = pca.transform(data)
puts "Reduced Coordinates (First 3):"
reduced_data[0...3].each { |pt| puts "  #{pt.map(&.round(4))}" }


# --- 2. K-Means Clustering ---
puts "\n--- 2. K-Means Clustering ---"
# Partition our reduced 2D dataset into K=3 clusters
kmeans = Stochasta::KMeans::Model.new(k: 3, max_iter: 100).fit(reduced_data)

puts "Centroids of the 3 Clusters:"
kmeans.centroids.each_with_index do |c, idx|
  puts "  Cluster #{idx}: #{c.map(&.round(4))}"
end

# Check which cluster each data point is assigned to
puts "Assignments per point:"
reduced_data.each_with_index do |point, idx|
  cluster_idx = kmeans.predict(point)
  puts "  Point #{idx} #{data[idx]} -> Assigned to Cluster #{cluster_idx}"
end


# --- 3. DBSCAN Clustering ---
puts "\n--- 3. DBSCAN Clustering (Density-Based) ---"
# DBSCAN finds density regions using Epsilon (eps) radius and Min Samples.
# We set eps = 3.0, min_samples = 2.
dbscan = Stochasta::DBSCAN::Model.new(eps: 3.0, min_samples: 2).fit(data)

puts "DBSCAN assignments (Noise points are labeled as -1):"
dbscan.labels.each_with_index do |label, idx|
  label_str = label == -1 ? "NOISE" : "Cluster #{label}"
  puts "  Point #{idx} #{data[idx]} -> #{label_str}"
end
