module Stochasta
  module KMeans
    # Calculates Euclidean distance between two vectors
    def self.euclidean_distance(v1 : Array(Float64), v2 : Array(Float64)) : Float64
      sum = 0.0
      v1.size.times do |i|
        diff = v1[i] - v2[i]
        sum += diff * diff
      end
      Math.sqrt(sum)
    end

    class Model
      property centroids : Array(Array(Float64))
      property assignments : Array(Int32)
      property k : Int32
      property max_iter : Int32
      property tolerance : Float64

      def initialize(@k : Int32, @max_iter : Int32 = 300, @tolerance : Float64 = 1e-4)
        @centroids = [] of Array(Float64)
        @assignments = [] of Int32
      end

      # Fits the data and clusters them
      def fit(data : Array(Array(Float64))) : self
        raise ArgumentError.new("Empty dataset provided") if data.empty?
        raise ArgumentError.new("K must be <= number of data points") if @k > data.size

        dimensions = data.first.size
        # Initialize centroids randomly from the dataset
        @centroids = data.sample(@k).map(&.dup)
        @assignments = Array(Int32).new(data.size, -1)

        @max_iter.times do
          # 1. Assign points to the closest centroid
          new_assignments = Array(Int32).new(data.size)
          data.each do |point|
            min_dist = Float64::MAX
            best_centroid_idx = 0
            @centroids.each_with_index do |centroid, idx|
              dist = KMeans.euclidean_distance(point, centroid)
              if dist < min_dist
                min_dist = dist
                best_centroid_idx = idx
              end
            end
            new_assignments << best_centroid_idx
          end

          # 2. Compute new centroids as the mean of all points assigned to them
          cluster_sums = Array(Array(Float64)).new(@k) { Array(Float64).new(dimensions, 0.0) }
          cluster_counts = Array(Int32).new(@k, 0)

          data.each_with_index do |point, point_idx|
            c_idx = new_assignments[point_idx]
            cluster_counts[c_idx] += 1
            dimensions.times do |d|
              cluster_sums[c_idx][d] += point[d]
            end
          end

          new_centroids = Array(Array(Float64)).new(@k)
          @k.times do |c_idx|
            if cluster_counts[c_idx] > 0
              new_centroids << cluster_sums[c_idx].map { |sum| sum / cluster_counts[c_idx] }
            else
              # If a cluster is empty, pick a random point from the data as the new centroid
              new_centroids << data.sample.dup
            end
          end

          # 3. Check for convergence (if centroids didn't move significantly)
          max_movement = 0.0
          @k.times do |i|
            dist = KMeans.euclidean_distance(@centroids[i], new_centroids[i])
            max_movement = dist if dist > max_movement
          end

          @centroids = new_centroids
          @assignments = new_assignments

          break if max_movement < @tolerance
        end

        self
      end

      # Predict the closest cluster for a new data point
      def predict(point : Array(Float64)) : Int32
        min_dist = Float64::MAX
        best_centroid_idx = 0
        @centroids.each_with_index do |centroid, idx|
          dist = KMeans.euclidean_distance(point, centroid)
          if dist < min_dist
            min_dist = dist
            best_centroid_idx = idx
          end
        end
        best_centroid_idx
      end
    end
  end
end
