module Stochasta
  module KMedoids
    class Model
      property medoids : Array(Array(Float64))
      property assignments : Array(Int32)
      property k : Int32
      property max_iter : Int32

      def initialize(@k : Int32, @max_iter : Int32 = 100)
        @medoids = [] of Array(Float64)
        @assignments = [] of Int32
      end

      # Fits the dataset and assigns clusters
      def fit(data : Array(Array(Float64))) : self
        raise ArgumentError.new("Empty dataset") if data.empty?
        raise ArgumentError.new("K must be >= 1") if @k < 1
        n = data.size
        raise ArgumentError.new("K must be <= number of points") if @k > n

        # Initialize medoids spread out deterministically from the dataset
        medoid_indices = Array(Int32).new(@k) do |i|
          (i * n) // @k
        end
        @medoids = medoid_indices.map { |idx| data[idx].dup }
        @assignments = Array(Int32).new(n, -1)

        @max_iter.times do
          # 1. Assign each point to the closest medoid
          new_assignments = Array(Int32).new(n)
          data.each do |point|
            min_dist = Float64::MAX
            best_idx = 0
            @medoids.each_with_index do |medoid, idx|
              dist = KMeans.euclidean_distance(point, medoid)
              if dist < min_dist
                min_dist = dist
                best_idx = idx
              end
            end
            new_assignments << best_idx
          end

          # 2. Update Medoids: find the point in each cluster that minimizes sum of distances to others in the cluster
          new_medoids = Array(Array(Float64)).new(@k)
          
          @k.times do |c_idx|
            # Collect points belonging to cluster c_idx
            cluster_points = [] of Array(Float64)
            data.each_with_index do |point, idx|
              cluster_points << point if new_assignments[idx] == c_idx
            end

            if cluster_points.empty?
              # If empty, pick a random point from the entire dataset
              new_medoids << data.sample.dup
            else
              # Find point that minimizes sum of distances
              best_medoid = cluster_points.first
              min_total_dist = Float64::MAX
              
              cluster_points.each do |candidate|
                sum_dist = 0.0
                cluster_points.each do |other|
                  sum_dist += KMeans.euclidean_distance(candidate, other)
                end
                if sum_dist < min_total_dist
                  min_total_dist = sum_dist
                  best_medoid = candidate
                end
              end
              new_medoids << best_medoid.dup
            end
          end

          # Check convergence
          medoids_changed = false
          @k.times do |i|
            if KMeans.euclidean_distance(@medoids[i], new_medoids[i]) > 1e-9
              medoids_changed = true
              break
            end
          end

          @medoids = new_medoids
          @assignments = new_assignments

          break unless medoids_changed
        end

        self
      end

      # Predict cluster of a new point
      def predict(point : Array(Float64)) : Int32
        min_dist = Float64::MAX
        best_idx = 0
        @medoids.each_with_index do |medoid, idx|
          dist = KMeans.euclidean_distance(point, medoid)
          if dist < min_dist
            min_dist = dist
            best_idx = idx
          end
        end
        best_idx
      end
    end
  end
end
