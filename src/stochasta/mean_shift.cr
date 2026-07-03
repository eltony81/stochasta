module Stochasta
  module MeanShift
    class Model
      property bandwidth : Float64
      property max_iter : Int32
      property tolerance : Float64
      property centroids : Array(Array(Float64))
      property assignments : Array(Int32)

      def initialize(@bandwidth : Float64, @max_iter : Int32 = 300, @tolerance : Float64 = 1e-3)
        @centroids = [] of Array(Float64)
        @assignments = [] of Int32
      end

      # Fits the Mean Shift clustering model
      def fit(data : Array(Array(Float64))) : self
        raise ArgumentError.new("Empty dataset") if data.empty?
        raise ArgumentError.new("Bandwidth must be > 0") if @bandwidth <= 0
        n = data.size
        dimensions = data.first.size

        # Copy original points to perform shift
        shifted_points = data.map(&.dup)

        @max_iter.times do
          max_shift = 0.0

          n.times do |i|
            point = shifted_points[i]
            
            # Find all points within bandwidth
            in_window = [] of Array(Float64)
            data.each do |other|
              if KMeans.euclidean_distance(point, other) <= @bandwidth
                in_window << other
              end
            end

            # Calculate mean of points in window
            new_point = Array(Float64).new(dimensions, 0.0)
            in_window.each do |w_point|
              dimensions.times { |d| new_point[d] += w_point[d] }
            end
            new_point.map! { |sum| sum / in_window.size }

            # Calculate shift distance
            shift = KMeans.euclidean_distance(point, new_point)
            max_shift = shift if shift > max_shift

            shifted_points[i] = new_point
          end

          break if max_shift < @tolerance
        end

        # Group converged points into unique cluster centroids
        @centroids = [] of Array(Float64)
        shifted_points.each do |point|
          # Round values slightly to prevent numerical noise causing separate clusters
          rounded_point = point.map { |v| (v / @tolerance).round * @tolerance }

          # Check if this center is already added
          found = false
          @centroids.each do |centroid|
            if KMeans.euclidean_distance(rounded_point, centroid) <= @bandwidth
              found = true
              break
            end
          end
          @centroids << rounded_point unless found
        end

        # Assign original points to the closest centroid
        @assignments = Array(Int32).new(n)
        data.each do |row|
          min_dist = Float64::MAX
          best_idx = 0
          @centroids.each_with_index do |centroid, idx|
            dist = KMeans.euclidean_distance(row, centroid)
            if dist < min_dist
              min_dist = dist
              best_idx = idx
            end
          end
          @assignments << best_idx
        end

        self
      end

      # Predict cluster of a new point
      def predict(point : Array(Float64)) : Int32
        min_dist = Float64::MAX
        best_idx = 0
        @centroids.each_with_index do |centroid, idx|
          dist = KMeans.euclidean_distance(point, centroid)
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
