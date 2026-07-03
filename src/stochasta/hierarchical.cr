module Stochasta
  module Hierarchical
    enum Linkage
      Single
      Complete
      Average
    end

    class Model
      property k : Int32
      property linkage : Linkage
      property labels : Array(Int32)

      def initialize(@k : Int32, @linkage : Linkage = Linkage::Average)
        @labels = [] of Int32
      end

      # Fits the dataset and labels each point with its cluster ID
      def fit(data : Array(Array(Float64))) : self
        n = data.size
        raise ArgumentError.new("Empty dataset") if n == 0
        raise ArgumentError.new("K must be >= 1") if @k < 1
        raise ArgumentError.new("K must be <= number of points") if @k > n

        # Initially, each point is its own cluster
        # clusters[i] contains indices of points in that cluster
        clusters = (0...n).map { |i| [i] }

        # Precompute distance matrix between all pairs of individual points
        dist_matrix = Array(Array(Float64)).new(n) { Array(Float64).new(n, 0.0) }
        n.times do |i|
          n.times do |j|
            if i < j
              dist_matrix[i][j] = KMeans.euclidean_distance(data[i], data[j])
              dist_matrix[j][i] = dist_matrix[i][j]
            end
          end
        end

        # Merge clusters until only K remain
        while clusters.size > @k
          min_dist = Float64::MAX
          best_i = 0
          best_j = 1

          # Find the two closest clusters
          clusters.size.times do |i|
            ((i + 1)...clusters.size).each do |j|
              dist = cluster_distance(clusters[i], clusters[j], dist_matrix)
              if dist < min_dist
                min_dist = dist
                best_i = i
                best_j = j
              end
            end
          end

          # Merge cluster best_j into best_i
          clusters[best_i].concat(clusters[best_j])
          clusters.delete_at(best_j)
        end

        # Map cluster groupings back to labels array
        @labels = Array(Int32).new(n, -1)
        clusters.each_with_index do |cluster, c_idx|
          cluster.each do |point_idx|
            @labels[point_idx] = c_idx
          end
        end

        self
      end

      # Computes the distance between two clusters based on the linkage criteria
      private def cluster_distance(c1 : Array(Int32), c2 : Array(Int32), dist_matrix : Array(Array(Float64))) : Float64
        case @linkage
        when Linkage::Single
          min = Float64::MAX
          c1.each do |i|
            c2.each do |j|
              d = dist_matrix[i][j]
              min = d if d < min
            end
          end
          min
        when Linkage::Complete
          max = -1.0
          c1.each do |i|
            c2.each do |j|
              d = dist_matrix[i][j]
              max = d if d > max
            end
          end
          max
        when Linkage::Average
          sum = 0.0
          c1.each do |i|
            c2.each do |j|
              sum += dist_matrix[i][j]
            end
          end
          sum / (c1.size * c2.size)
        else
          raise "Unknown linkage"
        end
      end
    end
  end
end
