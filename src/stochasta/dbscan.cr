module Stochasta
  module DBSCAN
    class Model
      property eps : Float64
      property min_samples : Int32
      property labels : Array(Int32)

      # Label constants
      NOISE = -1
      UNVISITED = -2

      def initialize(@eps : Float64, @min_samples : Int32)
        @labels = [] of Int32
      end

      # Fits the data and returns the cluster assignments
      def fit(data : Array(Array(Float64))) : self
        n = data.size
        @labels = Array(Int32).new(n, UNVISITED)
        cluster_id = 0

        n.times do |i|
          next if @labels[i] != UNVISITED

          # Find neighbors
          neighbors = range_query(data, i)

          if neighbors.size < @min_samples
            @labels[i] = NOISE
          else
            # Start a new cluster
            @labels[i] = cluster_id
            expand_cluster(data, neighbors, cluster_id)
            cluster_id += 1
          end
        end

        self
      end

      private def expand_cluster(data : Array(Array(Float64)), neighbors : Array(Int32), cluster_id : Int32)
        queue = neighbors.dup
        # Use a Set/hash-based lookup for speed
        in_queue = Set(Int32).new(queue)

        idx = 0
        while idx < queue.size
          point_idx = queue[idx]
          idx += 1

          if @labels[point_idx] == NOISE
            # Noise point becomes border point of the cluster
            @labels[point_idx] = cluster_id
          end

          next if @labels[point_idx] != UNVISITED

          # Mark as part of cluster
          @labels[point_idx] = cluster_id

          # Find neighbors of this point
          new_neighbors = range_query(data, point_idx)

          if new_neighbors.size >= @min_samples
            new_neighbors.each do |n_idx|
              unless in_queue.includes?(n_idx)
                queue << n_idx
                in_queue << n_idx
              end
            end
          end
        end
      end

      private def range_query(data : Array(Array(Float64)), query_idx : Int32) : Array(Int32)
        query_point = data[query_idx]
        neighbors = [] of Int32
        data.each_with_index do |point, idx|
          if KMeans.euclidean_distance(query_point, point) <= @eps
            neighbors << idx
          end
        end
        neighbors
      end
    end
  end
end
