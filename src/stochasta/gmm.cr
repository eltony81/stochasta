module Stochasta
  module GMM
    class Component
      property weight : Float64
      property mean : Array(Float64)
      property variance : Array(Float64) # Diagonal covariance (variances)

      def initialize(@weight : Float64, @mean : Array(Float64), @variance : Array(Float64))
      end

      # Computes the probability density function for a given point
      def pdf(x : Array(Float64)) : Float64
        d = x.size
        prod = 1.0
        d.times do |i|
          v = @variance[i]
          # Avoid division by zero
          v = 1e-9 if v < 1e-9
          diff = x[i] - @mean[i]
          exponent = Math.exp(-(diff * diff) / (2.0 * v))
          denom = Math.sqrt(2.0 * Math::PI * v)
          prod *= (exponent / denom)
        end
        prod
      end
    end

    class Model
      property components : Array(Component)
      property k : Int32
      property max_iter : Int32
      property tolerance : Float64

      def initialize(@k : Int32, @max_iter : Int32 = 100, @tolerance : Float64 = 1e-4)
        @components = [] of Component
      end

      # Fits the GMM on the dataset using Expectation-Maximization
      def fit(data : Array(Array(Float64))) : self
        raise ArgumentError.new("Empty dataset") if data.empty?
        n_samples = data.size
        n_features = data.first.size
        raise ArgumentError.new("K must be <= number of samples") if @k > n_samples

        # 1. Initialize components spread out deterministically from dataset
        samples = Array(Array(Float64)).new(@k) do |i|
          data[(i * n_samples) // @k].dup
        end
        
        # Estimate global variance for initial variance
        global_mean = Array(Float64).new(n_features, 0.0)
        data.each { |row| n_features.times { |d| global_mean[d] += row[d] } }
        global_mean.map! { |sum| sum / n_samples }
        
        global_variance = Array(Float64).new(n_features, 0.0)
        data.each do |row|
          n_features.times do |d|
            diff = row[d] - global_mean[d]
            global_variance[d] += diff * diff
          end
        end
        global_variance.map! { |sum| sum / n_samples }
        global_variance.map! { |v| v < 1e-3 ? 1e-3 : v } # Clamp minimum variance

        @components = Array(Component).new(@k) do |i|
          Component.new(1.0 / @k, samples[i].dup, global_variance.dup)
        end

        # Responsibilities matrix [n_samples x k]
        r = Array(Array(Float64)).new(n_samples) { Array(Float64).new(@k, 0.0) }

        @max_iter.times do |iter|
          # --- E-Step: Compute responsibilities ---
          n_samples.times do |i|
            row = data[i]
            sum = 0.0
            @k.times do |j|
              val = @components[j].weight * @components[j].pdf(row)
              r[i][j] = val
              sum += val
            end
            
            # Normalize
            if sum > 1e-12
              @k.times { |j| r[i][j] /= sum }
            else
              @k.times { |j| r[i][j] = 1.0 / @k }
            end
          end

          # --- M-Step: Update parameters ---
          prev_means = @components.map { |c| c.mean.dup }

          @k.times do |j|
            # Sum of responsibilities for component j
            n_j = 0.0
            n_samples.times { |i| n_j += r[i][j] }
            
            # Avoid division by zero
            n_j_safe = n_j < 1e-12 ? 1e-12 : n_j

            # 1. Update Mean
            new_mean = Array(Float64).new(n_features, 0.0)
            n_samples.times do |i|
              n_features.times do |d|
                new_mean[d] += r[i][j] * data[i][d]
              end
            end
            new_mean.map! { |sum| sum / n_j_safe }

            # 2. Update Variance
            new_variance = Array(Float64).new(n_features, 0.0)
            n_samples.times do |i|
              n_features.times do |d|
                diff = data[i][d] - new_mean[d]
                new_variance[d] += r[i][j] * diff * diff
              end
            end
            new_variance.map! { |sum| sum / n_j_safe }
            new_variance.map! { |v| v < 1e-5 ? 1e-5 : v } # Ensure variance is strictly positive

            # 3. Update Weight
            new_weight = n_j / n_samples

            @components[j].mean = new_mean
            @components[j].variance = new_variance
            @components[j].weight = new_weight
          end

          # Check convergence: mean movement
          max_movement = 0.0
          @k.times do |j|
            movement = KMeans.euclidean_distance(prev_means[j], @components[j].mean)
            max_movement = movement if movement > max_movement
          end

          break if max_movement < @tolerance
        end

        self
      end

      # Predicts the responsibilities (soft assignments) for a given point
      def predict_probabilities(x : Array(Float64)) : Array(Float64)
        probs = @components.map { |c| c.weight * c.pdf(x) }
        sum = probs.sum
        if sum > 1e-12
          probs.map { |p| p / sum }
        else
          Array(Float64).new(@k, 1.0 / @k)
        end
      end

      # Predicts the most likely cluster (hard assignment) for a given point
      def predict(x : Array(Float64)) : Int32
        probs = predict_probabilities(x)
        max_val = -1.0
        best_idx = 0
        probs.each_with_index do |p, idx|
          if p > max_val
            max_val = p
            best_idx = idx
          end
        end
        best_idx
      end
    end
  end
end
