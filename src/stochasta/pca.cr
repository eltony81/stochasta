module Stochasta
  class PCA
    property n_components : Int32
    property means : Array(Float64)
    property components : Array(Array(Float64)) # Eigenvectors (rows are components)
    property eigenvalues : Array(Float64)
    property explained_variance_ratio : Array(Float64)

    def initialize(@n_components : Int32)
      @means = [] of Float64
      @components = [] of Array(Float64)
      @eigenvalues = [] of Float64
      @explained_variance_ratio = [] of Float64
    end

    # Fits the PCA model on a dataset
    def fit(data : Array(Array(Float64))) : self
      raise ArgumentError.new("Empty dataset") if data.empty?
      n_samples = data.size
      n_features = data.first.size
      raise ArgumentError.new("n_components must be <= n_features") if @n_components > n_features

      # 1. Compute means
      @means = Array(Float64).new(n_features, 0.0)
      data.each do |row|
        n_features.times do |j|
          @means[j] += row[j]
        end
      end
      @means.map! { |sum| sum / n_samples }

      # 2. Compute Covariance Matrix (size n_features x n_features)
      cov = Array(Array(Float64)).new(n_features) { Array(Float64).new(n_features, 0.0) }
      n_features.times do |j|
        n_features.times do |k|
          sum = 0.0
          data.each do |row|
            sum += (row[j] - @means[j]) * (row[k] - @means[k])
          end
          cov[j][k] = sum / (n_samples - 1)
        end
      end

      # 3. Find eigenvalues/eigenvectors of Covariance matrix using Jacobi method
      evals, evecs = PCA.jacobi_eigenvalues(cov)

      # 4. Sort by eigenvalues descending
      indices = (0...n_features).to_a.sort_by { |i| -evals[i] }
      
      sorted_evals = indices.map { |i| evals[i] }
      sorted_evecs = indices.map { |i| evecs.map { |row| row[i] } } # Column vectors to row vectors

      # 5. Store principal components and variance ratios
      total_variance = sorted_evals.sum
      total_variance = 1.0 if total_variance.abs < 1e-9

      @eigenvalues = sorted_evals[0...@n_components]
      @components = sorted_evecs[0...@n_components]
      @explained_variance_ratio = @eigenvalues.map { |val| val / total_variance }

      self
    end

    # Transform the dataset into the reduced dimensional space
    def transform(data : Array(Array(Float64))) : Array(Array(Float64))
      data.map do |row|
        # Center the data point
        centered = Array(Float64).new(row.size) { |i| row[i] - @means[i] }
        
        # Project onto each of the selected components
        Array(Float64).new(@n_components) do |c_idx|
          component = @components[c_idx]
          val = 0.0
          centered.size.times do |i|
            val += centered[i] * component[i]
          end
          val
        end
      end
    end

    # Fit and transform
    def fit_transform(data : Array(Array(Float64))) : Array(Array(Float64))
      fit(data)
      transform(data)
    end

    # Jacobi eigenvalue algorithm for symmetric matrix A
    # Returns {eigenvalues, eigenvectors_matrix} where eigenvectors_matrix is matrix of columns
    def self.jacobi_eigenvalues(matrix : Array(Array(Float64)), max_iterations : Int32 = 1000, tolerance : Float64 = 1e-9) : Tuple(Array(Float64), Array(Array(Float64)))
      n = matrix.size
      a = matrix.map(&.dup)
      v = Array(Array(Float64)).new(n) { |i| Array(Float64).new(n) { |j| i == j ? 1.0 : 0.0 } }

      max_iterations.times do
        # Find the largest off-diagonal element
        row = 0
        col = 1
        max_val = 0.0

        n.times do |i|
          n.times do |j|
            if i < j && a[i][j].abs > max_val
              max_val = a[i][j].abs
              row = i
              col = j
            end
          end
        end

        break if max_val < tolerance

        # Compute rotation angle
        app = a[row][row]
        aqq = a[col][col]
        apq = a[row][col]

        theta = 0.5 * (aqq - app) / apq
        t = if theta >= 0
              1.0 / (theta + Math.sqrt(theta * theta + 1.0))
            else
              -1.0 / (-theta + Math.sqrt(theta * theta + 1.0))
            end

        c = 1.0 / Math.sqrt(t * t + 1.0)
        s = t * c

        # Rotate matrix a
        tau = s / (1.0 + c)
        
        # Update elements row and col of diagonal
        a[row][row] = app - t * apq
        a[col][col] = aqq + t * apq
        a[row][col] = 0.0
        a[col][row] = 0.0

        n.times do |i|
          next if i == row || i == col
          arp = a[i][row]
          arq = a[i][col]
          
          a[i][row] = arp - s * (arq + arp * tau)
          a[row][i] = a[i][row]
          
          a[i][col] = arq + s * (arp - arq * tau)
          a[col][i] = a[i][col]
        end

        # Update eigenvectors matrix v
        n.times do |i|
          vip = v[i][row]
          viq = v[i][col]
          
          v[i][row] = vip - s * (viq + vip * tau)
          v[i][col] = viq + s * (vip - viq * tau)
        end
      end

      eigenvalues = Array(Float64).new(n) { |i| a[i][i] }
      {eigenvalues, v}
    end
  end
end
