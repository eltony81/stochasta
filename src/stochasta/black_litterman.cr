module Stochasta
  module Portfolio
    module BlackLitterman
      # Matrix multiplication: A * B
      def self.multiply(a : Array(Array(Float64)), b : Array(Array(Float64))) : Array(Array(Float64))
        rows_a = a.size
        cols_a = a.first.size
        cols_b = b.first.size
        
        result = Array(Array(Float64)).new(rows_a) { Array(Float64).new(cols_b, 0.0) }
        rows_a.times do |i|
          cols_b.times do |j|
            sum = 0.0
            cols_a.times do |k|
              sum += a[i][k] * b[k][j]
            end
            result[i][j] = sum
          end
        end
        result
      end

      # Matrix-Vector multiplication: A * v
      def self.multiply_vector(a : Array(Array(Float64)), v : Array(Float64)) : Array(Float64)
        rows = a.size
        cols = a.first.size
        result = Array(Float64).new(rows, 0.0)
        rows.times do |i|
          sum = 0.0
          cols.times do |k|
            sum += a[i][k] * v[k]
          end
          result[i] = sum
        end
        result
      end

      # Matrix Transpose
      def self.transpose(a : Array(Array(Float64))) : Array(Array(Float64))
        rows = a.size
        cols = a.first.size
        result = Array(Array(Float64)).new(cols) { Array(Float64).new(rows, 0.0) }
        rows.times do |i|
          cols.times do |j|
            result[j][i] = a[i][j]
          end
        end
        result
      end

      # Invert matrix using Gauss-Jordan elimination
      def self.invert(matrix : Array(Array(Float64))) : Array(Array(Float64))
        n = matrix.size
        # Augment matrix with identity matrix
        a = matrix.map(&.dup)
        i_mat = Array(Array(Float64)).new(n) { |i| Array(Float64).new(n) { |j| i == j ? 1.0 : 0.0 } }

        n.times do |i|
          # Find pivot
          pivot_row = i
          ((i + 1)...n).each do |r|
            pivot_row = r if a[r][i].abs > a[pivot_row][i].abs
          end

          # Swap rows
          a[i], a[pivot_row] = a[pivot_row], a[i]
          i_mat[i], i_mat[pivot_row] = i_mat[pivot_row], i_mat[i]

          pivot = a[i][i]
          raise ArgumentError.new("Matrix is singular and cannot be inverted") if pivot.abs < 1e-12

          # Normalize pivot row
          n.times do |j|
            a[i][j] /= pivot
            i_mat[i][j] /= pivot
          end

          # Eliminate other rows
          n.times do |r|
            next if r == i
            factor = a[r][i]
            n.times do |j|
              a[r][j] -= factor * a[i][j]
              i_mat[r][j] -= factor * i_mat[i][j]
            end
          end
        end

        i_mat
      end

      # Computes the Black-Litterman expected returns vector
      # Params:
      # - prior_returns: expected returns prior (Pi vector, size N)
      # - covariance: asset covariance matrix (Sigma matrix, size N x N)
      # - p_matrix: view picker matrix (P matrix, size K x N)
      # - q_vector: investor views vector (Q vector, size K)
      # - tau: scale factor of prior covariance (usually 0.025 to 0.05)
      # - omega: covariance of view uncertainty (K x K diagonal matrix, or estimated automatically if nil)
      def self.estimate_returns(
        prior_returns : Array(Float64),
        covariance : Array(Array(Float64)),
        p_matrix : Array(Array(Float64)),
        q_vector : Array(Float64),
        tau : Float64 = 0.025,
        omega : Array(Array(Float64))? = nil
      ) : Array(Float64)
        n = prior_returns.size
        k = q_vector.size

        # Prior covariance matrix tau * Sigma
        tau_sigma = covariance.map { |row| row.map { |v| v * tau } }
        tau_sigma_inv = invert(tau_sigma)

        # If omega is not provided, estimate it as P * (tau * Sigma) * P^T (diagonal elements only)
        resolved_omega = if omega.nil?
                           # Compute P * (tau * Sigma) * P^T
                           p_tau_sigma = multiply(p_matrix, tau_sigma)
                           p_tau_sigma_pt = multiply(p_tau_sigma, transpose(p_matrix))
                           
                           # Keep only diagonal elements to represent independent view uncertainties
                           Array(Array(Float64)).new(k) do |i|
                             Array(Float64).new(k) do |j|
                               if i == j
                                 val = p_tau_sigma_pt[i][j]
                                 val < 1e-6 ? 1e-6 : val
                               else
                                 0.0
                               end
                             end
                           end
                         else
                           omega
                         end

        omega_inv = invert(resolved_omega)

        # Term A: P^T * omega_inv
        pt = transpose(p_matrix)
        pt_omega_inv = multiply(pt, omega_inv)

        # Term B: P^T * omega_inv * P
        pt_omega_inv_p = multiply(pt_omega_inv, p_matrix)

        # Left term to invert: (tau * Sigma)^-1 + P^T * omega_inv * P
        left_matrix = Array(Array(Float64)).new(n) do |i|
          Array(Float64).new(n) do |j|
            tau_sigma_inv[i][j] + pt_omega_inv_p[i][j]
          end
        end
        left_matrix_inv = invert(left_matrix)

        # Right term vector: (tau * Sigma)^-1 * Pi + P^T * omega_inv * Q
        tau_sigma_inv_pi = multiply_vector(tau_sigma_inv, prior_returns)
        pt_omega_inv_q = multiply_vector(pt_omega_inv, q_vector)

        right_vector = Array(Float64).new(n) do |i|
          tau_sigma_inv_pi[i] + pt_omega_inv_q[i]
        end

        # E[R] = LeftMatrix^-1 * RightVector
        multiply_vector(left_matrix_inv, right_vector)
      end
    end
  end
end
