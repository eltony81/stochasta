module Stochasta
  module Portfolio
    class Optimizer
      property expected_returns : Array(Float64)
      property covariance_matrix : Array(Array(Float64))
      property n_assets : Int32

      def initialize(@expected_returns : Array(Float64), @covariance_matrix : Array(Array(Float64)))
        @n_assets = @expected_returns.size
        raise ArgumentError.new("Covariance matrix must be square and match expected returns size") if @covariance_matrix.size != @n_assets || @covariance_matrix.any? { |row| row.size != @n_assets }
      end

      # Computes the expected return of a portfolio given its weights
      def portfolio_return(weights : Array(Float64)) : Float64
        ret = 0.0
        @n_assets.times { |i| ret += weights[i] * @expected_returns[i] }
        ret
      end

      # Computes the variance of a portfolio given its weights
      def portfolio_variance(weights : Array(Float64)) : Float64
        variance = 0.0
        @n_assets.times do |i|
          @n_assets.times do |j|
            variance += weights[i] * weights[j] * @covariance_matrix[i][j]
          end
        end
        variance
      end

      # Normalizes a raw weight vector to sum to 1.0
      def normalize_weights(raw : Array(Float64)) : Array(Float64)
        sum = raw.sum
        sum = 1e-9 if sum.abs < 1e-9
        raw.map { |w| w.abs / sum.abs }
      end

      # Finds the weights that minimize the portfolio variance
      # Constraints: long-only (weights >= 0), sum of weights = 1.0
      def min_variance_weights(iterations : Int32 = 100, swarm_size : Int32 = 40) : Array(Float64)
        # Bounding box for weights (0.0 to 1.0 before normalization)
        bounds = Array(Tuple(Float64, Float64)).new(@n_assets, {0.0, 1.0})

        # Cost function: calculate variance of normalized weights
        cost_fn = ->(position : Array(Float64)) {
          w = normalize_weights(position)
          portfolio_variance(w)
        }

        pso = Stochasta::PSO::Optimizer.new(
          swarm_size: swarm_size,
          bounds: bounds,
          w: 0.5,
          c1: 1.5,
          c2: 1.5,
          minimize: true,
          &cost_fn
        )

        best_raw, _ = pso.optimize(iterations, &cost_fn)
        normalize_weights(best_raw)
      end

      # Finds the weights that maximize the Sharpe Ratio
      # Constraints: long-only (weights >= 0), sum of weights = 1.0
      def max_sharpe_weights(risk_free_rate : Float64 = 0.0, iterations : Int32 = 100, swarm_size : Int32 = 40) : Array(Float64)
        bounds = Array(Tuple(Float64, Float64)).new(@n_assets, {0.0, 1.0})

        # Cost function: minimize the negative Sharpe Ratio of normalized weights
        cost_fn = ->(position : Array(Float64)) {
          w = normalize_weights(position)
          p_return = portfolio_return(w)
          p_variance = portfolio_variance(w)
          p_std_dev = Math.sqrt(p_variance)
          
          p_std_dev = 1e-9 if p_std_dev < 1e-9
          # Sharpe = (Return - Rf) / StdDev. To maximize it, we minimize negative Sharpe.
          -(p_return - risk_free_rate) / p_std_dev
        }

        pso = Stochasta::PSO::Optimizer.new(
          swarm_size: swarm_size,
          bounds: bounds,
          w: 0.5,
          c1: 1.5,
          c2: 1.5,
          minimize: true,
          &cost_fn
        )

        best_raw, _ = pso.optimize(iterations, &cost_fn)
        normalize_weights(best_raw)
      end
    end
  end
end
