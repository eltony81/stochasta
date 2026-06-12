module Stochasta
  module Portfolio
    module Risk
      # Returns the Z-score for a given confidence level (e.g., 0.95 -> 1.64485, 0.99 -> 2.32635)
      # Uses inverse error function approximation or standard values for common metrics
      def self.z_score(confidence : Float64) : Float64
        # Common values for performance
        case confidence
        when 0.90
          1.28155
        when 0.95
          1.64485
        when 0.975
          1.95996
        when 0.99
          2.32635
        when 0.999
          3.09023
        else
          # Rational approximation for inverse CDF of standard normal
          # Winitzki approximation or simple inverse erf
          p = 1.0 - confidence
          t = Math.sqrt(-2.0 * Math.log(p))
          # Coefficients for standard approximation
          c0 = 2.515517
          c1 = 0.802853
          c2 = 0.010328
          d1 = 1.432788
          d2 = 0.189269
          d3 = 0.001308
          t - (c0 + c1*t + c2*t*t) / (1.0 + d1*t + d2*t*t + d3*t*t*t)
        end
      end

      # Computes the parametric Value at Risk (VaR) under normal distribution assumption
      # Returns the loss as a positive value (fraction of portfolio value)
      def self.parametric_var(mean_return : Float64, std_dev : Float64, confidence : Float64 = 0.95) : Float64
        z = z_score(confidence)
        # Loss is positive
        -(mean_return - z * std_dev)
      end

      # Computes the parametric Conditional Value at Risk (CVaR) under normal distribution
      # Returns the loss as a positive value
      def self.parametric_cvar(mean_return : Float64, std_dev : Float64, confidence : Float64 = 0.95) : Float64
        z = z_score(confidence)
        # Normal PDF at Z-score
        pdf_z = (1.0 / Math.sqrt(2.0 * Math::PI)) * Math.exp(-0.5 * z * z)
        alpha = 1.0 - confidence
        
        # Loss is positive
        -(mean_return - std_dev * (pdf_z / alpha))
      end

      # Computes Monte Carlo Value at Risk (VaR)
      # Returns the loss as a positive value
      def self.monte_carlo_var(simulated_returns : Array(Float64), confidence : Float64 = 0.95) : Float64
        sorted = simulated_returns.sort
        n = sorted.size
        # Index corresponding to the (1 - confidence) percentile of losses
        idx = ((1.0 - confidence) * n).to_i
        idx = 0 if idx < 0
        idx = n - 1 if idx >= n
        
        -sorted[idx]
      end

      # Computes Monte Carlo Conditional Value at Risk (CVaR)
      # Returns the loss as a positive value
      def self.monte_carlo_cvar(simulated_returns : Array(Float64), confidence : Float64 = 0.95) : Float64
        sorted = simulated_returns.sort
        n = sorted.size
        idx = ((1.0 - confidence) * n).to_i
        idx = 1 if idx < 1
        
        # Average of all returns in the tail (below the VaR threshold)
        tail = sorted[0...idx]
        -tail.sum / tail.size
      end
    end
  end
end
