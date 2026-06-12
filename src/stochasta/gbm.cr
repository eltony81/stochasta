module Stochasta
  module Portfolio
    module GBM
      # Box-Muller transform to generate a standard normal random variable N(0, 1)
      def self.random_normal : Float64
        u1 = Random.rand
        # Ensure u1 is not zero to avoid log(0)
        u1 = 1e-12 if u1 < 1e-12
        u2 = Random.rand
        Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
      end

      # Simulates a single price path using Geometric Brownian Motion
      # Params:
      # - s0: initial price
      # - mu: expected annual drift
      # - sigma: annual volatility
      # - t: total time in years (e.g. 1.0)
      # - steps: number of time steps
      # Returns an Array(Float64) representing the price path of size steps + 1
      def self.simulate_path(s0 : Float64, mu : Float64, sigma : Float64, t : Float64 = 1.0, steps : Int32 = 252) : Array(Float64)
        dt = t / steps
        drift = (mu - 0.5 * sigma * sigma) * dt
        vol = sigma * Math.sqrt(dt)

        path = Array(Float64).new(steps + 1)
        path << s0

        current_price = s0
        steps.times do
          z = random_normal
          current_price *= Math.exp(drift + vol * z)
          path << current_price
        end

        path
      end

      # Simulates multiple price paths
      # Returns an Array(Array(Float64)) containing n_paths arrays
      def self.simulate_paths(s0 : Float64, mu : Float64, sigma : Float64, t : Float64 = 1.0, steps : Int32 = 252, n_paths : Int32 = 1000) : Array(Array(Float64))
        paths = Array(Array(Float64)).new(n_paths)
        n_paths.times do
          paths << simulate_path(s0, mu, sigma, t, steps)
        end
        paths
      end
    end
  end
end
