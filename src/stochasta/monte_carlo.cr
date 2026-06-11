module Stochasta
  module MonteCarlo
    # Runs a stochastic simulation N times and returns the results.
    # Yields the run index to the block, which should return the outcome.
    def self.simulate(n : Int32) : Array(T) forall T
      results = Array(T).new(n)
      n.times do |i|
        results << yield(i)
      end
      results
    end

    # Performs 1D Monte Carlo integration of a function f over [a, b]
    def self.integrate(a : Float64, b : Float64, samples : Int32 = 100_000) : Float64
      sum = 0.0
      samples.times do
        x = a + (b - a) * Random.rand
        sum += yield(x)
      end
      (b - a) * sum / samples
    end

    # Performs Multi-dimensional Monte Carlo integration of a function f
    # over the bounding boxes defined by bounds (array of {min, max} tuples)
    def self.integrate_multi(bounds : Array(Tuple(Float64, Float64)), samples : Int32 = 100_000) : Float64
      v = 1.0
      bounds.each do |min, max|
        v *= (max - min)
      end

      sum = 0.0
      samples.times do
        point = bounds.map { |min, max| min + (max - min) * Random.rand }
        sum += yield(point)
      end

      v * sum / samples
    end
  end
end
