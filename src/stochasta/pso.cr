module Stochasta
  module PSO
    class Particle
      property position : Array(Float64)
      property velocity : Array(Float64)
      property best_position : Array(Float64)
      property fitness : Float64
      property best_fitness : Float64

      def initialize(@position : Array(Float64), @velocity : Array(Float64), @fitness : Float64)
        @best_position = @position.dup
        @best_fitness = @fitness
      end

      # Updates the best position if the current fitness is better (lower is better for minimization)
      def update_best(minimize : Bool = true)
        if minimize
          if @fitness < @best_fitness
            @best_fitness = @fitness
            @best_position = @position.dup
          end
        else
          if @fitness > @best_fitness
            @best_fitness = @fitness
            @best_position = @position.dup
          end
        end
      end
    end

    class Optimizer
      property particles : Array(Particle)
      property global_best_position : Array(Float64)
      property global_best_fitness : Float64
      property bounds : Array(Tuple(Float64, Float64))
      property w : Float64  # Inertia weight
      property c1 : Float64 # Cognitive coefficient
      property c2 : Float64 # Social coefficient
      property minimize : Bool

      def initialize(
        swarm_size : Int32,
        @bounds : Array(Tuple(Float64, Float64)),
        @w : Float64 = 0.5,
        @c1 : Float64 = 1.5,
        @c2 : Float64 = 1.5,
        @minimize : Bool = true,
        &fitness_fn : Array(Float64) -> Float64
      )
        dimensions = @bounds.size
        @particles = Array(Particle).new(swarm_size)
        @global_best_fitness = @minimize ? Float64::MAX : Float64::MIN
        @global_best_position = Array(Float64).new(dimensions, 0.0)

        swarm_size.times do
          # Random initial position within bounds
          pos = @bounds.map { |min, max| min + (max - min) * Random.rand }
          # Random initial velocity
          vel = @bounds.map { |min, max| ((max - min) * 0.1) * (Random.rand * 2.0 - 1.0) }
          
          fit = fitness_fn.call(pos)
          particle = Particle.new(pos, vel, fit)
          @particles << particle

          # Check global best
          if @minimize ? (fit < @global_best_fitness) : (fit > @global_best_fitness)
            @global_best_fitness = fit
            @global_best_position = pos.dup
          end
        end
      end

      # Performs one iteration of the optimizer
      def step(&fitness_fn : Array(Float64) -> Float64)
        dimensions = @bounds.size

        @particles.each do |p|
          new_vel = Array(Float64).new(dimensions)
          new_pos = Array(Float64).new(dimensions)

          dimensions.times do |d|
            r1 = Random.rand
            r2 = Random.rand

            cognitive = @c1 * r1 * (p.best_position[d] - p.position[d])
            social = @c2 * r2 * (@global_best_position[d] - p.position[d])
            
            # Update velocity
            v_new = @w * p.velocity[d] + cognitive + social
            new_vel << v_new

            # Update position
            p_new = p.position[d] + v_new
            
            # Clamp to bounds
            min, max = @bounds[d]
            if p_new < min
              p_new = min
              v_new = 0.0 # Stop velocity if boundary hit
            elsif p_new > max
              p_new = max
              v_new = 0.0
            end

            new_pos << p_new
          end

          p.velocity = new_vel
          p.position = new_pos
          p.fitness = fitness_fn.call(p.position)
          p.update_best(@minimize)

          # Update global best
          if @minimize ? (p.fitness < @global_best_fitness) : (p.fitness > @global_best_fitness)
            @global_best_fitness = p.fitness
            @global_best_position = p.position.dup
          end
        end
      end

      # Optimizes for `iterations` steps
      def optimize(iterations : Int32, &fitness_fn : Array(Float64) -> Float64) : Tuple(Array(Float64), Float64)
        iterations.times do
          step(&fitness_fn)
        end
        {@global_best_position, @global_best_fitness}
      end
    end
  end
end
