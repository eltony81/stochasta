module Stochasta
  module Bayes
    # Simple Bayesian parameter estimator / calculator
    module Estimator
      # Calculates P(A|B) = P(B|A) * P(A) / P(B)
      def self.posterior(prior : Float64, likelihood : Float64, evidence : Float64) : Float64
        raise ArgumentError.new("Probabilities must be between 0 and 1") unless (0.0..1.0).includes?(prior) && (0.0..1.0).includes?(likelihood) && evidence > 0.0
        (likelihood * prior) / evidence
      end

      # Calculates posterior using P(B) = P(B|A)*P(A) + P(B|~A)*P(~A)
      def self.posterior_with_marginal(prior : Float64, likelihood_true : Float64, likelihood_false : Float64) : Float64
        evidence = likelihood_true * prior + likelihood_false * (1.0 - prior)
        posterior(prior, likelihood_true, evidence)
      end
    end

    # Multinomial/Categorical Naive Bayes Classifier
    # C is the Class type, F is the Feature type
    class NaiveClassifier(C, F)
      property class_counts : Hash(C, Int32)
      property feature_counts : Hash(C, Hash(F, Int32))
      property total_samples : Int32
      property vocabulary : Set(F)

      def initialize
        @class_counts = Hash(C, Int32).new(0)
        @feature_counts = Hash(C, Hash(F, Int32)).new { |h, k| h[k] = Hash(F, Int32).new(0) }
        @total_samples = 0
        @vocabulary = Set(F).new
      end

      # Trains the classifier on a single sample of features with its label
      def train(features : Enumerable(F), label : C)
        @class_counts[label] += 1
        @total_samples += 1
        
        features.each do |f|
          @feature_counts[label][f] += 1
          @vocabulary << f
        end
      end

      # Predicts the most likely class for the given features
      # Uses log probabilities to prevent underflow, and Laplace smoothing (+1)
      def predict(features : Enumerable(F)) : C
        raise Exception.new("Classifier has not been trained yet") if @total_samples == 0

        best_class = nil
        best_log_prob = -Float64::INFINITY

        @class_counts.each_key do |label|
          # Prior: P(C)
          prior = @class_counts[label].to_f / @total_samples
          log_prob = Math.log(prior)

          # Sum features count for this class (for denominator of likelihood)
          # We add vocabulary size for Laplace smoothing
          total_features_in_class = @feature_counts[label].values.sum
          denom = total_features_in_class + @vocabulary.size

          features.each do |f|
            next unless @vocabulary.includes?(f) # Skip unseen features in vocabulary if any

            # Likelihood: P(f|C) with Laplace smoothing
            count = @feature_counts[label][f]
            prob = (count + 1).to_f / denom
            log_prob += Math.log(prob)
          end

          if best_class.nil? || log_prob > best_log_prob
            best_log_prob = log_prob
            best_class = label
          end
        end

        best_class.not_nil!
      end

      # Computes probability distribution over all classes for the given features
      def predict_probabilities(features : Enumerable(F)) : Hash(C, Float64)
        raise Exception.new("Classifier has not been trained yet") if @total_samples == 0

        scores = Hash(C, Float64).new

        @class_counts.each_key do |label|
          prior = @class_counts[label].to_f / @total_samples
          log_prob = Math.log(prior)
          total_features_in_class = @feature_counts[label].values.sum
          denom = total_features_in_class + @vocabulary.size

          features.each do |f|
            next unless @vocabulary.includes?(f)
            count = @feature_counts[label][f]
            prob = (count + 1).to_f / denom
            log_prob += Math.log(prob)
          end
          scores[label] = log_prob
        end

        # Normalize log probabilities safely
        max_log = scores.values.max
        exp_scores = scores.transform_values { |v| Math.exp(v - max_log) }
        sum_exp = exp_scores.values.sum

        exp_scores.transform_values { |v| v / sum_exp }
      end
    end
  end
end
