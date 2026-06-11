# Bayesian Inference & Classifier Example
#
# This example demonstrates:
# 1. Simple Bayesian updating (posterior probabilities given prior, likelihood, and evidence)
# 2. Building, training, and predicting with a Naive Bayes Categorical Classifier.

require "../src/stochasta"

# --- 1. Bayesian Probability Updates ---
puts "--- 1. Bayesian Updating ---"

# Let's say a medical test is 99% accurate (sensitivity/likelihood = 0.99)
# The false positive rate is 5% (likelihood of positive given healthy = 0.05)
# The disease prevalence is 1% (prior = 0.01)
# What is the probability that a person has the disease given they test positive?
prior = 0.01
likelihood_true = 0.99
likelihood_false = 0.05

posterior = Stochasta::Bayes::Estimator.posterior_with_marginal(
  prior: prior,
  likelihood_true: likelihood_true,
  likelihood_false: likelihood_false
)

puts "Prior probability: #{prior * 100}%"
puts "Posterior probability after positive test: #{(posterior * 100).round(2)}%"


# --- 2. Naive Bayes Classifier ---
puts "\n--- 2. Naive Bayes Classifier ---"

# Instantiate a Classifier where the Class label is String and Features are Strings
classifier = Stochasta::Bayes::NaiveClassifier(String, String).new

# Train the model with training data
# Spam examples
classifier.train(["buy", "cheap", "viagra", "now"], "spam")
classifier.train(["earn", "money", "fast", "cash"], "spam")
classifier.train(["viagra", "cheap", "click", "here"], "spam")

# Ham (Normal) examples
classifier.train(["hello", "how", "are", "you"], "ham")
classifier.train(["meeting", "at", "noon", "today"], "ham")
classifier.train(["project", "updates", "meeting", "status"], "ham")

# Let's perform predictions on new unseen emails
email_1 = ["cheap", "money", "now"]
email_2 = ["meeting", "status", "tomorrow"]

prediction_1 = classifier.predict(email_1)
prediction_2 = classifier.predict(email_2)

puts "Email 1 #{email_1} Classified as: #{prediction_1}"
puts "Email 2 #{email_2} Classified as: #{prediction_2}"

# Display class probabilities for Email 1
probabilities = classifier.predict_probabilities(email_1)
probabilities.each do |label, prob|
  puts "  P(#{label} | Email 1) = #{(prob * 100).round(2)}%"
end
