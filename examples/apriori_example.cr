# Apriori Association Rule Mining Example
#
# This example demonstrates extracting frequent itemsets and association rules
# from a transaction database (e.g. supermarket purchase records).

require "../src/stochasta"

# Define a transactional dataset (each transaction is a Set of items)
transactions = [
  Set{"milk", "bread", "butter"},       # Transaction 1
  Set{"bread", "butter"},              # Transaction 2
  Set{"milk", "bread"},                # Transaction 3
  Set{"milk", "butter"},               # Transaction 4
  Set{"bread", "butter", "diaper", "beer"} # Transaction 5
]

puts "Transactions:"
transactions.each_with_index do |t, idx|
  puts "  Tx #{idx + 1}: #{t.to_a.join(", ")}"
end

# Instantiate the Apriori algorithm with:
# - min_support: 0.4 (itemsets must appear in at least 40% of transactions)
# - min_confidence: 0.6 (rules must have at least 60% confidence)
apriori = Stochasta::Apriori(String).new(min_support: 0.4, min_confidence: 0.6)

# Execute the algorithm
frequent_sets, rules = apriori.run(transactions)

puts "\n--- Frequent Itemsets (Support >= 0.4) ---"
frequent_sets.each do |itemset, support|
  puts "  Itemset: {#{itemset.to_a.join(", ")}} | Support: #{(support * 100).round(1)}%"
end

puts "\n--- Generated Association Rules (Confidence >= 0.6) ---"
# Support: overall popularity of the rule
# Confidence: likelihood that item B is purchased when item A is purchased
# Lift: ratio of observed support to expected support if A and B were independent (> 1 means positive correlation)
rules.each_with_index do |rule, idx|
  puts "  Rule ##{idx + 1}: {#{rule.antecedent.to_a.join(", ")}} => {#{rule.consequent.to_a.join(", ")}}"
  puts "    Support:    #{(rule.support * 100).round(1)}%"
  puts "    Confidence: #{(rule.confidence * 100).round(1)}%"
  puts "    Lift:       #{rule.lift.round(3)}"
end
