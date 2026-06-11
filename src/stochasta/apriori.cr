module Stochasta
  class Apriori(T)
    struct Rule(T)
      property antecedent : Set(T)
      property consequent : Set(T)
      property support : Float64
      property confidence : Float64
      property lift : Float64

      def initialize(@antecedent : Set(T), @consequent : Set(T), @support : Float64, @confidence : Float64, @lift : Float64)
      end
    end

    property min_support : Float64
    property min_confidence : Float64

    def initialize(@min_support : Float64, @min_confidence : Float64 = 0.5)
    end

    # Runs the Apriori algorithm on a set of transactions
    # Returns a tuple of {frequent_itemsets, association_rules}
    def run(transactions : Array(Set(T))) : Tuple(Hash(Set(T), Float64), Array(Rule(T)))
      n_transactions = transactions.size.to_f
      return {Hash(Set(T), Float64).new, [] of Rule(T)} if n_transactions == 0

      # 1. Generate 1-itemsets
      item_counts = Hash(T, Int32).new(0)
      transactions.each do |t|
        t.each { |item| item_counts[item] += 1 }
      end

      # Filter by min_support
      current_frequent = Hash(Set(T), Float64).new
      item_counts.each do |item, count|
        support = count.to_f / n_transactions
        if support >= @min_support
          current_frequent[Set{item}] = support
        end
      end

      all_frequent = current_frequent.dup
      k = 2

      while !current_frequent.empty?
        # Generate candidates of size k from frequent itemsets of size k-1
        candidates = generate_candidates(current_frequent.keys, k)
        break if candidates.empty?

        # Count candidate supports
        candidate_counts = Hash(Set(T), Int32).new(0)
        transactions.each do |t|
          candidates.each do |cand|
            if cand.subset_of?(t)
              candidate_counts[cand] += 1
            end
          end
        end

        # Filter candidates by min_support
        current_frequent = Hash(Set(T), Float64).new
        candidate_counts.each do |cand, count|
          support = count.to_f / n_transactions
          if support >= @min_support
            current_frequent[cand] = support
          end
        end

        all_frequent.merge!(current_frequent)
        k += 1
      end

      # 2. Generate Association Rules
      rules = generate_rules(all_frequent)

      {all_frequent, rules}
    end

    # Generates candidate k-itemsets by joining frequent (k-1)-itemsets
    private def generate_candidates(frequent_itemsets : Array(Set(T)), k : Int32) : Set(Set(T))
      candidates = Set(Set(T)).new
      size = frequent_itemsets.size

      size.times do |i|
        ((i + 1)...size).each do |j|
          set1 = frequent_itemsets[i]
          set2 = frequent_itemsets[j]

          # Union the two itemsets
          union_set = set1 | set2
          if union_set.size == k
            # Optional pruning step: check if all subsets of size k-1 are frequent
            if prune_candidate?(union_set, frequent_itemsets)
              candidates << union_set
            end
          end
        end
      end

      candidates
    end

    private def prune_candidate?(candidate : Set(T), frequent_itemsets : Array(Set(T))) : Bool
      # Every subset of size k-1 must be in frequent_itemsets
      frequent_set = Set(Set(T)).new(frequent_itemsets)
      candidate.each do |item|
        subset = candidate.dup
        subset.delete(item)
        return false unless frequent_set.includes?(subset)
      end
      true
    end

    # Generates association rules from frequent itemsets
    private def generate_rules(frequent_itemsets : Hash(Set(T), Float64)) : Array(Rule(T))
      rules = [] of Rule(T)

      frequent_itemsets.each_key do |itemset|
        next if itemset.size < 2

        # Generate all subsets of the itemset to form antecedents
        subsets = get_all_proper_subsets(itemset)
        subsets.each do |antecedent|
          consequent = itemset - antecedent
          next if consequent.empty?

          support_both = frequent_itemsets[itemset]
          support_ant = frequent_itemsets[antecedent]
          support_cons = frequent_itemsets[consequent]

          confidence = support_both / support_ant
          if confidence >= @min_confidence
            lift = confidence / support_cons
            rules << Rule(T).new(antecedent, consequent, support_both, confidence, lift)
          end
        end
      end

      rules
    end

    private def get_all_proper_subsets(set : Set(T)) : Array(Set(T))
      arr = set.to_a
      subsets = [] of Set(T)

      # 1 to (2^n - 2) to get all proper, non-empty subsets
      limit = (1 << arr.size) - 1
      (1...limit).each do |i|
        subset = Set(T).new
        arr.size.times do |j|
          if (i & (1 << j)) != 0
            subset << arr[j]
          end
        end
        subsets << subset
      end

      subsets
    end
  end
end
