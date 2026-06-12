require "./spec_helper"

describe Stochasta do
  it "Genetic Algorithm solves OneMax" do
    # OneMax problem: genome is Array(Int32) of 0s and 1s, maximize sum
    fitness_fn = ->(g : Array(Int32)) { g.sum.to_f }
    crossover_fn = ->(p1 : Array(Int32), p2 : Array(Int32)) {
      # 1-point crossover
      pt = p1.size // 2
      p1[0...pt] + p2[pt...]
    }
    mutate_fn = ->(g : Array(Int32)) {
      # Flip a random bit
      idx = Random.rand(g.size)
      g = g.dup
      g[idx] = g[idx] == 0 ? 1 : 0
      g
    }

    initial_population = Array(Array(Int32)).new(20) do
      Array(Int32).new(10) { Random.rand(2) }
    end

    engine = Stochasta::Genetic::Engine(Array(Int32)).new(
      population_size: 20,
      mutation_rate: 0.1,
      crossover_rate: 0.8,
      fitness_fn: fitness_fn,
      crossover_fn: crossover_fn,
      mutate_fn: mutate_fn,
      initial_genomes: initial_population
    )

    best = engine.evolve(100)
    best.fitness.should be_close(10.0, 3.0) # Should get close to 10 ones
  end

  it "Monte Carlo integrates x^2" do
    result = Stochasta::MonteCarlo.integrate(0.0, 1.0, 10_000) do |x|
      x * x
    end
    result.should be_close(0.3333, 0.05)
  end

  it "Monte Carlo multi-dimensional integration works" do
    # Integrate f(x, y) = x * y over [0, 1] x [0, 1] -> exact is 0.25
    result = Stochasta::MonteCarlo.integrate_multi([{0.0, 1.0}, {0.0, 1.0}], 10_000) do |point|
      point[0] * point[1]
    end
    result.should be_close(0.25, 0.05)
  end

  it "PSO minimizes x^2 + y^2" do
    fitness_fn = ->(pos : Array(Float64)) { pos[0]*pos[0] + pos[1]*pos[1] }
    optimizer = Stochasta::PSO::Optimizer.new(
      swarm_size: 30,
      bounds: [{-10.0, 10.0}, {-10.0, 10.0}],
      w: 0.5,
      c1: 1.5,
      c2: 1.5,
      minimize: true,
      &fitness_fn
    )

    best_pos, best_fit = optimizer.optimize(50, &fitness_fn)
    best_fit.should be_close(0.0, 0.05)
    best_pos[0].should be_close(0.0, 0.2)
    best_pos[1].should be_close(0.0, 0.2)
  end

  it "Bayes Estimator computes correct posterior" do
    # P(Cancer) = 0.01, P(Positive|Cancer) = 0.9, P(Positive|No Cancer) = 0.09
    # P(Cancer|Positive) = (0.9 * 0.01) / (0.9*0.01 + 0.09*0.99) = 0.009 / (0.009 + 0.0891) = 0.092
    post = Stochasta::Bayes::Estimator.posterior_with_marginal(0.01, 0.9, 0.09)
    post.should be_close(0.092, 0.001)
  end

  it "Naive Bayes Classifier categorizes text" do
    classifier = Stochasta::Bayes::NaiveClassifier(String, String).new
    
    # Train
    classifier.train(["cheap", "buy", "click"], "spam")
    classifier.train(["money", "cash", "cheap"], "spam")
    classifier.train(["hello", "friend", "meeting"], "ham")
    classifier.train(["meeting", "project", "hello"], "ham")

    # Predict
    classifier.predict(["cheap", "money"]).should eq("spam")
    classifier.predict(["hello", "project"]).should eq("ham")

    probs = classifier.predict_probabilities(["cheap"])
    (probs["spam"] > probs["ham"]).should be_true
  end

  it "KMeans clusters points" do
    # 2 obvious clusters
    data = [
      [1.0, 1.0], [1.2, 0.8], [0.8, 1.2],
      [10.0, 10.0], [9.8, 10.2], [10.2, 9.8]
    ]

    model = Stochasta::KMeans::Model.new(k: 2).fit(data)
    
    # Check that points in the same cluster get the same label
    cluster_a = model.predict([1.0, 1.0])
    cluster_b = model.predict([10.0, 10.0])
    cluster_a.should_not eq(cluster_b)

    model.predict([1.1, 0.9]).should eq(cluster_a)
    model.predict([9.9, 10.1]).should eq(cluster_b)
  end

  it "DBSCAN clusters core points and marks noise" do
    # 2 clusters plus a noise point
    data = [
      [1.0, 1.0], [1.1, 1.0], [1.0, 1.1],
      [10.0, 10.0], [10.1, 10.0], [10.0, 10.1],
      [50.0, 50.0] # Noise
    ]

    model = Stochasta::DBSCAN::Model.new(eps: 2.0, min_samples: 2).fit(data)
    
    model.labels[0].should_not eq(-1)
    model.labels[3].should_not eq(-1)
    model.labels[0].should_not eq(model.labels[3])
    model.labels[6].should eq(-1) # Noise
  end

  it "PCA reduces dimension and projects correctly" do
    # Correlated data: y = x
    data = [
      [1.0, 1.0],
      [2.0, 2.0],
      [3.0, 3.0],
      [4.0, 4.0]
    ]

    pca = Stochasta::PCA.new(n_components: 1).fit(data)
    pca.explained_variance_ratio[0].should be_close(1.0, 1e-5) # 100% variance along y=x

    transformed = pca.transform(data)
    transformed.size.should eq(4)
    transformed.first.size.should eq(1)
  end

  it "Apriori finds association rules" do
    transactions = [
      Set{"milk", "bread", "butter"},
      Set{"bread", "butter"},
      Set{"milk", "bread"},
      Set{"milk", "butter"},
      Set{"bread", "butter", "diaper", "beer"}
    ]

    apriori = Stochasta::Apriori(String).new(min_support: 0.4, min_confidence: 0.6)
    frequent_sets, rules = apriori.run(transactions)

    # Bread and Butter are very frequent together
    frequent_sets.has_key?(Set{"bread", "butter"}).should be_true
    rules.any? { |r| r.antecedent == Set{"bread"} && r.consequent == Set{"butter"} }.should be_true
  end

  it "Simulated Annealing minimizes x^2" do
    cost_fn = ->(x : Float64) { x * x }
    neighbor_fn = ->(x : Float64) { x + (Random.rand * 2.0 - 1.0) }

    best_state, best_cost = Stochasta::SimulatedAnnealing.optimize(
      initial_state: 10.0,
      initial_temp: 100.0,
      cooling_rate: 0.95,
      cost_fn: cost_fn,
      min_temp: 0.001
    ) do |current|
      neighbor_fn.call(current)
    end

    best_cost.should be_close(0.0, 0.1)
    best_state.should be_close(0.0, 0.3)
  end

  it "GMM clusters data points probabilistically" do
    data = [
      [1.0, 1.0], [1.1, 0.9], [0.9, 1.1],
      [10.0, 10.0], [9.9, 10.1], [10.1, 9.9]
    ]

    model = Stochasta::GMM::Model.new(k: 2).fit(data)

    p1 = model.predict([1.0, 1.0])
    p2 = model.predict([10.0, 10.0])
    p1.should_not eq(p2)

    probs = model.predict_probabilities([1.0, 1.0])
    (probs[p1] > 0.9).should be_true
  end

  it "Hierarchical Agglomerative Clustering groups points" do
    data = [
      [1.0, 1.0], [1.2, 0.8],
      [10.0, 10.0], [9.8, 10.2]
    ]

    model = Stochasta::Hierarchical::Model.new(k: 2, linkage: Stochasta::Hierarchical::Linkage::Average).fit(data)
    
    model.labels[0].should eq(model.labels[1])
    model.labels[2].should eq(model.labels[3])
    model.labels[0].should_not eq(model.labels[2])
  end

  it "KMedoids clusters using actual dataset points as centers" do
    data = [
      [1.0, 1.0], [1.1, 0.9],
      [10.0, 10.0], [9.9, 10.1]
    ]

    model = Stochasta::KMedoids::Model.new(k: 2).fit(data)
    
    c1 = model.predict([1.0, 1.0])
    c2 = model.predict([10.0, 10.0])
    c1.should_not eq(c2)

    # Medoids must be exact points in the dataset
    data.should contain(model.medoids[0])
    data.should contain(model.medoids[1])
  end

  it "Mean Shift finds cluster centroids dynamically" do
    data = [
      [1.0, 1.0], [1.1, 0.9], [0.9, 1.1],
      [10.0, 10.0], [9.9, 10.1], [10.1, 9.9]
    ]

    model = Stochasta::MeanShift::Model.new(bandwidth: 2.0).fit(data)

    # Should find exactly 2 clusters
    model.centroids.size.should eq(2)
    c1 = model.predict([1.0, 1.0])
    c2 = model.predict([10.0, 10.0])
    c1.should_not eq(c2)
  end

  it "Differential Evolution minimizes Sphere function" do
    sphere_fn = ->(pos : Array(Float64)) { pos[0]**2 + pos[1]**2 }
    optimizer = Stochasta::DifferentialEvolution::Optimizer.new(
      pop_size: 15,
      bounds: [{-5.0, 5.0}, {-5.0, 5.0}],
      f: 0.8,
      cr: 0.9,
      &sphere_fn
    )

    best_pos, best_cost = optimizer.optimize(40, &sphere_fn)
    best_cost.should be_close(0.0, 0.05)
    best_pos[0].should be_close(0.0, 0.2)
  end

  it "Ant Colony Optimization solves a simple TSP" do
    # 4 cities in a line (0 -> 1 -> 2 -> 3)
    # Distances are: 0-1: 10, 1-2: 10, 2-3: 10, 3-0: 30
    # The optimal cycle is 0 -> 1 -> 2 -> 3 -> 0 (length = 60)
    distance_matrix = [
      [0.0, 10.0, 20.0, 30.0],
      [10.0, 0.0, 10.0, 20.0],
      [20.0, 10.0, 0.0, 10.0],
      [30.0, 20.0, 10.0, 0.0]
    ]

    solver = Stochasta::AntColony::TSPSolver.new(
      distance_matrix: distance_matrix,
      n_ants: 10,
      alpha: 1.0,
      beta: 2.0,
      evaporation_rate: 0.1
    )

    best_tour, best_length = solver.solve(30)
    best_length.should be_close(60.0, 1.0)
  end

  it "Artificial Bee Colony minimizes Sphere function" do
    sphere_fn = ->(pos : Array(Float64)) { pos[0]**2 + pos[1]**2 }
    optimizer = Stochasta::ArtificialBeeColony::Optimizer.new(
      swarm_size: 16,
      bounds: [{-5.0, 5.0}, {-5.0, 5.0}],
      limit: 15,
      &sphere_fn
    )

    best_pos, best_cost = optimizer.optimize(40, &sphere_fn)
    best_cost.should be_close(0.0, 0.05)
    best_pos[0].should be_close(0.0, 0.2)
  end

  it "Portfolio Optimizer finds minimum variance and maximum Sharpe weights" do
    expected_returns = [0.10, 0.15, 0.12]
    covariance = [
      [0.04, 0.01, 0.02],
      [0.01, 0.09, 0.015],
      [0.02, 0.015, 0.05]
    ]

    opt = Stochasta::Portfolio::Optimizer.new(expected_returns, covariance)

    # 1. Min variance
    min_w = opt.min_variance_weights(iterations: 30)
    min_w.size.should eq(3)
    min_w.sum.should be_close(1.0, 1e-6)
    min_w.each { |w| (w >= 0.0).should be_true }

    # 2. Max Sharpe
    max_w = opt.max_sharpe_weights(risk_free_rate: 0.02, iterations: 30)
    max_w.size.should eq(3)
    max_w.sum.should be_close(1.0, 1e-6)
  end

  it "Black-Litterman adjusts returns based on prior and views" do
    prior_returns = [0.08, 0.12]
    covariance = [
      [0.04, 0.02],
      [0.02, 0.09]
    ]
    # View: Asset 1 return will be 15% (absolute)
    p_matrix = [[0.0, 1.0]]
    q_vector = [0.15]

    adjusted = Stochasta::Portfolio::BlackLitterman.estimate_returns(
      prior_returns: prior_returns,
      covariance: covariance,
      p_matrix: p_matrix,
      q_vector: q_vector,
      tau: 0.025
    )

    adjusted.size.should eq(2)
    # The return of the second asset should shift from 12% towards 15%
    (adjusted[1] > 0.12).should be_true
  end

  it "Portfolio Risk computes parametric and Monte Carlo VaR and CVaR" do
    mean = 0.08
    std = 0.15

    p_var = Stochasta::Portfolio::Risk.parametric_var(mean, std, 0.95)
    p_cvar = Stochasta::Portfolio::Risk.parametric_cvar(mean, std, 0.95)

    p_var.should be_close(0.1667, 0.05)
    p_cvar.should be_close(0.229, 0.05)

    sims = Array(Float64).new(1000)
    1000.times do
      sims << (mean + Stochasta::Portfolio::GBM.random_normal * std)
    end

    mc_var = Stochasta::Portfolio::Risk.monte_carlo_var(sims, 0.95)
    mc_cvar = Stochasta::Portfolio::Risk.monte_carlo_cvar(sims, 0.95)

    mc_var.should be_close(p_var, 0.05)
    (mc_cvar > mc_var).should be_true
  end

  it "Geometric Brownian Motion simulates paths correctly" do
    s0 = 100.0
    mu = 0.1
    sigma = 0.2
    
    path = Stochasta::Portfolio::GBM.simulate_path(s0, mu, sigma, t: 1.0, steps: 10)
    path.size.should eq(11)
    path.first.should eq(100.0)

    paths = Stochasta::Portfolio::GBM.simulate_paths(s0, mu, sigma, t: 1.0, steps: 10, n_paths: 10)
    paths.size.should eq(10)
    paths.first.first.should eq(100.0)
  end
end
