# Stochasta

**Stochasta** is a fast, lightweight, and dependency-free Crystal shard providing modular implementations of stochastic, optimization, and unsupervised learning algorithms.

## What Stochasta Offers

Stochasta brings powerful stochastic models, nature-inspired search metaheuristics, and unsupervised data science algorithms to the Crystal ecosystem. It offers:

1. **Metaheuristic Global Optimization**: Find global extrema for complex continuous or discrete functions where traditional mathematical optimization fails, using Genetic Algorithms, PSO, Differential Evolution, Artificial Bee Colony, and Simulated Annealing.
2. **Combinatorial Optimization**: Solve routing and sequence-based discrete problems like the Traveling Salesperson Problem (TSP) using Ant Colony Optimization.
3. **Advanced Clustering Suite**: Group data structures using diverse paradigms—centroid partition (K-Means), density reachability (DBSCAN, Mean Shift), representative cluster medoids (K-Medoids), or bottom-up merge trees (Hierarchical Agglomerative Clustering).
4. **Probabilistic & Bayesian Reasoning**: Evaluate Bayes' theorem posterior probability updates and classify multi-class categorical texts using Laplace-smoothed Naive Bayes.
5. **Dimensionality Reduction**: Project complex multi-dimensional datasets to lower dimensions (e.g. for visualization or preprocessing) via Principal Component Analysis (PCA) using a built-in Jacobi eigenvalue decomposition.
6. **Association Mining**: Discover hidden relationships in transactional/shopping databases with the Apriori association rule mining algorithm.
7. **Stochastic Simulations**: Run Monte Carlo sample simulations and numerical multi-dimensional integration.

## Features

- **Genetic Algorithms** (`Stochasta::Genetic`): Flexible evolutionary engine with customizable selection (tournament), crossover, mutation, and support for elitism.
- **Monte Carlo Simulation** (`Stochasta::MonteCarlo`): Stochastic simulation trials, 1D and multi-dimensional numerical integration.
- **Particle Swarm Optimization (PSO)** (`Stochasta::PSO`): Continuous optimization in bounded spaces.
- **Bayesian Statistics** (`Stochasta::Bayes`): Categorical Naive Bayes classifiers with Laplace smoothing and Bayesian updating calculators.
- **K-Means Clustering** (`Stochasta::KMeans`): Standard iterative K-Means clustering algorithm.
- **Gaussian Mixture Models (GMM)** (`Stochasta::GMM`): Soft probability-based clustering trained via Expectation-Maximization.
- **Hierarchical Clustering** (`Stochasta::Hierarchical`): Agglomerative clustering supporting Single, Complete, and Average linkages.
- **K-Medoids (PAM)** (`Stochasta::KMedoids`): Partitioning Around Medoids using actual dataset points as centroids.
- **Mean Shift** (`Stochasta::MeanShift`): Density-based clustering to automatically discover cluster centers.
- **DBSCAN Clustering** (`Stochasta::DBSCAN`): Density-based spatial clustering for arbitrary shapes and noise extraction.
- **Principal Component Analysis (PCA)** (`Stochasta::PCA`): Unsupervised dimensionality reduction powered by a pure Crystal symmetric Jacobi eigenvalue solver.
- **Apriori Algorithm** (`Stochasta::Apriori`): Association rule mining and frequent itemsets locator.
- **Differential Evolution (DE)** (`Stochasta::DifferentialEvolution`): Global vector population optimizer for continuous spaces.
- **Ant Colony Optimization (ACO)** (`Stochasta::AntColony`): Combinatorial Traveling Salesperson Problem (TSP) solver.
- **Artificial Bee Colony (ABC)** (`Stochasta::ArtificialBeeColony`): Swarm-based global optimizer modeled after honey bee foraging.
- **Simulated Annealing** (`Stochasta::SimulatedAnnealing`): Probabilistic optimization technique for finding global extrema.

## Installation

1. Add the dependency to your `shard.yml`:

   ```yaml
   dependencies:
     stochasta:
       github: eltony81/stochasta
   ```

2. Run `shards install`

## Usage Examples

Complete, well-commented execution files are available in the [examples/](file:///home/tony/Projects/stochasta/examples) directory:

- [genetic_example.cr](file:///home/tony/Projects/stochasta/examples/genetic_example.cr): Solve a OneMax bitstring optimization problem using Genetic Algorithms.
- [monte_carlo_example.cr](file:///home/tony/Projects/stochasta/examples/monte_carlo_example.cr): Estimate Pi and run 1D/2D numerical integrals.
- [pso_example.cr](file:///home/tony/Projects/stochasta/examples/pso_example.cr): Find the global minimum of the Rosenbrock function.
- [de_example.cr](file:///home/tony/Projects/stochasta/examples/de_example.cr): Optimize continuous dimensions using Differential Evolution.
- [aco_example.cr](file:///home/tony/Projects/stochasta/examples/aco_example.cr): Solve the Traveling Salesperson Problem (TSP) using Ant Colony System.
- [abc_example.cr](file:///home/tony/Projects/stochasta/examples/abc_example.cr): Find global minimums using Artificial Bee Colony.
- [bayes_example.cr](file:///home/tony/Projects/stochasta/examples/bayes_example.cr): Run Bayesian inference and classify spam/ham messages.
- [clustering_example.cr](file:///home/tony/Projects/stochasta/examples/clustering_example.cr): Combine PCA dimension reduction, K-Means partitioning, and DBSCAN noise detection.
- [gmm_example.cr](file:///home/tony/Projects/stochasta/examples/gmm_example.cr): Run soft probabilistic clustering with Gaussian Mixture Models.
- [hierarchical_example.cr](file:///home/tony/Projects/stochasta/examples/hierarchical_example.cr): Cluster data using bottom-up Hierarchical Agglomerative Clustering.
- [kmedoids_example.cr](file:///home/tony/Projects/stochasta/examples/kmedoids_example.cr): Partition data around medoids chosen from actual points.
- [mean_shift_example.cr](file:///home/tony/Projects/stochasta/examples/mean_shift_example.cr): Run density-based cluster centroid discovery.
- [apriori_example.cr](file:///home/tony/Projects/stochasta/examples/apriori_example.cr): Mine shopping transaction databases for association rules (Support, Confidence, Lift).
- [annealing_example.cr](file:///home/tony/Projects/stochasta/examples/annealing_example.cr): Probabilistic optimization on continuous mathematical functions.

### Quick Sample: Naive Bayes Classification

```crystal
require "stochasta"

# Initialize a Naive Bayes classifier
classifier = Stochasta::Bayes::NaiveClassifier(String, String).new

# Train the model
classifier.train(["buy", "cheap", "deal"], "spam")
classifier.train(["meeting", "project", "deadline"], "ham")

# Predict on new unseen features
label = classifier.predict(["cheap", "deadline"])
puts "Classified as: #{label}"
```

## Running Tests

Stochasta has a comprehensive spec suite. To run the tests, execute:

```bash
crystal spec
```

## Contributors

- [tony](https://github.com/eltony81) - creator and maintainer
