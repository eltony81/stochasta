# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Stochasta is a fast, lightweight, dependency-free Crystal shard implementing stochastic, optimization, and unsupervised learning algorithms (genetic algorithms, PSO, differential evolution, ACO, ABC, simulated annealing, clustering, PCA, Bayes, Apriori, and a quant-finance suite for portfolio optimization, Black-Litterman, VaR/CVaR, and GBM).

## Commands

```bash
shards install       # install dependencies (none currently required — pure Crystal, stdlib only)
crystal spec          # run the full spec suite
crystal spec spec/stochasta_spec.cr -e "Genetic Algorithm solves OneMax"  # run a single example by name
crystal build examples/pso_example.cr -o /tmp/pso_example && /tmp/pso_example  # run an example file
crystal tool format   # format source per .editorconfig conventions (2-space indent, LF, trim trailing whitespace)
```

There is no CI config, linter (no Ameba), or build step beyond `crystal build`/`crystal spec` in this repo.

## Architecture

- **One source file per algorithm** under `src/stochasta/`, each `require`d wholesale via the glob in `src/stochasta.cr` (`require "./stochasta/*"`). Adding a new algorithm means adding a new file here — there is no manual require list to update.
- **Namespacing**: every file opens `module Stochasta` then a nested algorithm-specific module (e.g. `module Genetic`, `module KMeans`, `module Portfolio`). Related algorithms that share a domain nest further inside a shared module — all quant-finance code (`portfolio_optimizer.cr`, `black_litterman.cr`, `portfolio_risk.cr`, `gbm.cr`) lives under `Stochasta::Portfolio::*` even though it's split across files.
- **Two API shapes** recur across the library:
  - **Stateful `Model`/`Engine`/`Optimizer` classes** for algorithms with fit/predict or iterative-improvement semantics (e.g. `Stochasta::KMeans::Model#fit`/`#predict`, `Stochasta::Genetic::Engine#evolve`, `Stochasta::PSO::Optimizer#optimize`). These hold mutable `property` state (centroids, population, swarm) and validate inputs with `raise ArgumentError.new(...)` at the start of public methods.
  - **Stateless module functions** for pure computations (e.g. `Stochasta::MonteCarlo.integrate`, `Stochasta::Bayes::Estimator.posterior_with_marginal`, `Stochasta::Portfolio::Risk.parametric_var`).
- **Objective/behavior functions are injected as typed Procs or blocks**, not subclassed — e.g. `fitness_fn : T -> Float64`, `crossover_fn : (T, T) -> T` on `Genetic::Engine`, or a `&block` yielded per candidate in `PSO::Optimizer`/`DifferentialEvolution::Optimizer`/`ArtificialBeeColony::Optimizer`. Follow this convention (proc/block parameters) rather than introducing strategy classes or inheritance when adding new optimizers.
- Generic algorithms are parameterized with Crystal generics (`Individual(T)`, `Engine(T)`, `Bayes::NaiveClassifier(K, V)`) so they work over arbitrary genome/label types, not just numeric vectors.
- **Tests are consolidated into a single file**, `spec/stochasta_spec.cr` (not one spec file per source file), with one `it` block per algorithm demonstrating a realistic, checkable scenario (e.g. OneMax for GA, Sphere/Rosenbrock for continuous optimizers, a 4-city TSP for ACO). Follow this pattern for new algorithms rather than creating a new spec file. Numeric assertions use `be_close` with tolerances appropriate to the algorithm's stochastic convergence, not exact equality.
- **`examples/*.cr`** are standalone, runnable, well-commented demonstrations of each algorithm's public API (one file per algorithm, named `<algorithm>_example.cr`) — kept in sync with the README's "Usage Examples" table. When adding an algorithm, add both a spec `it` block and an example file, and add a row to `README.md`'s Features and Usage Examples sections.
