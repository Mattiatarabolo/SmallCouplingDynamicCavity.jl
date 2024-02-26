# SmallCouplingDynamicCavity

*Small Coupling expansion of the Dynamic Cavity method for epidemic inference*

[![Build Status](https://github.com/Mattiatarabolo/SmallCouplingDynamicCavity.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Mattiatarabolo/SmallCouplingDynamicCavity.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://MattiaTarabolo.github.io/SmallCouplingDynamicCavity.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://MattiaTarabolo.github.io/SmallCouplingDynamicCavity.jl/dev)
<!---[![codecov.io](http://codecov.io/github/Mattiatarabolo/SmallCouplingDynamicCavity.jl/coverage.svg?branch=main)](http://codecov.io/github/Mattiatarabolo/SmallCouplingDynamicCavity.jl/coverage.svg?branch=main)--->



## Overview

The purpose of _SmallCouplingDynamicCavity.jl_ is to provide a general and computationally efficient solution for Bayesian epidemic inference and risk assessment. The package offers an efficient structure implementation for the most used epidemic models, such as Susceptible-Infected (SI), Susceptible-Infected-Recovered (SIR), Susceptible-Infected-Susceptible (SIS) and Susceptible-Infected-Recovered-Susceptible (SIRS).

For all these models, the package provides:

- a simulation tool, which allows to sample an epidemic outbreak with specified parameters
- a statistical inference tool, which allows to obtain fully bayesian estimates of the epidemic uotbreak

## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia-repl
pkg> add SmallCouplingDynamicCavity
```

Or, equivalently, via the `Pkg` API:

```julia-repl
julia> import Pkg; Pkg.add("SmallCouplingDynamicCavity")
```

## Basic use

Define an infection model through the available structures.

```julia-repl
# SI model with 0.0 self-infection rate, 4 individuals and 5 epidemic timesteps
julia> infection_model = SI(0.0, 4, 5)
SI([0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0])
```

Define an epidemic model
```julia-repl
# SI epidemic model on a graph G with infection probability encoded by the matrix λ 
julia> model = EpidemicModel(infection_model, G, 5, log.(1 .- λ))
```    

- ### Sampling
    ```julia-repl
    # Sample the epidemic cascade specifying the patient zero as individual 1
    julia> config = sim_epidemics(model, patient_zero=[1])
    4×6 Matrix{Float64}:
    1.0  1.0  1.0  1.0  1.0  1.0
    0.0  1.0  1.0  1.0  1.0  1.0
    0.0  1.0  1.0  1.0  1.0  1.0
    0.0  0.0  0.0  1.0  1.0  1.0
    ```
- ### Inference
    ```julia-repl
    # Insert the observations as a matrix (-1.0 = unobserved, 0.0 = observed S, 1.0 = observed I)
    julia> model.obsmat .= [-1.0 -1.0 1.0 -1.0 -1.0 -1.0; 0.0 -1.0 -1.0 -1.0 1.0 -1.0; -1.0 -1.0 -1.0 -1.0 -1.0 -1.0;  -1.0 -1.0 -1.0 -1.0 -1.0 1.0]
    4×6 Matrix{Float64}:
    -1.0  -1.0   1.0  -1.0  -1.0  -1.0
     0.0  -1.0  -1.0  -1.0   1.0  -1.0
    -1.0  -1.0  -1.0  -1.0  -1.0  -1.0
    -1.0  -1.0  -1.0  -1.0  -1.0   1.0

    # Run the inference algorithm with maximum 10 iterations, a convergence threshold of 0.1, and a damping factor of 0.0. The prior probability of being infected at time 0 is 1/4, and the observation probability obsprob is user-specified
    julia> nodes = run_SCDC(model, obsprob, 1/4, 10, 0.1, 0.0)
    Converged after 4 iterations
    4-element Vector{SmallCouplingDynamicCavity.Node{SI}}
    ```

A more detailed use of the package is presented in the [Tutorial](notebook/Tutorial.ipynb).

