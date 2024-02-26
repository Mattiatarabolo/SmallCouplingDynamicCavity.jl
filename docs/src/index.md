# SmallCouplingDynamicCavity.jl

Documentation for SmallCouplingDynamicCavity.jl. The GitHub repository can be found at [SmallCouplingDynamicCavity.jl.git](https://github.com/Mattiatarabolo/SmallCouplingDynamicCavity.jl).


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

### [Index](@id main-index)

```@index
Pages = ["functions.md"]
```