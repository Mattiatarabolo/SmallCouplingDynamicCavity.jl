# SmallCouplingDynamicCavity

*Small Coupling expansion of the Dynamic Cavity method for epidemic inference*

[![Build Status](https://github.com/Mattiatarabolo/SmallCouplingDynamicCavity.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Mattiatarabolo/SmallCouplingDynamicCavity.jl/actions/workflows/CI.yml?query=branch%3Amain)


## Overview

The purpose of _SmallCouplingDynamicCavity.jl_ is to provide a general and computationally efficient solution for Bayesian epidemic inference and risk assessment. The package offers an efficient structure implementation for the most used epidemic models, such as Susceptible-Infected (SI), Susceptible-Infected-Recovered (SIR), Susceptible-Infected-Susceptible (SIS) and Susceptible-Infected-Recovered-Susceptible (SIRS).

For all these models, the package provides:

- a simulation tool, which allows to sample an epidemic outbreak with specified parameters
- a statistical inference tool, which allows to obtain fully bayesian estimates of the epidemic uotbreak

## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add SmallCouplingDynamicCavity
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("SmallCouplingDynamicCavity")
```
