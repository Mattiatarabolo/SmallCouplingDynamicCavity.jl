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

### Sampling
```julia-repl
# Sample the epidemic cascade specifying the patient zero as individual 1
julia> config = sim_epidemics(model, patient_zero=[1])
4×6 Matrix{Float64}:
1.0  1.0  1.0  1.0  1.0  1.0
0.0  1.0  1.0  1.0  1.0  1.0
0.0  1.0  1.0  1.0  1.0  1.0
0.0  0.0  0.0  1.0  1.0  1.0
```
### Inference
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