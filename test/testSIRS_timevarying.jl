NV = 10 # number of graph vertices
k = 3 # average degree

# define the constants
T = 5 # total time
γ = 1/NV # Patient zero probability
λ₀ = 0.7 # Infection rate
r₀ = 0.2 # Recovery rate
σ₀ = 0.3 # Loss of immunity rate

G = Vector{SimpleGraph{Int64}}()
λ = zeros(NV, NV, T)

#genrate Erdos-Renyi random graphs with average connectivity k
Random.seed!(1)
for t in 1:T
    g = erdos_renyi(NV, k/NV)
    push!(G,g)
    for e in edges(g)
        λ[src(e), dst(e), t] = rand() * λ₀
        λ[dst(e), src(e), t] = rand() * λ₀
    end
end

# define de epidemic model
infectionmodel = SIRS(0.0, r₀, σ₀, NV, T)
model = EpidemicModel(infectionmodel, G, T, log.(1 .- λ))

configtest=[1 2 2 0 1 1; 
            0 1 1 2 0 0; 
            0 0 0 1 1 1; 
            0 0 0 1 2 2; 
            0 0 0 0 1 1; 
            0 0 0 0 1 1; 
            0 0 1 1 1 1; 
            0 0 0 0 1 1; 
            0 0 1 1 1 1; 
            0 1 1 1 2 0]

# generate observations at the last time
# define the observation probability
function obsprob(Ob, x)
    if Ob == -1
        return 1.0
    else
        return Float64(Ob == x)
    end
end

obsmat = ones(Int8, NV, T+1) * (-1)
for iₗ in 1:NV
    obsmat[iₗ, end] = configtest[iₗ, end]
end

# insert the observations into the model structure
model.obsmat .= obsmat

# computational constants
epsconv::Float64 = 1e-10 # convergence threshold
maxiter::Int = 5e2 # max number of iterations
damp::Float64 = 0.1 # damping factor
μ_cutoff::Float64 = -1 # cutoff for convergence

margtest = load("data/margSIRS_timevarying.jld2", "marg")

@testset "SimSIRS_timevarying" begin
    # epidemic simulation
    Random.seed!(2)
    config = sim_epidemics(model, patient_zero=[1])

    @test config ≈ configtest
end

@testset "inferenceSIRS_timevarying" begin
    Random.seed!(1)
    nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp, μ_cutoff = μ_cutoff)

    marg = zeros(NV,3,T+1)
    for (i,node) in enumerate(nodes)
        marg[i,:,:] = node.marg.m
    end

    @test marg ≈ margtest
end