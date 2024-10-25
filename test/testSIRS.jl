NV = 10 # number of graph vertices
k = 3 # average degree

#genrate an Erdos-Renyi random graph with average connectivity k
Random.seed!(1)
G = erdos_renyi(NV, k/NV)

# define the constants
T = 5 # total time
γ = 1/NV # Patient zero probability
λ₀ = 0.7 # Infection rate
r₀ = 0.2 # Recovery rate
σ₀ = 0.3 # Loss of immunity rate

# constant infection probability
λ = zeros(NV, NV, T)
for e in edges(G)
    λ[src(e), dst(e), :] = ones(T) * λ₀
    λ[dst(e), src(e), :] = ones(T) * λ₀
end

# define de epidemic model
infectionmodel = SIRS(0.0, r₀, σ₀, NV, T)
model = EpidemicModel(infectionmodel, G, T, log.(1 .- λ))

configtest=[1 1 1 1 2 0; 
            0 0 1 1 1 1; 
            0 0 0 0 1 1; 
            0 0 0 1 1 1; 
            0 0 1 2 2 2; 
            0 0 0 0 0 0; 
            0 0 0 1 1 1; 
            0 0 0 0 0 1; 
            0 0 0 0 1 1; 
            0 1 1 1 2 0]

# generate observations at the last time

obsmat = ones(NV, T+1) * (-1.0)
for iₗ in 1:NV
    obsmat[iₗ, end] = configtest[iₗ, end]
end

# insert the observations into the model structure
model.obsmat .= obsmat

# computational constants
epsconv = 1e-10 # convergence threshold
maxiter = 500 # max number of iterations
damp = 0.1 # damping factor
μ_cutoff = -1.0 # cutoff for convergence

margtest = load("data/margSIRS.jld2", "marg")

@testset "SimSIRS" begin
    # epidemic simulation
    Random.seed!(6)
    config = sim_epidemics(model, patient_zero=[1])

    @test config ≈ configtest
end

@testset "inferenceSIRS" begin
    Random.seed!(1)
    nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp, μ_cutoff=μ_cutoff, n_iter_nc=100, damp_nc=0.1)

    marg = zeros(NV,3,T+1)
    for (i,node) in enumerate(nodes)
        marg[i,:,:] = node.marg.m
    end

    @test marg ≈ margtest
end


@testset "inferenceSIscheme" begin
    maxiter = [400, 300]  # max number of iterations scheme
    damp = [0.9, 0.5]  # damping factor scheme

    Random.seed!(1)
    nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp, μ_cutoff = μ_cutoff)

    marg = zeros(NV,3,T+1)
    for (i,node) in enumerate(nodes)
        marg[i,:,:] = node.marg.m
    end

    margtestscheme = load("data/margSIRSscheme.jld2", "marg")
    @test marg ≈ margtestscheme
end
