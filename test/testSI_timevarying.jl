NV = 10 # number of graph vertices
k = 3 # average degree

# define the constants
T = 5 # total time
γ = 1/NV # Patient zero probability
λ₀ = 0.3 # Infection rate

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
infectionmodel = SI(0.0, NV, T)
model = EpidemicModel(infectionmodel, G, T, log.(1 .- λ))

configtest=[1 1 1 1 1 1; 
            0 0 0 1 1 1; 
            0 0 0 0 0 0; 
            0 0 0 0 1 1; 
            0 0 0 0 0 1; 
            0 0 0 0 0 0; 
            0 0 0 0 0 0; 
            0 0 0 0 0 1; 
            0 0 0 0 0 0; 
            0 0 0 0 0 0]

# generate observations at the last time
obsmat = ones(Int8, NV, T+1) * (-1)
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

margtest = load("data/margSI_timevarying.jld2", "marg")

@testset "SimSI_timevarying" begin
    # epidemic simulation
    Random.seed!(3)
    config = sim_epidemics(model, patient_zero=[1])

    @test config ≈ configtest
end

@testset "inferenceSI_timevarying" begin
    Random.seed!(1)
    nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp, μ_cutoff = μ_cutoff)

    marg = zeros(NV,2,T+1)
    for (i,node) in enumerate(nodes)
        marg[i,:,:] = node.marg.m
    end

    @test marg ≈ margtest
end


@testset "inferenceSIscheme" begin
    maxiter = [90, 50]  # max number of iterations scheme
    damp = [0.9, 0.5]  # damping factor scheme

    Random.seed!(1)
    nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp, μ_cutoff = μ_cutoff)

    marg = zeros(NV,2,T+1)
    for (i,node) in enumerate(nodes)
        marg[i,:,:] = node.marg.m
    end

    margtestscheme = load("data/margSIscheme_timevarying.jld2", "marg")
    @test marg ≈ margtestscheme
end
