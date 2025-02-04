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
rng = Xoshiro(1)
for t in 1:T
    g = erdos_renyi(NV, k/NV, rng=rng)
    push!(G,g)
    for e in edges(g)
        λ[src(e), dst(e), t] = rand(rng) * λ₀
        λ[dst(e), src(e), t] = rand(rng) * λ₀
    end
end

# define de epidemic model
infectionmodel = SIR(0.0, r₀, NV, T)
model = EpidemicModel(infectionmodel, G, T, log.(1 .- λ))

configtest = load_object("data/configSIR_timevarying.jld2")

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

margtest = load_object("data/margSIR_timevarying.jld2")

@testset "SimSIR_timevarying" begin
    # epidemic simulation
    Random.seed!(rng, 3)
    config = sim_epidemics(model, patient_zero=[1], reject=true, rng=rng)

    @test config ≈ configtest
end

@testset "inferenceSIR_timevarying" begin
    Random.seed!(rng, 1)
    nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp, μ_cutoff = μ_cutoff, rng=rng)

    marg = zeros(NV,3,T+1)
    for (i,node) in enumerate(nodes)
        marg[i,:,:] = node.marg.m
    end

    @test marg ≈ margtest
end

########### checking averaging method when non-converged ##########
n_iter_nc = 10
damp_nc = 0.3

@testset "inferenceSIR_timevarying_nc" begin
    Random.seed!(rng, 1)
    nodes_nc = run_SCDC(model, obsprob, γ, 2, epsconv, damp, μ_cutoff=μ_cutoff, n_iter_nc=n_iter_nc, damp_nc=damp_nc, rng=rng)
    Random.seed!(rng, 1)
    nodes = run_SCDC(model, obsprob, γ, 2, epsconv, damp, μ_cutoff=μ_cutoff, rng=rng)
    run_SCDC!(nodes, model, γ, 0, epsconv, damp, μ_cutoff=μ_cutoff, n_iter_nc=n_iter_nc, damp_nc=damp_nc, rng=rng)

    marg = [node.marg.m[2,T+1] for node in nodes]
    marg_nc = [node.marg.m[2,T+1] for node in nodes_nc]

    @test marg ≈ marg_nc
end

############ checking averaging method when non-converged with damping scheme ##########
maxiter = [100, 90]  # max number of iterations scheme
damp = [0.9, 0.5]  # damping factor scheme
@testset "inferenceSIRscheme_timevarying" begin
    Random.seed!(rng, 1)
    nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp, μ_cutoff = μ_cutoff, rng=rng)

    marg = zeros(NV,3,T+1)
    for (i,node) in enumerate(nodes)
        marg[i,:,:] = node.marg.m
    end

    margtestscheme = load_object("data/margSIRscheme_timevarying.jld2")
    @test marg ≈ margtestscheme
end

########### checking averaging method when non-converged with damping scheme ##########
n_iter_nc = 10
damp_nc = 0.3

@testset "inferenceSIRscheme_timevarying_nc" begin
    Random.seed!(rng, 1)
    nodes_nc = run_SCDC(model, obsprob, γ, [2,2], epsconv, damp, μ_cutoff=μ_cutoff, n_iter_nc=n_iter_nc, damp_nc=damp_nc, rng=rng)
    Random.seed!(rng, 1)
    nodes = run_SCDC(model, obsprob, γ, [2,2], epsconv, damp, μ_cutoff=μ_cutoff, rng=rng)
    run_SCDC!(nodes, model, γ, [0,0], epsconv, damp, μ_cutoff=μ_cutoff, n_iter_nc=n_iter_nc, damp_nc=damp_nc, rng=rng)

    marg = [node.marg.m[2,T+1] for node in nodes]
    marg_nc = [node.marg.m[2,T+1] for node in nodes_nc]

    @test marg ≈ marg_nc
end