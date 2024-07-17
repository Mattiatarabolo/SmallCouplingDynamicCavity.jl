NV = 10 # number of graph vertices
k = 3 # average degree

# define the constants
T = 5 # total time
γ = 1/NV # Patient zero probability
λ₀ = 0.5 # Infection rate
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
infectionmodel = SIS(0.0, r₀, NV, T)
model = EpidemicModel(infectionmodel, G, T, log.(1 .- λ))

configtest=[1.0 1.0 1.0 1.0 1.0 0.0; 
            0.0 1.0 1.0 1.0 0.0 0.0; 
            0.0 0.0 1.0 1.0 1.0 0.0; 
            0.0 0.0 0.0 1.0 1.0 1.0; 
            0.0 0.0 0.0 0.0 1.0 1.0; 
            0.0 0.0 0.0 0.0 1.0 1.0; 
            0.0 0.0 0.0 0.0 0.0 0.0; 
            0.0 0.0 1.0 1.0 0.0 0.0; 
            0.0 0.0 0.0 0.0 0.0 0.0; 
            0.0 0.0 0.0 0.0 0.0 0.0]


# generate observations at the last time
# define the observation probability
function obsprob(Ob::Float64, x::Float64)
    if Ob == -1.0
        return 1.0
    else
        return Float64(Ob == x)
    end
end

obsmat = ones(NV, T+1) * (-1.0)
for iₗ in 1:NV
    obsmat[iₗ, end] = configtest[iₗ, end]
end

# insert the observations into the model structure
model.obsmat .= obsmat

# computational constants
epsconv = 1e-10 # convergence threshold
maxiter = 5e2 # max number of iterations
damp = 0.1 # damping factor
μ_cutoff = -1 # cutoff for convergence

margtest = load("data/margSIS_timevarying.jld2", "marg")

@testset "SimSIS_timevarying" begin
    # epidemic simulation
    Random.seed!(3)
    config = sim_epidemics(model, patient_zero=[1])

    @test config ≈ configtest
end

@testset "inferenceSIS_timevarying" begin
    Random.seed!(1)
    nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp, μ_cutoff = μ_cutoff)

    marg = zeros(NV,2,T+1)
    for (i,node) in enumerate(nodes)
        marg[i,:,:] = node.marg.m
    end

    @test marg ≈ margtest
end