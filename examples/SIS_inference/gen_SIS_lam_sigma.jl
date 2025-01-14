using Graphs
using NPZ, CSV, DataFrames, JLD2
using Random, StatsBase
using SmallCouplingDynamicCavity
using ProgressMeter


########## Proximity generator ##########
function proximity(n::Int, lmax::Float64; rng::AbstractRNG=Xoshiro(1234))
    # Uniformly distribute n vertices in the unit square
    x = rand(rng, n)
    y = rand(rng, n)

    # Initialize the graph
    G = SimpleGraph(n)

    # Connect vertices with distance less than lmax with expoentially decreasing probability
    for i in 1:n
        for j in i+1:n
            d = sqrt((x[i] - x[j])^2 + (y[i] - y[j])^2)
            if d < lmax && rand(rng) < exp(-d)
                add_edge!(G, i, j)
                add_edge!(G, j, i)
            end
        end
    end
    return G
end

# check if data directory exists
isdir("data") || mkdir("data")
isdir("data/test_SIS") || mkdir("data/test_SIS")
isdir("data/test_SIS/var_lambda_sigma") || mkdir("data/test_SIS/var_lambda_sigma")

# Parameters
T = 15
NV = 300
lmax = round(sqrt(1.8/NV), digits=3)
n_sr = 2
γ = n_sr/NV
λs = round.(collect(range(0.1, 0.9, length=20)), digits=3)
σs   = round.(10 .^ (collect(range(log10(0.01), log10(0.9), length=20))), digits=3)
λ_σs = collect(Iterators.product(λs, σs))
nobs = floor(Int, .6*NV)

isdir("data/test_SIS/var_lambda_sigma/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_nobs-$(nobs)") || mkdir("data/test_SIS/var_lambda_sigma/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_nobs-$(nobs)")

# Random number generator
seed = 1234
rng = Xoshiro(seed)

n_samples = 30

println("Proximity:")
# Generate the data
@showprogress for (idx, (λ₀,σ₀)) in enumerate(λ_σs)
    isdir("data/test_SIS/var_lambda_sigma/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_nobs-$(nobs)/lam_$(λ₀)_sigma_$(σ₀)") || mkdir("data/test_SIS/var_lambda_sigma/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_nobs-$(nobs)/lam_$(λ₀)_sigma_$(σ₀)")
    for s in 1:n_samples
        path = "data/test_SIS/var_lambda_sigma/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_nobs-$(nobs)/lam_$(λ₀)_sigma_$(σ₀)/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_lam-$(λ₀)_sigma-$(σ₀)_nobs-$(nobs)_s-$(s).jld2"
        # Generate the graph
        G = proximity(NV, lmax, rng=rng)
        λ = zeros(NV, NV, T)

        for t in 0:T-1
            for e in edges(G)
                λ[src(e), dst(e), t+1] = λ₀
                λ[dst(e), src(e), t+1] = λ₀
            end
        end

        # define de epidemic model
        infectionmodel = SIS(0.0, σ₀, NV, T)
        model = EpidemicModel(infectionmodel, G, T, log.(1 .- λ))

        # Generate the epidemic cascade
        p0 = sample(rng, 1:NV, n_sr; replace=false, ordered=true)
        config = sim_epidemics(model, patient_zero=p0)

        # generate observations at the last time
        obsmat = ones(Int8, nv(G), T+1) * Int8(-1)
        
        for (idx_temp, iₗ) in enumerate(sample(rng, 1:NV, nobs; replace=false, ordered=true))
            obsmat[iₗ, floor(Int, T/2)] = config[iₗ, floor(Int, T/2)]
        end

        # Save the graph, configurations and observations
        JLD2.save(path, "graph", G, "config", config, "obsmat", obsmat)
    end
end