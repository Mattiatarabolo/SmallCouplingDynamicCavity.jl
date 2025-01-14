using Graphs
using NPZ, CSV, DataFrames
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
isdir("data/test_SI") || mkdir("data/test_SI")
isdir("data/test_SI/var_obs") || mkdir("data/test_SI/var_obs")
isdir("data/test_SI/var_obs/data_rand_geom") || mkdir("data/test_SI/var_obs/data_rand_geom")

# Parameters
T = 15
NV = 300
lmax = round(sqrt(1.8/NV), digits=3)
n_sr = 2
γ = n_sr/NV
λ₀ = 0.07
fracs_obs = collect(range(.05, .95, length=10))

isdir("data/test_SI/var_obs/data_rand_geom/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_lamb-$(λ₀)") || mkdir("data/test_SI/var_obs/data_rand_geom/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_lamb-$(λ₀)")

# Random number generator
seed = 1234
rng = Xoshiro(seed)

n_samples = 50

println("Proximity:")
# Generate the data
λ₀ = round(λ₀, digits=4)

@showprogress for s in 1:n_samples
    path = "data/test_SI/var_obs/data_rand_geom/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_lamb-$(λ₀)/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_lam-$(λ₀)_s-$(s)"
    # Generate the graph
    G = proximity(NV, lmax, rng=rng)
    contacts = zeros(ne(G)*2*T, 4)
    λ = zeros(NV, NV, T)
    idx = 1
    for t in 0:T-1
        for e in edges(G)
            i, j = src(e), dst(e)

            contacts[idx, :] = [t, i-1, j-1, λ₀]
            contacts[idx+1, :] = [t, j-1, i-1, λ₀]
            idx += 2

            λ[i, j, t+1] = λ₀
            λ[j, i, t+1] = λ₀
        end
    end

    # Save the graph and contacts
    npzwrite(path * "_contacts.npy", contacts)

    # define de epidemic model
    infectionmodel = SI(0.0, NV, T)
    model = EpidemicModel(infectionmodel, G, T, log.(1 .- λ))

    # Generate the epidemic cascade
    p0 = sample(rng, 1:NV, n_sr; replace=false, ordered=true)
    config = sim_epidemics(model, patient_zero=p0)
    config_npz = zeros(1, 2, NV)
    config_npz[1, 1, :] = config[:, 1]
    config_npz[1, 2, :] = config[:, end]

    # Save the epidemic cascade
    npzwrite(path * "_confs.npy", config_npz)
    
    for frac_obs in fracs_obs
        nobs = floor(Int, frac_obs*NV)
        
        nobs_dirpath = "data/test_SI/var_obs/data_rand_geom/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_lamb-$(λ₀)/nobs_$(nobs)"
        isdir(nobs_dirpath) || mkdir(nobs_dirpath)
        
        nobs_path = nobs_dirpath*"/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_lam-$(λ₀)_s-$(s)_obs_sparse.csv"
        
        # generate observations at the last time
        obsmat = ones(nv(G), T+1) * (-1.0)
        obs_node = zeros(Int, nobs)
        obs_time = ones(Int, nobs) * T
        obs_state = zeros(Int, nobs)
        unobs_node = zeros(Int, NV-nobs)
        unobs_state = zeros(Int, NV-nobs)
        
        while sum(unobs_state)==0 || sum(unobs_state)==nobs
            for (idx_temp, iₗ) in enumerate(sample(rng, 1:NV, nobs; replace=false, ordered=true))
                obsmat[iₗ, T+1] = config[iₗ, T+1]
                obs_node[idx_temp] = iₗ - 1
                obs_state[idx_temp] = config[iₗ, T+1]
            end
            
            i = 1
            for idx in 1:NV
                if idx ∉ obs_node.+1
                    unobs_node[i] = idx
                    unobs_state[i] = config[idx, T+1]
                    i += 1
                end
            end
        end

        # Save the observations
        obs_df = DataFrame(node=obs_node, obs_st=obs_state, time=obs_time)
        CSV.write(nobs_path, obs_df)
    end
end