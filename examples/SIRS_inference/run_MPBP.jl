using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs, Statistics, Random, StatsBase
using ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
using JLD2
using TensorTrains: summary_compact;
using LinearAlgebra

isdir("data") || mkdir("data")
isdir("logs") || mkdir("logs")

T = 20
N = 100
c = 2.5

ρ = 0.15
σ = 0.15
n_sr = 2
γ = n_sr/N

of = 0.75 #observation fractions
nobs = floor(Int, N * of)
obs_time = 10

isdir("data/ER_N-$(N)_T-$(T)_c-$(c)_rho-$(ρ)_sigma-$(σ)_nsrc-$(n_sr)_nobs-$(nobs)_obst-$(obs_time)") || mkdir("data/ER_N-$(N)_T-$(T)_c-$(c)_rho-($ρ)_sigma-$(σ)_nsrc-$(n_sr)_nobs-$(nobs)_obst-$(obs_time)")
isdir("logs/ER_N-$(N)_T-$(T)_c-$(c)_rho-$(ρ)_sigma-$(σ)_nsrc-$(n_sr)_nobs-$(nobs)_obst-$(obs_time)") || mkdir("logs/ER_N-$(N)_T-$(T)_c-$(c)_rho-($ρ)_sigma-$(σ)_nsrc-$(n_sr)_nobs-$(nobs)_obst-$(obs_time)")
isdir("logs/ER_N-$(N)_T-$(T)_c-$(c)_rho-($ρ)_sigma-$(σ)_nsrc-$(n_sr)_nobs-$(nobs)_obst-$(obs_time)/MPBP") || mkdir("logs/ER_N-$(N)_T-$(T)_c-$(c)_rho-($ρ)_sigma-$(σ)_nsrc-$(n_sr)_nobs-$(nobs)_obst-$(obs_time)/MPBP")

λs = [0.1, 0.5, 0.55, 0.6]
nseeds = 50
seeds = rand(Xoshiro(1234), 1:10^5, nseeds)
s_dict = Dict(seed=>s for (s,seed) in enumerate(seeds))

ks = [1,2,3,4]

datadir = "data/ER_N-$(N)_T-$(T)_c-$(c)_rho-($ρ)_sigma-$(σ)_nsrc-$(n_sr)_nobs-$(nobs)_obst-$(obs_time)"
@showprogress for (seed, λ) in Iterators.product(seeds, λs)
    isdir(datadir*"/lambda-$(λ)") || mkdir(datadir*"/lambda-$(λ)")
    s = s_dict[seed]
    gg = erdos_renyi(N, c/N; seed)
    g = IndexedGraph(gg)
    sirs = SIRS(g, λ, ρ, σ, T; γ)
    bp = mpbp(sirs)
    reset!(bp)
    rng = MersenneTwister(seed)
    X, observed = draw_node_observations!(bp, nobs, times = obs_time + 1, softinf=Inf; rng)
    JLD2.save(datadir*"/lambda-$(λ)/s-$(s)_graph_confs_observed_seed.jld2", "graph", g, "confs", X, "observed", observed, "seed", seed)

    for k in ks
        logsdir = "logs/ER_N-$(N)_T-$(T)_c-$(c)_rho-($ρ)_sigma-$(σ)_nsrc-$(n_sr)_nobs-$(nobs)_obst-$(obs_time)/MPBP/k-$(k)"
        isdir(logsdir) || mkdir(logsdir)
        isdir(logsdir*"/lambda-$(λ)") || mkdir(logsdir*"/lambda-$(λ)")
        iters, = iterate!(bp, maxiter=30; cb = CB_BP(bp, showprogress=false), svd_trunc = TruncBondThresh(k,1e-5), tol=1e-5, damp=0.2);
        JLD2.save_object(logsdir*"/lambda-$(λ)/s-$(s)_margs_bp.jld2", beliefs(bp))
    end
end
            