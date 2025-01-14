import MatrixProductBP as MPBP
import MatrixProductBP.Models as MPBPm
import SmallCouplingDynamicCavity as SCDC
using Graphs
using JLD2
using IndexedGraphs
using Random, StatsBase
using Statistics
using TensorTrains: summary_compact
using LinearAlgebra

# Define .log printing utils
function write_log(logstring::AbstractString, logfile)
    open(logfile, "a+") do file
        write(file, logstring*"\n")
    end
end

# Define .err printing utils
function write_err(errstring::AbstractString, errfile)
    open(errfile, "a+") do file
        write(file, errstring*"\n")
    end
end

# Define the saving structure, in which the results will be saved
struct SaveStruct
    s::Int
    ε_autoinf::Float64
    maxiter::Union{Int,Array{Int}}
    damp::Union{Float64,Array{Float64}}
    μ_cutoff::Float64
    fpn::Float64
    n_iter_nc::Int
    damp_nc::Float64
    fMAP_true::Float64

    SaveStruct(s) = new(s, 0.0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
    SaveStruct(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, n_iter_nc, damp_nc, fMAP_true) = new(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, n_iter_nc, damp_nc, fMAP_true)
end

function init_save(n_sims, save_file)
    if isfile(save_file)
        save_vec = JLD2.load_object(save_file)
        return save_vec
    else
        # initialising the saving file
        save_vec = [SaveStruct(s) for s in 1:n_sims]
        return save_vec
    end
end

# Function to find the index of the corresponding s
function find_s_index(s, save_vec)
    for (i, save_struct) in enumerate(save_vec)
        if save_struct.s == s
            return i
        end
    end
    return 0
end

# Define .jld2 saving utilities
function save_function(s,ε_autoinf,maxiter,damp,μ_cutoff,fpn,n_iter_nc,damp_nc,fMAP_true,save_vec,save_file)
    save_vec[find_s_index(s,save_vec)] = SaveStruct(s,ε_autoinf,maxiter,damp,μ_cutoff,fpn,n_iter_nc,damp_nc,fMAP_true)
    JLD2.save_object(save_file, save_vec)
end


# define the observation probability
function obsprob(Ob, x)
    if Ob == -1
        return 1.0
    else
        return Float64(Ob == x)
    end
end


function fMAP_true_func(margs, X)
    N, T = size(X)
    T -= 1
    fMAP_true = sum((argmax(margs[i][:, t]) == (X[i, t])) / (N * T) for i in 1:N for t in 1:T)
    return fMAP_true
end



# check if data directory exists
isdir("logs") || mkdir("logs")


# Define the parameters
T = 20
N = 100
c = 2.5

ρ = 0.15
σ = 0.15
γ = 2/N;

λs = 0.1:0.05:0.6
nseeds = 50
seeds = 1:nseeds
of = 0.75

# computational constants
epsconv = 1e-8 # convergence threshold
ε_autoinfs = [1e-4, 1e-5, 1e-7, 1e-10, 1e-15, 1e-20, 1e-25]
n_iters = [[10, 50, 200]]
damps = [[0.0, 0.1, 0.3], [0.0, 0.5, 0.7], [0.1, 0.3, 0.5], [0.1, 0.5, 0.7], [0.1, 0.5, 0.9]] # damping factor
μ_cutoffs = [-1., -10., -20., -30.] # cutoff for convergence
fpns = [0.0] # false positive and false negative rates
n_iter_ncs = [10, 50]
damp_ncs = [0.0, 0.3, 0.5, 0.7, 0.9]

rng = MersenneTwister()

for (l_idx, λ₀) in enumerate(λs)
    println("λ = $λ₀")
    logs_path = "logs/lam_$(λ₀)"
    isdir(logs_path) || mkdir(logs_path)
    logfile = logs_path*"/optim_pars_ER.log"
    save_file = logs_path*"/optim_pars_ER.jld2"
    err_file = logs_path*"/optim_pars_ER.err"

    println("Starting ER graph inference")
    write_log("Starting ER graph inference", logfile)
    write_err("Starting ER graph inference", err_file)

    # initialising the csv file
    save_vec = init_save(nseeds, save_file)

    for s in seeds
        maxval = save_vec[find_s_index(s,save_vec)].fMAP_true
        # generate graph (same as MPBP)
        gg = erdos_renyi(N, c/N; seed=s)
        g = IndexedGraph(gg)

        # simulate the epidemic
        sirs = MPBPm.SIRS(g, λ₀, ρ, σ, T; γ)
        bp = MPBP.mpbp(sirs)
        obs_time = 10
        nobs = floor(Int, N * of)
        obs_fraction = nobs / N
        Random.seed!(rng, s)
        X, observed = MPBP.draw_node_observations!(bp, nobs, times = [obs_time + 1], softinf=Inf; rng) # X planted, observed list of tuples (node,tobs)
        # generate observations
        obsmat = ones(Int8, N, T+1) * Int8(-1)
        for (iₗ, τₗ) in observed
            obsmat[iₗ, τₗ] = X[iₗ, τₗ]-1
        end
        
        # generate infection matrix
        λ = zeros(N, N, T+1)
        for e in edges(gg)
            λ[src(e), dst(e), :] .+= λ₀
            λ[dst(e), src(e), :] .+= λ₀
        end

        #g, sirs, bp, obs_time, nobs, obs_fraction, rng2, observed = nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing

        for (ε_autoinf, maxiter, damp, μ_cutoff, fpn) in Iterators.product(ε_autoinfs, n_iters, damps, μ_cutoffs, fpns)
            # define de epidemic model
            infectionmodel = SCDC.SIRS(ε_autoinf, ρ, σ, N, T)
            model = SCDC.EpidemicModel(infectionmodel, gg, T, log.(1 .- λ), obsmat)

            # run the inference
            Random.seed!(rng, 1234)
            try
                nodes = SCDC.run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            catch
                println("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn")
                write_err("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn", err_file)
                
                μ_cutoff /= 10
                Random.seed!(rng, 1234)
                nodes = SCDC.run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            end

            for (n_iter_nc, damp_nc) in Iterators.product(n_iter_ncs, damp_ncs)
                nodes_nc = deepcopy(nodes)
                Random.seed!(rng, 1234)
                try
                    SCDC.run_SCDC!(nodes_nc, model, γ, 0, epsconv, 0.0; μ_cutoff=μ_cutoff, n_iter_nc=n_iter_nc, damp_nc=damp_nc, rng=rng)
                catch
                    println("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc")
                    write_err("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc", err_file)
                    return nothing
                end
                
                margs = [node.marg.m for node in nodes_nc]
                nodes_nc = nothing

                fMAP_true = fMAP_true_func(margs, X)
                margs = nothing

                if fMAP_true > maxval
                    maxval = fMAP_true
                    println("s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc, fMAP_true: $fMAP_true")
                    write_log("s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc, fMAP_true: $fMAP_true", logfile)
                    save_function(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, n_iter_nc, damp_nc, fMAP_true, save_vec, save_file)
                end
            end
            nodes = nothing
        end
    end
end