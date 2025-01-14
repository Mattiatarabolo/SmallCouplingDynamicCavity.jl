using Graphs
using NPZ, CSV, DataFrames, JLD2
using SmallCouplingDynamicCavity
using Random, StatsBase
using Statistics

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


# Define the observation probability function
function obsprob_fp_fn(Ob, x; fp::Float64=0.0, fn::Float64=0.0)
    if Ob == 0
        return (1-fp)*(1-x) + fn*x
    elseif Ob == 1
        return fp*(1-x) + (1-fn)*x
    elseif Ob == -1
        return 1.0
    end
end


function fMAP_true_func(margs, X)
    N, T = size(X)
    T -= 1
    fMAP_true = sum((argmax(margs[i][:, t]) == (X[i, t]+1)) / (N * T) for i in 1:N for t in 2:T+1)
    return fMAP_true
end



################################################################################
#####################      Varying lambda     ##################################
################################################################################
# check if data directory exists
isdir("logs") || mkdir("logs")
isdir("logs/test_SIS") || mkdir("logs/test_SIS")
isdir("logs/test_SIS/var_lambda_sigma") || mkdir("logs/test_SIS/var_lambda_sigma")


##################### Random Geometric Graphs ##################################
n_sample = 30
sims = collect(1:n_sample)

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

isdir("logs/test_SIS/var_lambda_sigma/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_nobs-$(nobs)") || mkdir("logs/test_SIS/var_lambda_sigma/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_nobs-$(nobs)")

# computational constants
epsconv = 1e-6 # convergence threshold
ε_autoinfs = [1e-15]
n_iters = [100]
damps = [0.0] # damping factor
μ_cutoffs = [-10.] # cutoff for convergence
fpns = [0.0] # false positive and false negative rates
n_iter_ncs = [50]
damp_ncs = [0.1]


# Random number generator
seed = 1234
rng = Xoshiro(seed)

for (idx, (λ₀,σ₀)) in enumerate(λ_σs)
    println("λ = $λ₀, σ = $σ₀")
    logs_path = "logs/test_SIS/var_lambda_sigma/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_nobs-$(nobs)/lam_$(λ₀)_sigma_$(σ₀)"
    isdir(logs_path) || mkdir(logs_path)
    logfile = logs_path*"/optim_pars_rand_geom.log"
    save_file = logs_path*"/optim_pars_rand_geom.jld2"
    err_file = logs_path*"/optim_pars_rand_geom.err"

    println("Starting random geometric graph inference")
    write_log("Starting random geometric graph inference", logfile)
    write_err("Starting random geometric graph inference", err_file)
    
    # initialising the csv file
    save_vec = init_save(n_sample, save_file)
    
    for s in sims
        maxval = save_vec[find_s_index(s,save_vec)].fMAP_true
        
        # Load the data
        path = "data/test_SIS/var_lambda_sigma/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_nobs-$(nobs)/lam_$(λ₀)_sigma_$(σ₀)/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_lam-$(λ₀)_sigma-$(σ₀)_nobs-$(nobs)_s-$(s).jld2"
        data = JLD2.load(path)

        G = data["graph"]
        config = data["config"]
        obsmat = data["obsmat"]

        # create the interaction matrix
        λ = zeros(NV, NV, T)
        for t in 0:T-1
            for e in edges(G)
                λ[src(e), dst(e), t+1] = λ₀
                λ[dst(e), src(e), t+1] = λ₀
            end
        end

        data = nothing

        for (ε_autoinf, maxiter, damp, μ_cutoff, fpn) in Iterators.product(ε_autoinfs, n_iters, damps, μ_cutoffs, fpns)
            # Define the epidemic model
            infectionmodel = SIS(ε_autoinf, σ₀, NV, T)
            obsprob(Ob, x) = obsprob_fp_fn(Ob, x; fp=fpn, fn=fpn)
            model = EpidemicModel(infectionmodel, G, T, log.(1 .- λ), obsmat)

            # run the inference
            Random.seed!(rng, 1234)
            try
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            catch
                println("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn")
                write_err("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn", err_file)
                
                μ_cutoff /= 10
                Random.seed!(rng, 1234)
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            end

            for (n_iter_nc, damp_nc) in Iterators.product(n_iter_ncs, damp_ncs)
                nodes_nc = deepcopy(nodes)
                Random.seed!(rng, 1234)
                try
                    run_SCDC!(nodes_nc, model, γ, 0, epsconv, 0.0; μ_cutoff=μ_cutoff, n_iter_nc=n_iter_nc, damp_nc=damp_nc, rng=rng)
                catch
                    println("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc")
                    write_err("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc", err_file)
                    return nothing
                end
                
                margs = [node.marg.m for node in nodes_nc]
                nodes_nc = nothing

                fMAP_true = fMAP_true_func(margs, config)
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
