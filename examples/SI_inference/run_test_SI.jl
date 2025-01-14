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
    AUC::Float64
    converged::Bool

    SaveStruct(s) = new(s, 0.0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, false)
    SaveStruct(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, n_iter_nc, damp_nc, auc, converged) = new(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, n_iter_nc, damp_nc, auc, converged)
end

function init_save_rg(n_sims, save_file)
    if isfile(save_file)
        save_vec = JLD2.load_object(save_file)
        return save_vec
    else
        # initialising the saving file
        save_vec = [SaveStruct(s) for s in 1:n_sims]
        return save_vec
    end
end

function init_save(n_sims, save_file)
    if isfile(save_file)
        save_vec = JLD2.load_object(save_file)
        return save_vec
    else
        # initialising the saving file
        save_vec = [SaveStruct(s) for s in 0:n_sims-1]
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
function save_function(s,ε_autoinf,maxiter,damp,μ_cutoff,fpn,n_iter_nc,damp_nc,auc,converged,save_vec,save_file)
    save_vec[find_s_index(s,save_vec)] = SaveStruct(s,ε_autoinf,maxiter,damp,μ_cutoff,fpn,n_iter_nc,damp_nc,auc,converged)
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



#=
################################################################################
#####################      Varying lambda     ##################################
################################################################################
# check if data directory exists
isdir("logs") || mkdir("logs")
isdir("logs/test_SI") || mkdir("logs/test_SI")
isdir("logs/test_SI/var_lambda") || mkdir("logs/test_SI/var_lambda")


##################### Random Geometric Graphs ##################################
n_sample = 50
sims = collect(1:n_sample)

# Parameters
T = 15
NV = 300
lmax = sqrt(1.8/NV)
n_sr = 2
γ = n_sr/NV
λs = 10 .^ (collect(range(log10(0.015), log10(0.5), length=10)))[8:end]
nobs = floor(Int, .3*NV)

# computational constants
epsconv = 1e-4 # convergence threshold
ε_autoinfs = [1e-5, 1e-10, 1e-15, 1e-20]
n_iters = [500, 1000, 5000]
damps = [0.0, 0.2, 0.5, 0.7] # damping factor
μ_cutoffs = [-10., -100.] # cutoff for convergence
fpns = [0.0, 1e-9, 1e-15] # false positive and false negative rates
n_iter_ncs = [0]
damp_ncs = [0.0]


# Random number generator
seed = 1234
rng = Xoshiro(seed)

for λ₀ in λs
    λ₀ = round(λ₀, digits=4)
    println("λ = $λ₀")
    logs_path = "logs/test_SI/var_lambda/lam_$(λ₀)"
    isdir(logs_path) || mkdir(logs_path)
    logfile = logs_path*"/optim_pars_rand_geom.log"
    save_file = logs_path*"/optim_pars_rand_geom.jld2"
    err_file = logs_path*"/optim_pars_rand_geom.err"

    println("Starting random geometric graph inference")
    write_log("Starting random geometric graph inference", logfile)
    write_err("Starting random geometric graph inference", err_file)
    
    # initialising the csv file
    save_vec = init_save_rg(n_sample, save_file)
    
    if λ₀==0.2294
        sims = collect(21:n_sample)
    else
        sims = collect(1:n_sample)
    end
    
    for s in sims

        maxAUC = save_vec[find_s_index(s,save_vec)].AUC
        
        # Load the data
        path = "data/test_SI/var_lambda/data_rand_geom/lam_$(λ₀)/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_lam-$(λ₀)_nobs-$(nobs)_s-$(s)"
        contacts = npzread(path*"_contacts.npy")
        confs = npzread(path*"_confs.npy")
        obs_df = CSV.read(path*"_obs_sparse.csv", DataFrame)
        obs_df.node .+= 1
        obs_df.time .+= 1

        # create the graph
        λ = zeros(NV, NV, T)
        G = SimpleGraph(NV, 0)

        for (t,i,j,λᵢⱼ) in eachrow(contacts)
            if λᵢⱼ != λ₀
                throw(ArgumentError("λᵢⱼ ≠ λ₀"))
            end
            if t==0.0
                add_edge!(G, Int(i)+1, Int(j)+1)
                λ[Int(i)+1, Int(j)+1, :] .= λᵢⱼ
            end
        end

        # generate observations at the last time
        obsmat = ones(Int8, NV, T+1) * Int8(-1)
        for (τₗ, iₗ, Oₗ) in zip(obs_df.time, obs_df.node, obs_df.obs_st)
            obsmat[iₗ,τₗ] = Int8(Oₗ)
        end

        # define unobserved nodes
        unobs_nodes = Vector(1:NV)
        filter!(e->!(e in obs_df.node), unobs_nodes)
        confs_unobs = Int8.(confs[1,2,unobs_nodes])

        obs_df,contacts,confs = nothing,nothing,nothing

        for (ε_autoinf, maxiter, damp, μ_cutoff, fpn) in Iterators.product(ε_autoinfs, n_iters, damps, μ_cutoffs, fpns)
            # Define the epidemic model
            infectionmodel = SI(ε_autoinf, NV, T)
            obsprob(Ob, x) = obsprob_fp_fn(Ob, x; fp=fpn, fn=fpn)
            model = EpidemicModel(infectionmodel, G, T, log.(1 .- λ), obsmat)

            # run the inference
            Random.seed!(rng, 1234)
            try
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            catch
                write_err("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn", err_file)
                
                μ_cutoff /= 10
                Random.seed!(rng, 1234)
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            end
            
            #if model.converged
                margs = [node.marg.m[2,T+1] for node in nodes]
                nodes = nothing

                (_, _, auc) = ROC_curve(margs[unobs_nodes], confs_unobs)
                margs = nothing
                
                if auc > maxAUC
                    maxAUC = auc
                    println("s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, AUC: $auc")
                    write_log("s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, AUC: $auc", logfile)
                    save_function(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, 0, 0.0, auc, model.converged, save_vec, save_file)
                end
                #=
                continue
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
                
                margs = [node.marg.m[2,T+1] for node in nodes_nc]
                nodes_nc = nothing

                (_, _, auc) = ROC_curve(margs[unobs_nodes], confs_unobs)
                margs = nothing

                if auc > maxAUC
                    maxAUC = auc
                    println("s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc, AUC: $auc")
                    write_log("s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc, AUC: $auc", logfile)
                    save_function(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, n_iter_nc, damp_nc, auc, false, save_vec, save_file)
                end
            end
            nodes = nothing
            =#
        end
    end
end
=#






################################################################################
#####################       Varying nobs      ##################################
################################################################################
# check if data directory exists
isdir("logs") || mkdir("logs")
isdir("logs/test_SI") || mkdir("logs/test_SI")
isdir("logs/test_SI/var_obs") || mkdir("logs/test_SI/var_obs")
isdir("logs/test_SI/var_obs/data_rand_geom") || mkdir("logs/test_SI/var_obs/data_rand_geom")

##################### Random Geometric Graphs ##################################
n_sample = 50
sims = collect(1:n_sample)

# Parameters
T = 15
NV = 300
lmax = round(sqrt(1.8/NV), digits=3)
n_sr = 2
γ = n_sr/NV
λ₀ = 0.07
fracs_obs = collect(range(.05, .95, length=10))

isdir("logs/test_SI/var_obs/data_rand_geom/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_lamb-$(λ₀)") || mkdir("logs/test_SI/var_obs/data_rand_geom/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_lamb-$(λ₀)")

# computational constants
epsconv = 1e-5 # convergence threshold
ε_autoinfs = [1e-5, 1e-10, 1e-15, 1e-20]
n_iters = [500, 1000, 3000]
damps = [0.0, 0.2, 0.5, 0.7] # damping factor
μ_cutoffs = [-10., -100.] # cutoff for convergence
fpns = [0.0, 1e-9, 1e-15] # false positive and false negative rates
n_iter_ncs = [0]
damp_ncs = [0.0]


# Random number generator
seed = 1234
rng = Xoshiro(seed)

for frac_obs in fracs_obs
    nobs = floor(Int, frac_obs*NV)
    println("nobs = $nobs")
    logs_path = "logs/test_SI/var_obs/data_rand_geom/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_lamb-$(λ₀)/nobs_$(nobs)"
    isdir(logs_path) || mkdir(logs_path)
    logfile = logs_path*"/optim_pars_rand_geom.log"
    save_file = logs_path*"/optim_pars_rand_geom.jld2"
    err_file = logs_path*"/optim_pars_rand_geom.err"

    println("Starting random geometric graph inference")
    write_log("Starting random geometric graph inference", logfile)
    write_err("Starting random geometric graph inference", err_file)
    
    # initialising the csv file
    save_vec = init_save_rg(n_sample, save_file)
    
    for s in sims

        maxAUC = save_vec[find_s_index(s,save_vec)].AUC
        
        # Load the data
        path = "data/test_SI/var_obs/data_rand_geom/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_lamb-$(λ₀)/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_lam-$(λ₀)_s-$(s)"
        nobs_path = "data/test_SI/var_obs/data_rand_geom/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_lamb-$(λ₀)/nobs_$(nobs)/N-$(NV)_lmax-$(lmax)_tlim-$(T)_nsrc-$(n_sr)_lam-$(λ₀)_s-$(s)_obs_sparse.csv"
        contacts = npzread(path*"_contacts.npy")
        confs = npzread(path*"_confs.npy")
        obs_df = CSV.read(nobs_path, DataFrame)
        obs_df.node .+= 1
        obs_df.time .+= 1

        # create the graph
        λ = zeros(NV, NV, T)
        G = SimpleGraph(NV, 0)

        for (t,i,j,λᵢⱼ) in eachrow(contacts)
            if λᵢⱼ != λ₀
                throw(ArgumentError("λᵢⱼ ≠ λ₀"))
            end
            if t==0.0
                add_edge!(G, Int(i)+1, Int(j)+1)
                λ[Int(i)+1, Int(j)+1, :] .= λᵢⱼ
            end
        end

        # generate observations at the last time
        obsmat = ones(Int8, NV, T+1) * Int8(-1)
        for (τₗ, iₗ, Oₗ) in zip(obs_df.time, obs_df.node, obs_df.obs_st)
            obsmat[iₗ,τₗ] = Int8(Oₗ)
        end

        # define unobserved nodes
        unobs_nodes = Vector(1:NV)
        filter!(e->!(e in obs_df.node), unobs_nodes)
        confs_unobs = Int8.(confs[1,2,unobs_nodes])

        obs_df,contacts,confs = nothing,nothing,nothing

        for (ε_autoinf, maxiter, damp, μ_cutoff, fpn) in Iterators.product(ε_autoinfs, n_iters, damps, μ_cutoffs, fpns)
            # Define the epidemic model
            infectionmodel = SI(ε_autoinf, NV, T)
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
            
            #if model.converged
                margs = [node.marg.m[2,T+1] for node in nodes]
                nodes = nothing

                (_, _, auc) = ROC_curve(margs[unobs_nodes], confs_unobs)
                margs = nothing
                
                if auc > maxAUC
                    maxAUC = auc
                    println("s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, AUC: $auc")
                    write_log("s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, AUC: $auc", logfile)
                    save_function(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, 0, 0.0, auc, model.converged, save_vec, save_file)
                end
            #=
                continue
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
                
                margs = [node.marg.m[2,T+1] for node in nodes_nc]
                nodes_nc = nothing

                (_, _, auc) = ROC_curve(margs[unobs_nodes], confs_unobs)
                margs = nothing

                if auc > maxAUC
                    maxAUC = auc
                    println("s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc, AUC: $auc")
                    write_log("s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc, AUC: $auc", logfile)
                    save_function(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, n_iter_nc, damp_nc, auc, false, save_vec, save_file)
                end
            end
            nodes = nothing
            =#
        end
    end
end
