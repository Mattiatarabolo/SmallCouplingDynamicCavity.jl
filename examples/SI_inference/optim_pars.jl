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

    SaveStruct(s) = new(s, 0.0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
    SaveStruct(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, n_iter_nc, damp_nc, auc) = new(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, n_iter_nc, damp_nc, auc)
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
function save_function(s,ε_autoinf,maxiter,damp,μ_cutoff,fpn,n_iter_nc,damp_nc,auc,save_vec,save_file)
    save_vec[find_s_index(s,save_vec)] = SaveStruct(s,ε_autoinf,maxiter,damp,μ_cutoff,fpn,n_iter_nc,damp_nc,auc)
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



################################################################################
##################### Random Geometric Graphs ##################################
################################################################################
function optim_pars_rand_geom()
    logfile = "optim_pars_rand_geom.log"
    save_file = "optim_pars_rand_geom.jld2"
    err_file = "optim_pars_rand_geom.err"

    println("Starting random geometric graph inference")
    write_log("Starting random geometric graph inference", logfile)
    write_err("Starting random geometric graph inference", err_file)
    
    sims = colect(1:100)
    n_sample = length(sims)
    
    # initialising the csv file
    save_vec = init_save_rg(100, save_file)

    T = 28
    NV = 600
    γ = 2/NV
    λ₀ = 0.08

    # computational constants
    epsconv = 1e-5 # convergence threshold

    ε_autoinfs = [1e-10, 5e-13, 1e-15]
    n_iters = [[100,500,2000],[300,1000,5000]]
    damps = [[0.0,0.5,0.9],[0.1, 0.25, 0.5],[0.3,0.5,0.7]] # damping factor
    μ_cutoffs = [-1e2] # cutoff for convergence
    fpns = [0.0, 1e-8, 1e-12, 1e-15] # false positive and false negative rates
    n_iter_ncs = [0, 10, 50, 100, 500]
    damp_ncs = [0.0, 0.25, 0.5, 0.75, 0.9]

    for s in sims
        maxAUC = save_vec[find_s_index(s,save_vec)].AUC

        # Load the data
        path = "data/data_rand_geom/psym0_inf80_rnd_geom_n_600_d_10_tlim_28_lam_0.08_mu_0_s_$(s)_pe_1.0_nsrc_2"
        contacts = npzread(path*"_contacts.npy")
        confs = npzread(path*"_confs.npy")
        obs_df = CSV.read(path*"_0_obs_sparse.csv", DataFrame)
        obs_df.node .+= 1
        obs_df.time .+= 1

        # create the graph
        λ = zeros(NV, NV, T)
        G = SimpleGraph(NV, 0)

        for (t,i,j,λᵢⱼ) in eachrow(contacts)
            if t==0.0
                add_edge!(G, Int(i)+1, Int(j)+1)
                λ[Int(i)+1, Int(j)+1, :] .+= λᵢⱼ
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
            rng = Xoshiro(1234)
            try
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            catch
                println("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn")
                write_err("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn", err_file)
                
                μ_cutoff /= 10
                rng = Xoshiro(1234)
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            end

            for (n_iter_nc, damp_nc) in Iterators.product(n_iter_ncs, damp_ncs)
                nodes_nc = deepcopy(nodes)
                rng = Xoshiro(1234)
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
                    save_function(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, n_iter_nc, damp_nc, auc, save_vec, save_file)
                end
            end
            nodes = nothing
        end
    end
end




################################################################################
############################ Watts-Strogatz Graphs #############################
################################################################################
function optim_pars_watts_strogatz()
    logfile = "optim_pars_watts_strogatz.log"
    save_file = "optim_pars_watts_strogatz.jld2"
    err_file = "optim_pars_watts_strogatz.err"

    println("Starting Watts-Strogatz graph inference")
    write_log("Starting Watts-Strogatz graph inference", logfile)
    write_err("Starting Watts-Strogatz graph inference", err_file)
    #[0, 1, 2, 4, 9, 11, 13, 16, 17, 23, 28, 30, 31, 39, 50, 51, 53, 54, 55, 57, 59, 61, 65, 72, 76, 78, 82,
    sims = collect(0:100) 
    n_sample = length(sims)
    
    # initialising the csv file
    save_vec = init_save(101, save_file)

    T = 25
    NV = 600
    γ = 1/NV
    λ₀ = 0.16

    # computational constants
    epsconv = 1e-5 # convergence threshold

    ε_autoinfs = [1e-10, 1e-15]
    n_iters = [5, 50, 500]
    damps = [0.0,0.5,0.9] # damping factor
    μ_cutoffs = [-1e2] # cutoff for convergence
    fpns = [0.0, 1e-10, 1e-15] # false positive and false negative rates
    n_iter_ncs = [0, 10, 50, 100, 500]
    damp_ncs = [0.0, 0.25, 0.5, 0.75, 0.9]

    for s in sims
        maxAUC = save_vec[find_s_index(s,save_vec)].AUC

        # Load the data
        path = "data/data_WS/20rnd_pr0_WS_n_600_d_4_tlim_25_lam_0.16_mu_0_s_$(s)_pe_1.0"
        contacts = npzread(path*"_contacts.npy")
        confs = npzread(path*"_confs.npy")
        obs_df = CSV.read(path*"_0_obs_sparse.csv", DataFrame)
        obs_df.node .+= 1
        obs_df.time .+= 1

        # create the graph
        λ = zeros(NV, NV, T)
        G = SimpleGraph(NV, 0)
        
        for (t,i,j,λᵢⱼ) in eachrow(contacts)
            if t==0.0
                add_edge!(G, Int(i)+1, Int(j)+1)
                λ[Int(i)+1, Int(j)+1, :] .+= λᵢⱼ
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
            rng = Xoshiro(1234)
            try
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            catch
                println("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn")
                write_err("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn", err_file)
                
                μ_cutoff /= 10
                rng = Xoshiro(1234)
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            end

            for (n_iter_nc, damp_nc) in Iterators.product(n_iter_ncs, damp_ncs)
                nodes_nc = deepcopy(nodes)
                rng = Xoshiro(1234)
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
                    save_function(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, n_iter_nc, damp_nc, auc, save_vec, save_file)
                end
            end
            nodes = nothing
        end
    end
end





################################################################################
########################### OpenABM contact network ############################
################################################################################
function optim_pars_openabm()
    logfile = "optim_pars_openabm.log"
    save_file = "optim_pars_openabm.jld2"
    err_file = "optim_pars_openabm.err"

    println("Starting OpenABM contact network inference")
    write_log("Starting OpenABM contact network inference", logfile)
    write_err("Starting OpenABM contact network inference", err_file)
    
    sims = collect(0:99) 
    n_sample = length(sims)
    
    # initialising the csv file
    save_vec = init_save(100, save_file)

    T = 21
    NV = 1994
    γ = 2/NV

    # computational constants
    epsconv = 1e-5 # convergence threshold

    ε_autoinfs = [1e-10, 1e-15]
    n_iters = [5, 50, 500]
    damps = [0.0,0.5,0.9] # damping factor
    μ_cutoffs = [-1e2] # cutoff for convergence
    fpns = [0.0, 1e-10, 1e-15] # false positive and false negative rates
    n_iter_ncs = [0, 10, 50, 100, 500]
    damp_ncs = [0.0, 0.25, 0.5, 0.75, 0.9]

    for s in sims
        maxAUC = save_vec[find_s_index(s,save_vec)].AUC

        # Load the data
        path = "data/data_openABM/60rnd_pr0_data_gamma_n_1994_d_10_tlim_21_lam_0.5_mu_0_s_6_pe_1.0_nsrc_2"
        contacts = npzread(path*"_contacts.npy")
        confs = npzread(path*"_confs.npy")
        obs_df = CSV.read(path*"_$(s)_obs_sparse.csv", DataFrame)
        obs_df.node .+= 1
        obs_df.time .+= 1

        # create the graph
        G = [SimpleGraph(NV,0) for _ in 0:T-1]
        λ = zeros(NV, NV, T)

        for (t,i,j,λᵢⱼ) in eachrow(contacts)
            add_edge!(G[Int(t)+1], Int(i)+1, Int(j)+1)
            λ[Int(i)+1, Int(j)+1, Int(t)+1] += λᵢⱼ
        end

        # generate observations at the last time
        obsmat = ones(Int8, NV, T+1) * Int8(-1)
        for (τₗ, iₗ, Oₗ) in zip(obs_df.time, obs_df.node, obs_df.obs_st)
            obsmat[iₗ,τₗ] = Int8(Oₗ)
        end

        # define unobserved nodes
        unobs_nodes = Vector(1:NV)
        filter!(e->!(e in obs_df.node), unobs_nodes)
        confs_unobs = Int8.(confs[s+1,2,unobs_nodes])

        obs_df,contacts,confs = nothing,nothing,nothing

        for (ε_autoinf, maxiter, damp, μ_cutoff, fpn) in Iterators.product(ε_autoinfs, n_iters, damps, μ_cutoffs, fpns)
            # Define the epidemic model
            infectionmodel = SI(ε_autoinf, NV, T)
            obsprob(Ob, x) = obsprob_fp_fn(Ob, x; fp=fpn, fn=fpn)
            model = EpidemicModel(infectionmodel, G, T, log.(1 .- λ), obsmat)

            # run the inference
            rng = Xoshiro(1234)
            try
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            catch
                println("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn")
                write_err("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn", err_file)
                
                μ_cutoff /= 10
                rng = Xoshiro(1234)
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            end

            for (n_iter_nc, damp_nc) in Iterators.product(n_iter_ncs, damp_ncs)
                nodes_nc = deepcopy(nodes)
                rng = Xoshiro(1234)
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
                    save_function(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, n_iter_nc, damp_nc, auc, save_vec, save_file)
                end
            end
            nodes = nothing
        end
    end
end





################################################################################
############################ Covasim contact network ###########################
################################################################################
function optim_pars_covasim()
    logfile = "optim_pars_covasim.log"
    save_file = "optim_pars_covasim.jld2"
    err_file = "optim_pars_covasim.err"

    println("Starting Covasim contact network inference")
    write_log("Starting Covasim contact network inference", logfile)
    write_err("Starting Covasim contact network inference", err_file)
    
    sims = collect(0:99) #sample(0:99, n_sample; replace=false, ordered=true)
    n_sample = length(sims)
    
    # initialising the csv file
    save_vec = init_save(100, save_file)

    T = 24
    NV = 1000
    γ = 2/NV

    # computational constants
    epsconv = 1e-5 # convergence threshold
    
    ε_autoinfs = [1e-15]
    n_iters = [50, 100]
    damps = [0.0,0.5,0.9] # damping factor
    μ_cutoffs = [-1e4] # cutoff for convergence
    fpns = [0.0, 1e-10, 1e-15] # false positive and false negative rates
    n_iter_ncs = [0, 10, 50]
    damp_ncs = [0.0, 0.25, 0.5, 0.75, 0.9]

    for s in sims
        maxAUC = save_vec[find_s_index(s,save_vec)].AUC

        # Load the data
        path = "data/data_Covasim/to_24_psym0_data_n_1000_d_10_tlim_24_lam_0.5_mu_0_s_8_pe_1.0_nsrc_2"
        contacts = npzread(path*"_contacts.npy")
        confs = npzread(path*"_confs.npy")
        obs_df = CSV.read(path*"_$(s)_obs_sparse.csv", DataFrame)
        obs_df.node .+= 1
        obs_df.time .+= 1

        # create the graph
        G = [SimpleGraph(NV,0) for _ in 0:T-1]
        λ = zeros(NV, NV, T)

        for (t,i,j,λᵢⱼ) in eachrow(contacts)
            add_edge!(G[Int(t)+1], Int(i)+1, Int(j)+1)
            λ[Int(i)+1, Int(j)+1, Int(t)+1] += λᵢⱼ
        end

        # generate observations at the last time
        obsmat = ones(Int8, NV, T+1) * Int8(-1)
        for (τₗ, iₗ, Oₗ) in zip(obs_df.time, obs_df.node, obs_df.obs_st)
            obsmat[iₗ,τₗ] = Int8(Oₗ)
        end

        # define unobserved nodes
        unobs_nodes = Vector(1:NV)
        filter!(e->!(e in obs_df.node), unobs_nodes)
        confs_unobs = Int8.(confs[s+1,2,unobs_nodes])

        obs_df,contacts,confs = nothing,nothing,nothing

        for (ε_autoinf, maxiter, damp, μ_cutoff, fpn) in Iterators.product(ε_autoinfs, n_iters, damps, μ_cutoffs, fpns)
            # Define the epidemic model
            infectionmodel = SI(ε_autoinf, NV, T)
            obsprob(Ob, x) = obsprob_fp_fn(Ob, x; fp=fpn, fn=fpn)
            model = EpidemicModel(infectionmodel, G, T, log.(1 .- λ), obsmat)

            # run the inference
            try
                rng = Xoshiro(1234)
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            catch
                println("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn")
                write_err("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn", err_file)
                
                μ_cutoff /= 10
                rng = Xoshiro(1234)
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            end

            for (n_iter_nc, damp_nc) in Iterators.product(n_iter_ncs, damp_ncs)
                nodes_nc = deepcopy(nodes)
                try
                    rng = Xoshiro(1234)
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
                    save_function(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, n_iter_nc, damp_nc, auc, save_vec, save_file)
                end
            end
            nodes = nothing
        end
    end
end






################################################################################
######################## Office12 contact network  #############################
################################################################################
function optim_pars_office12()
    logfile = "optim_pars_office12.log"
    save_file = "optim_pars_office12.jld2"
    err_file = "optim_pars_office12.err"

    println("Starting Office12 contact network inference")
    write_log("Starting Office12 contact network inference", logfile)
    write_err("Starting Office12 contact network inference", err_file)
    
    sims = collect(0:99)
    n_sample = length(sims)
    
    # initialising the csv file
    save_vec = init_save(100, save_file)

    T = 12
    NV = 219
    γ = 2/NV

    # computational constants
    epsconv = 1e-5 # convergence threshold

    ε_autoinfs = [1e-10, 1e-15]
    n_iters = [5, 50, 100]
    damps = [0.0,0.5,0.9] # damping factor
    μ_cutoffs = [-1e4] # cutoff for convergence
    fpns = [0.0, 1e-10, 1e-15] # false positive and false negative rates
    n_iter_ncs = [0, 10, 50]
    damp_ncs = [0.0, 0.25, 0.5, 0.75, 0.9]

    for s in sims
        maxAUC = save_vec[find_s_index(s,save_vec)].AUC

        # Load the data
        path = "data/data_work_12/15rnd_pr0_data_gamma_n_219_d_10_tlim_12_lam_0.1_mu_0_s_6_pe_1.0_nsrc_2"
        contacts = npzread(path*"_contacts.npy")
        confs = npzread(path*"_confs.npy")
        obs_df = CSV.read(path*"_$(s)_obs_sparse.csv", DataFrame)
        obs_df.node .+= 1
        obs_df.time .+= 1

        # create the graph
        G = [SimpleGraph(NV,0) for _ in 0:T-1]
        λ = zeros(NV, NV, T)

        for (t,i,j,λᵢⱼ) in eachrow(contacts)
            add_edge!(G[Int(t)+1], Int(i)+1, Int(j)+1)
            λ[Int(i)+1, Int(j)+1, Int(t)+1] += λᵢⱼ
        end

        # generate observations at the last time
        obsmat = ones(Int8, NV, T+1) * Int8(-1)
        for (τₗ, iₗ, Oₗ) in zip(obs_df.time, obs_df.node, obs_df.obs_st)
            obsmat[iₗ,τₗ] = Int8(Oₗ)
        end

        # define unobserved nodes
        unobs_nodes = Vector(1:NV)
        filter!(e->!(e in obs_df.node), unobs_nodes)
        confs_unobs = Int8.(confs[s+1,2,unobs_nodes])

        obs_df,contacts,confs = nothing,nothing,nothing

        for (ε_autoinf, maxiter, damp, μ_cutoff, fpn) in Iterators.product(ε_autoinfs, n_iters, damps, μ_cutoffs, fpns)
            # Define the epidemic model
            infectionmodel = SI(ε_autoinf, NV, T)
            obsprob(Ob, x) = obsprob_fp_fn(Ob, x; fp=fpn, fn=fpn)
            model = EpidemicModel(infectionmodel, G, T, log.(1 .- λ), obsmat)

            # run the inference
            try
                rng = Xoshiro(1234)
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            catch
                println("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn")
                write_err("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn", err_file)
                
                μ_cutoff /= 10
                rng = Xoshiro(1234)
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            end

            for (n_iter_nc, damp_nc) in Iterators.product(n_iter_ncs, damp_ncs)
                nodes_nc = deepcopy(nodes)
                try
                    rng = Xoshiro(1234)
                    run_SCDC!(nodes_nc, model, γ, 0, epsconv, 0.0; μ_cutoff=μ_cutoff, n_iter_nc=n_iter_nc, damp_nc=damp_nc, rng=rng)
                catch
                    println("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc")
                    write_err("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, $damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc", err_file)
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
                    save_function(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, n_iter_nc, damp_nc, auc, save_vec, save_file)
                end
            end
            nodes = nothing
        end
    end
end








################################################################################
###################### Office24 contact network ################################
################################################################################
function optim_pars_office24()
    logfile = "optim_pars_office24.log"
    save_file = "optim_pars_office24.jld2"
    err_file = "optim_pars_office24.err"

    println("Starting Office24 contact network inference")
    write_log("Starting Office24 contact network inference", logfile)
    write_err("Starting Office24 contact network inference", err_file)
    
    sims = collect(0:99)
    n_sample = length(sims)
    
    # initialising the csv file
    save_vec = init_save(100, save_file)

    T = 24
    NV = 219
    γ = 2/NV

    # computational constants
    epsconv = 1e-5 # convergence threshold

    ε_autoinfs = [1e-10, 1e-15]
    n_iters = [5, 50, 100]
    damps = [0.0,0.5,0.9] # damping factor
    μ_cutoffs = [-1e4] # cutoff for convergence
    fpns = [0.0, 1e-10, 1e-15] # false positive and false negative rates
    n_iter_ncs = [0, 10, 50]
    damp_ncs = [0.0, 0.25, 0.5, 0.75, 0.9]

    for s in sims
        maxAUC = save_vec[find_s_index(s,save_vec)].AUC

        # Load the data
        path = "data/data_work_24/15rnd_pr0_data_gamma_n_219_d_10_tlim_24_lam_0.1_mu_0_s_6_pe_1.0_nsrc_2"
        contacts = npzread(path*"_contacts.npy")
        confs = npzread(path*"_confs.npy")
        obs_df = CSV.read(path*"_$(s)_obs_sparse.csv", DataFrame)
        obs_df.node .+= 1
        obs_df.time .+= 1

        # create the graph
        G = [SimpleGraph(NV,0) for _ in 0:T-1]
        λ = zeros(NV, NV, T)

        for (t,i,j,λᵢⱼ) in eachrow(contacts)
            add_edge!(G[Int(t)+1], Int(i)+1, Int(j)+1)
            λ[Int(i)+1, Int(j)+1, Int(t)+1] += λᵢⱼ
        end

        # generate observations at the last time
        obsmat = ones(Int8, NV, T+1) * Int8(-1)
        for (τₗ, iₗ, Oₗ) in zip(obs_df.time, obs_df.node, obs_df.obs_st)
            obsmat[iₗ,τₗ] = Int8(Oₗ)
        end

        # define unobserved nodes
        unobs_nodes = Vector(1:NV)
        filter!(e->!(e in obs_df.node), unobs_nodes)
        confs_unobs = Int8.(confs[s+1,2,unobs_nodes])

        obs_df,contacts,confs = nothing,nothing,nothing

        for (ε_autoinf, maxiter, damp, μ_cutoff, fpn) in Iterators.product(ε_autoinfs, n_iters, damps, μ_cutoffs, fpns)
            # Define the epidemic model
            infectionmodel = SI(ε_autoinf, NV, T)
            obsprob(Ob, x) = obsprob_fp_fn(Ob, x; fp=fpn, fn=fpn)
            model = EpidemicModel(infectionmodel, G, T, log.(1 .- λ), obsmat)

            # run the inference
            try
                rng = Xoshiro(1234)
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            catch
                println("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn")
                write_err("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn", err_file)
                
                μ_cutoff /= 10
                rng = Xoshiro(1234)
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            end

            for (n_iter_nc, damp_nc) in Iterators.product(n_iter_ncs, damp_ncs)
                nodes_nc = deepcopy(nodes)
                try
                    rng = Xoshiro(1234)
                    run_SCDC!(nodes_nc, model, γ, 0, epsconv, 0.0; μ_cutoff=μ_cutoff, n_iter_nc=n_iter_nc, damp_nc=damp_nc, rng=rng)
                catch
                    println("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc")
                    write_err("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, $damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc", err_file)
                    return nothing
                end
                
                margs = [node.marg.m[2,T+1] for node in nodes_nc]
                nodes_nc = nothing

                (_, _, auc) = ROC_curve(margs[unobs_nodes], confs_unobs)
                margs = nothing
                
                if auc > maxAUC
                    maxAUC = auc
                    println("s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc, AUC: $auc")
                    write_log("s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, $damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc, AUC: $auc", logfile)
                    save_function(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, n_iter_nc, damp_nc, auc, save_vec, save_file)
                end
            end
            nodes = nothing
        end
    end
end






################################################################################
####################### School18 contact network ###############################
################################################################################
function optim_pars_school18()
    logfile = "optim_pars_school18.log"
    save_file = "optim_pars_school18.jld2"
    err_file = "optim_pars_school18.err"

    println("Starting School18 contact network inference")
    write_log("Starting School18 contact network inference", logfile)
    write_err("Starting School18 contact network inference", err_file)
    
    sims = collect(0:99)
    n_sample = length(sims)
    
    # initialising the csv file
    save_vec = init_save(100, save_file)

    T = 18
    NV = 328
    γ = 2/NV

    # computational constants
    epsconv = 1e-5 # convergence threshold

    ε_autoinfs = [1e-10, 1e-15]
    n_iters = [5, 50, 100]
    damps = [0.0,0.5,0.9] # damping factor
    μ_cutoffs = [-1e4] # cutoff for convergence
    fpns = [0.0, 1e-10, 1e-15] # false positive and false negative rates
    n_iter_ncs = [0, 10, 50]
    damp_ncs = [0.0, 0.25, 0.5, 0.75, 0.9]

    for s in sims
        maxAUC = save_vec[find_s_index(s,save_vec)].AUC

        # Load the data
        path = "data/data_school_18/to_18_ps0_inf30_data_gamma_n_328_d_10_tlim_18_lam_0.5_mu_0_s_6_pe_1.0_nsrc_2"
        contacts = npzread(path*"_contacts.npy")
        confs = npzread(path*"_confs.npy")
        obs_df = CSV.read(path*"_$(s)_obs_sparse.csv", DataFrame)
        obs_df.node .+= 1
        obs_df.time .+= 1

        # create the graph
        G = [SimpleGraph(NV,0) for _ in 0:T-1]
        λ = zeros(NV, NV, T)

        for (t,i,j,λᵢⱼ) in eachrow(contacts)
            add_edge!(G[Int(t)+1], Int(i)+1, Int(j)+1)
            λ[Int(i)+1, Int(j)+1, Int(t)+1] += λᵢⱼ
        end
            
        # generate observations at the last time
        obsmat = ones(Int8, NV, T+1) * Int8(-1)
        for (τₗ, iₗ, Oₗ) in zip(obs_df.time, obs_df.node, obs_df.obs_st)
            obsmat[iₗ,τₗ] = Int8(Oₗ)
        end

        # define unobserved nodes
        unobs_nodes = Vector(1:NV)
        filter!(e->!(e in obs_df.node), unobs_nodes)
        confs_unobs = Int8.(confs[s+1,2,unobs_nodes])

        obs_df,contacts,confs = nothing,nothing,nothing

        for (ε_autoinf, maxiter, damp, μ_cutoff, fpn) in Iterators.product(ε_autoinfs, n_iters, damps, μ_cutoffs, fpns)
            # Define the epidemic model
            infectionmodel = SI(ε_autoinf, NV, T)
            obsprob(Ob, x) = obsprob_fp_fn(Ob, x; fp=fpn, fn=fpn)
            model = EpidemicModel(infectionmodel, G, T, log.(1 .- λ), obsmat)

            # run the inference
            try
                rng = Xoshiro(1234)
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            catch
                println("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn")
                write_err("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn", err_file)
                
                μ_cutoff /= 10
                rng = Xoshiro(1234)
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            end

            for (n_iter_nc, damp_nc) in Iterators.product(n_iter_ncs, damp_ncs)
                nodes_nc = deepcopy(nodes)
                try
                    rng = Xoshiro(1234)
                    run_SCDC!(nodes_nc, model, γ, 0, epsconv, 0.0; μ_cutoff=μ_cutoff, n_iter_nc=n_iter_nc, damp_nc=damp_nc, rng=rng)
                catch
                    println("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc")
                    write_err("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, $damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc", err_file)
                    return nothing
                end

                margs = [node.marg.m[2,T+1] for node in nodes_nc]
                nodes_nc = nothing

                (_, _, auc) = ROC_curve(margs[unobs_nodes], confs_unobs)
                margs = nothing

                if auc > maxAUC
                    maxAUC = auc
                    println("s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc, AUC: $auc")
                    write_log("s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, $damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc, AUC: $auc", logfile)
                    save_function(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, n_iter_nc, damp_nc, auc, save_vec, save_file)
                end
            end
            nodes = nothing
        end
    end
end






################################################################################
######################## School36 contact network ##############################
################################################################################
function optim_pars_school36()
    logfile = "optim_pars_school36.log"
    save_file = "optim_pars_school36.jld2"
    err_file = "optim_pars_school36.err"

    println("Starting School36 contact network inference")
    write_log("Starting School36 contact network inference", logfile)
    write_err("Starting School36 contact network inference", err_file)
    
    sims = collect(0:99)
    n_sample = length(sims)
    
    # initialising the csv file
    save_vec = init_save(100, save_file)

    T = 36
    NV = 328
    γ = 2/NV

    # computational constants
    epsconv = 1e-5 # convergence threshold

    ε_autoinfs = [1e-10, 1e-15]
    n_iters = [5, 50, 100]
    damps = [0.0,0.5,0.9] # damping factor
    μ_cutoffs = [-1e4] # cutoff for convergence
    fpns = [0.0, 1e-10, 1e-15] # false positive and false negative rates
    n_iter_ncs = [0, 10, 50, 100, 500]
    damp_ncs = [0.0, 0.25, 0.5, 0.75, 0.9]

    for s in sims
        maxAUC = save_vec[find_s_index(s,save_vec)].AUC

        # Load the data
        path = "data/data_school_36/to_36_ps0_inf30_data_gamma_n_328_d_10_tlim_36_lam_0.5_mu_0_s_6_pe_1.0_nsrc_2"
        contacts = npzread(path*"_contacts.npy")
        confs = npzread(path*"_confs.npy")
        obs_df = CSV.read(path*"_$(s)_obs_sparse.csv", DataFrame)
        obs_df.node .+= 1
        obs_df.time .+= 1

        # create the graph
        G = [SimpleGraph(NV,0) for _ in 0:T-1]
        λ = zeros(NV, NV, T)

        for (t,i,j,λᵢⱼ) in eachrow(contacts)
            add_edge!(G[Int(t)+1], Int(i)+1, Int(j)+1)
            λ[Int(i)+1, Int(j)+1, Int(t)+1] += λᵢⱼ
        end

        # generate observations at the last time
        obsmat = ones(Int8, NV, T+1) * Int8(-1)
        for (τₗ, iₗ, Oₗ) in zip(obs_df.time, obs_df.node, obs_df.obs_st)
            obsmat[iₗ,τₗ] = Int8(Oₗ)
        end

        # define unobserved nodes
        unobs_nodes = Vector(1:NV)
        filter!(e->!(e in obs_df.node), unobs_nodes)
        confs_unobs = Int8.(confs[s+1,2,unobs_nodes])

        obs_df,contacts,confs = nothing,nothing,nothing

        for (ε_autoinf, maxiter, damp, μ_cutoff, fpn) in Iterators.product(ε_autoinfs, n_iters, damps, μ_cutoffs, fpns)
            # Define the epidemic model
            infectionmodel = SI(ε_autoinf, NV, T)
            obsprob(Ob, x) = obsprob_fp_fn(Ob, x; fp=fpn, fn=fpn)
            model = EpidemicModel(infectionmodel, G, T, log.(1 .- λ), obsmat)

            # run the inference
            try
                rng = Xoshiro(1234)
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            catch
                println("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn")
                write_err("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, $μ_cutoff: $μ_cutoff, fpn: $fpn", err_file)
                
                μ_cutoff /= 10
                rng = Xoshiro(1234)
                nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp; μ_cutoff=μ_cutoff, rng=rng)
            end

            for (n_iter_nc, damp_nc) in Iterators.product(n_iter_ncs, damp_ncs)
                nodes_nc = deepcopy(nodes)
                try
                    rng = Xoshiro(1234)
                    run_SCDC!(nodes_nc, model, γ, 0, epsconv, 0.0; μ_cutoff=μ_cutoff, n_iter_nc=n_iter_nc, damp_nc=damp_nc, rng=rng)
                catch
                    println("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc")
                    write_err("ERROR at s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, $damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc", err_file)
                    return nothing
                end

                margs = [node.marg.m[2,T+1] for node in nodes_nc]
                nodes_nc = nothing

                (_, _, auc) = ROC_curve(margs[unobs_nodes], confs_unobs)
                margs = nothing

                if auc > maxAUC
                    maxAUC = auc
                    println("s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc, AUC: $auc")
                    write_log("s: $s, ε_autoinf: $ε_autoinf, maxiter: $maxiter, $damp: $damp, μ_cutoff: $μ_cutoff, fpn: $fpn, n_iter_nc: $n_iter_nc, damp_nc: $damp_nc, AUC: $auc", logfile)
                    save_function(s, ε_autoinf, maxiter, damp, μ_cutoff, fpn, n_iter_nc, damp_nc, auc, save_vec, save_file)
                end
            end
            nodes = nothing
        end
    end
end

