import SmallCouplingDynamicCavity as SCDC
using Graphs, Random, StatsBase
using ProgressMeter
using CSV, DataFrames, JLD2
using PyCall, SparseArrays
@pyimport sib


####### extract mean number of infected nodes ########
function n_inf(model::SCDC.EpidemicModel{SCDC.SI,TG}, nodes::Vector{SCDC.Node{SCDC.SI,TG}}) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    n_inf = zeros(model.T+1)
    @inbounds @fastmath for t in 1:model.T+1
        n_inf[t] = sum([inode.marg.m[2, t] for inode in nodes])
    end
    return n_inf
end

n_inf(X::Matrix{Int8}) = dropdims(sum(X,dims=1),dims=1)
function n_inf!(n_inf_vec::Vector{Float64},X::Matrix{Int8})
    N, T = size(X)
    @inbounds @fastmath for t in 1:T
        @inbounds @fastmath @simd for i in 1:N
            n_inf_vec[t] += X[i,t]
        end
    end
end

n_inf(out::Matrix{Float64}) = dropdims(sum(out,dims=1),dims=1)

#############  Mean-field  ####################
function run_fwd_IBMF(model::SCDC.EpidemicModel{SCDC.SI,TG}, λ::Array{Float64, 3}, prior::Vector{Float64}) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    # Format nodes
    nodes = SCDC.nodes_formatting(model)

    #initial marginals
    for inode in nodes
        inode.marg.m[2, 1] = prior[inode.i]
    end

    # update marginals
    @inbounds @fastmath for t in 1:model.T
        @inbounds @fastmath for inode in nodes
            prodmarg = 1.0
            @inbounds @fastmath @simd for j in inode.∂
                prodmarg *= 1 - λ[j, inode.i, t] * nodes[j].marg.m[2, t]
            end
            inode.marg.m[2, t+1] = inode.marg.m[2, t] + (1-inode.marg.m[2, t]) * (1 - prodmarg)
        end
    end
    return nodes
end


#################  SIB  ####################
function run_sibyl(
    N::Int, 
    T::Int, 
    λ_big::Array{Float64, 3},
    γ::Float64;
    dt::Float64 = 1/5,
    maxit::Int = 400, 
    tol::Float64 = 1e-14)

    T_bp = Int(round((T+1) / dt))

    Λ = [sparse(λ_big[:,:,t]) for t in 1:T+1]
    contacts = [(i-1,j-1,t,λ*dt) for t=0:T for (i,j,λ) in zip(findnz(Λ[t+1])...)];
    obs = [[(i,-1,t) for t=1:T_bp for i=0:N-1]; []]
    sort!(obs, lt=((i1,s1,t1),(i2,s2,t2))->(t1<t2))
    prob_sus = 0.5
    prob_seed = γ
    pseed = prob_seed / (2 - prob_seed)
    psus = prob_sus * (1 - pseed)
    params = sib.Params(prob_r=sib.Exponential(mu=0), pseed=pseed, psus=psus,pautoinf=0)
    f = sib.FactorGraph(contacts=contacts, observations=obs, params=params)
    sib.iterate(f, maxit=maxit,tol=tol)
    sib.iterate(f, maxit=maxit, damping=0.5, tol=tol)
    sib.iterate(f, maxit=maxit, damping=0.9, tol=tol)
    p_sib=[collect(n.bt) for n in f.nodes]
    m_sib = zeros(N, T_bp)
    for i=1:N
        m_sib[i,1] = p_sib[i][1]
        for t=2:T_bp
            m_sib[i,t] = m_sib[i,t-1] + p_sib[i][t]
        end
    end
    return m_sib
end


####################################  Sampling  ######################################
function run_samples(model::SCDC.EpidemicModel{SCDC.SI,TG}, γ::Float64, M::Int; reject::Bool=false) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    rng = Xoshiro(1234)
    X = zeros(Int8, model.N, model.T+1)
    n_inf_X = zeros(model.T+1)
    frac_inf = zeros(model.T+1)
    @showprogress for m in 1:M
        Random.seed!(rng, m+100)
        SCDC.sim_epidemics!(X, model; γ=γ, rng=rng, reject=reject)
        n_inf!(n_inf_X, X)
        frac_inf .+= n_inf_X
    end
    return frac_inf ./ M
end

# Chaotic map to generate a seed from m
function chaotic_map(m::Int)
    # A simple chaotic map: multiply, add a prime, and take modulo
    a, c, m_max =12845566510, 101002303453, 2^31 - 1
    return (a * m + c) % m_max
end

function run_samples_parallel(model::SCDC.EpidemicModel{SCDC.SI, TG}, γ::Float64, M::Int; reject::Bool=false) where {TG <: Union{<:AbstractGraph, Vector{<:AbstractGraph}}}
    # Generate unique rngs for reproducibility
    rngs = [Xoshiro() for _ in 1:Threads.nthreads()]  # Create one RNG per thread
    # Initialize variables
    frac_inf = zeros(model.T+1)
    # Lock for thread-safe incrementation
    my_lock = ReentrantLock()
    # Iterate
    Threads.@threads for m in 1:M
        thread_id = Threads.threadid()
        local_rng = rngs[thread_id]
        Random.seed!(local_rng, m)  # Thread-safe RNG with unique seed
        # Run epidemic simulation
        X = SCDC.sim_epidemics(model; γ=γ, rng=local_rng, reject=reject)
        n_inf_X = n_inf(X)
        # Update frac_inf
        lock(my_lock) do
            frac_inf .+= n_inf_X
        end
    end
    
    return frac_inf ./ M
end



#########################################  SCDCa forward dynamics  ############################################
function compute_fwd_prod(inode, t)
    prod = 1.0
    @inbounds @fastmath for (kindex, k) in enumerate(inode.∂)
        prod *= 1 - inode.cavities[kindex].m[t] * (1 - exp(inode.νs[kindex][t]))
    end
    return prod
end


function run_fwd_dynamicsSCDCa(model, prior)
    # Format nodes for inference
    nodes = SCDC.nodes_formatting(model)
    # Initialize message objects
    @inbounds @fastmath for inode in nodes
        inode.marg.m[2, 1] = prior[inode.i]
        @inbounds @fastmath for (jindex, j) in enumerate(inode.∂)
            inode.cavities[jindex].m[1] = prior[nodes[j].i]
        end
    end
    @inbounds @fastmath for t in 1:model.T
        @inbounds @fastmath for inode in nodes
            prod = compute_fwd_prod(inode, t)
            @inbounds @fastmath for (jindex, j) in enumerate(inode.∂)
                iindex = nodes[j].∂_idx[inode.i]
                nodes[j].cavities[iindex].m[t+1] = nodes[j].cavities[iindex].m[t] + (1-nodes[j].cavities[iindex].m[t])*(1 - prod / (1 - inode.cavities[jindex].m[t] * (1 - exp(inode.νs[jindex][t]))))
            end
            inode.marg.m[2,t+1] = inode.marg.m[2,t] + (1-inode.marg.m[2,t]) * (1-prod)
        end
    end
    return nodes
end


############################################################################################
#####################################  Bethe Lattice  ######################################
############################################################################################

function run_bethe(λ0vec, nsrc, T)
    NV = 485
    K = 4
    t_max = 5

    γ = nsrc/NV
    prior = ones(NV) * γ

    # Generate Bethe lattice
    V, E = SCDC.bethe_lattice(K, t_max)
    G = SimpleGraph(NV)
    for (i, j) in E
        add_edge!(G, i, j)
    end

    dict_res = Dict(λ0 => Dict(method=>zeros(T+1)  for method in ["MF";"EPI";"SIB";"sampling"]) for λ0 in λ0vec)

    for λ0 in λ0vec
        # constant infection probability
        λ = zeros(NV, NV, T+1)
        for e in edges(G)
            λ[src(e), dst(e), :] .+= λ0 
            λ[dst(e), src(e), :] .+= λ0
        end

        # define de epidemic model
        infectionmodel = SCDC.SI(0.0, NV, T)
        model = SCDC.EpidemicModel(infectionmodel, G, T, log.(1 .- λ[:,:,1:T]))

        println("running lambda = $λ0")
        out_sampling = run_samples_parallel(model, γ, 10000; reject=false)
        out_MF = run_fwd_IBMF(model, λ, prior)
        out_EPI = SCDC.run_fwd_dynamics(model, prior)
        out_EPIapprox = run_fwd_dynamicsSCDCa(model, prior)
        out_SIB = run_sibyl(NV, T, λ, γ; dt=1., maxit = 100, tol = 1e-10)
        
        dict_res[λ0]["MF"] = n_inf(model, out_MF)
        dict_res[λ0]["EPI"] = n_inf(model, out_EPI)
        dict_res[λ0]["EPIapprox"] = n_inf(model, out_EPIapprox)
        dict_res[λ0]["SIB"] = n_inf(out_SIB)
        dict_res[λ0]["sampling"] = out_sampling
    end

    isdir("res_lambda") || mkdir("res_lambda")
    JLD2.save("res_lambda/bethe_N-$(NV)_K-$(K)_tmax-$(t_max)_nsrc-$(nsrc).jld2", "res", dict_res, "K", K, "gamma", γ, "G", G, "lambdavec", λ0vec)
end


############################################################################################################
##############################################  RRG  #######################################################
############################################################################################################

function run_RRG(K, λ0vec, nsrc, T)
    NV = 500
    
    γ = nsrc/NV
    prior = ones(NV) * γ


    # Generate Bethe lattice
    rng = Xoshiro(1234)
    G = random_regular_graph(NV, K; rng=rng)

    dict_res = Dict(λ0 => Dict(method=>zeros(T+1)  for method in ["MF","EPI","EPIapprox","SIB","sampling"]) for λ0 in λ0vec)

    for λ0 in λ0vec
        # constant infection probability
        λ = zeros(NV, NV, T+1)
        for e in edges(G)
            λ[src(e), dst(e), :] .+= λ0 
            λ[dst(e), src(e), :] .+= λ0
        end

        # define de epidemic model
        infectionmodel = SCDC.SI(0.0, NV, T)
        model = SCDC.EpidemicModel(infectionmodel, G, T, log.(1 .- λ[:,:,1:T]))

        println("running lambda = $λ0")
        out_sampling = run_samples_parallel(model, γ, 10000; reject=false)
        out_MF = run_fwd_IBMF(model, λ, prior)
        out_EPI = SCDC.run_fwd_dynamics(model, prior)
        out_EPIapprox = run_fwd_dynamicsSCDCa(model, prior)
        out_SIB = run_sibyl(NV, T, λ, γ; dt=1., maxit = 100, tol = 1e-10)

        dict_res[λ0]["MF"] = n_inf(model, out_MF)
        dict_res[λ0]["EPI"] = n_inf(model, out_EPI)
        dict_res[λ0]["EPIapprox"] = n_inf(model, out_EPIapprox)
        dict_res[λ0]["SIB"] = n_inf(out_SIB)
        dict_res[λ0]["sampling"] = out_sampling
    end

    isdir("res_lambda") || mkdir("res_lambda")
    JLD2.save("res_lambda/RRG_N-$(NV)_K-$(K)_nsrc-$(nsrc).jld2", "res", dict_res, "K", K, "gamma", γ, "G", G, "lambdavec", λ0vec)
end


############################################################################################################
##############################################  Proximity  #################################################
############################################################################################################

####### Proximity generator ########
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

function run_Prox(λ0vec, nsrc, T)
    NV = 500
    lmax = round(sqrt(2.8/NV),digits=3)

    γ = nsrc/NV
    prior = ones(NV) * γ

    # Generate Bethe lattice
    rng = Xoshiro(1234)
    G = proximity(NV, lmax; rng=rng)

    dict_res = Dict(λ0 => Dict(method=>zeros(T+1)  for method in ["MF","EPI","EPIapprox","SIB","sampling"]) for λ0 in λ0vec)

    for λ0 in λ0vec
        # constant infection probability
        λ = zeros(NV, NV, T+1)
        for e in edges(G)
            λ[src(e), dst(e), :] .+= λ0 
            λ[dst(e), src(e), :] .+= λ0
        end

        # define de epidemic model
        infectionmodel = SCDC.SI(0.0, NV, T)
        model = SCDC.EpidemicModel(infectionmodel, G, T, log.(1 .- λ[:,:,1:T]))

        println("running lambda = $λ0")
        out_sampling = run_samples_parallel(model, γ, 10000; reject=false)
        out_MF = run_fwd_IBMF(model, λ, prior)
        out_EPI = SCDC.run_fwd_dynamics(model, prior)
        out_EPIapprox = run_fwd_dynamicsSCDCa(model, prior)
        out_SIB = run_sibyl(NV, T, λ, γ; dt=1., maxit = 100, tol = 1e-10)

        dict_res[λ0]["MF"] = n_inf(model, out_MF)
        dict_res[λ0]["EPI"] = n_inf(model, out_EPI)
        dict_res[λ0]["EPIapprox"] = n_inf(model, out_EPIapprox)
        dict_res[λ0]["SIB"] = n_inf(out_SIB)
        dict_res[λ0]["sampling"] = out_sampling
    end

    isdir("res_lambda") || mkdir("res_lambda")
    JLD2.save("res_lambda/Prox_N-$(NV)_lmax-$(lmax)_nsrc-$(nsrc).jld2", "res", dict_res, "lmax", lmax, "gamma", γ, "G", G, "lambdavec", λ0vec)
end

λ0vec = [0.05, 0.07, 0.1, 0.2, 0.3, 0.5]
nsrc = 5
println("Bethe")
run_bethe(λ0vec, nsrc, 25)
println("RRG, K=4")
run_RRG(4, λ0vec, nsrc, 25)
println("RRG, K=15")
run_RRG(15, λ0vec, nsrc, 15)
println("Proximity")
run_Prox(λ0vec, nsrc, 15)