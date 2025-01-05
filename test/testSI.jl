##### Test internal functions #####
NV = 3
G = SimpleGraph(NV)
add_edge!(G, 1, 2)
add_edge!(G, 1, 3)

T = 3
ν₀ = 1.0
γ = 0.5

ν = zeros(NV, NV, T)
for e in edges(G)
    ν[src(e), dst(e), :] = ones(T) * ν₀
    ν[dst(e), src(e), :] = ones(T) * ν₀
end

infectionmodel = SI(0.0, NV, T)
model = EpidemicModel(infectionmodel, G, T, ν)

prior = zeros(SmallCouplingDynamicCavity.n_states(model.Disease), model.N)
@inbounds @fastmath @simd for i in 1:model.N
    prior[1, i] = 1 - γ # x_i = S
    prior[2, i] = γ # x_i = I
end

# Format nodes for inference
nodes = SmallCouplingDynamicCavity.nodes_formatting(model, obsprob)

# Initialize message objects
M = SmallCouplingDynamicCavity.TransMat(model.T, model.Disease)
ρ = SmallCouplingDynamicCavity.FBm(model.T, model.Disease)
sumargexp = SmallCouplingDynamicCavity.SumM(model.T)

Mc1 = [exp(1) 1-exp(1);0 1;;;exp(1) 1-exp(1);0 1;;;exp(1) 1-exp(1);0 1]
Mc23 = [1 0;0 1;;;1 0;0 1;;;1 0;0 1]

ρc1fwm = [1/2 exp(1)/2 exp(2)/2 exp(3)/2;1/2 (2-exp(1))/2 (2-exp(2))/2 (2-exp(3))/2]
ρc23fwm = ones(2,T+1)/2
ρcbwm = ones(2,T+1)


mc1 = [1/2,(2-exp(1))/2,(2-exp(2))/2,(2-exp(3))/2]
mc23 = ones(T+1)/2
μc = zeros(T)

@testset "internalSI" begin
    # i=1
    inode = nodes[1]
    SmallCouplingDynamicCavity.compute_sumargexp!(inode, nodes, sumargexp, model)
        # j = 2
        jindex = 1
        jnode = nodes[2]
        iindex = jnode.∂_idx[inode.i]
        SmallCouplingDynamicCavity.compute_ρ!(inode, iindex, jnode, jindex, sumargexp, M, ρ, prior, model)
        @test M ≈ Mc1
        @test ρ.fwm ≈ ρc1fwm
        @test ρ.bwm ≈ ρcbwm
        SmallCouplingDynamicCavity.update_single_message!(0.0, jnode, iindex, ρ, M, 0.0, -Inf, model)
        @test jnode.cavities[iindex].m ≈ mc1
        @test jnode.cavities[iindex].μ ≈ μc
        # j = 3
        jindex = 2
        jnode = nodes[3]
        iindex = jnode.∂_idx[inode.i]
        SmallCouplingDynamicCavity.compute_ρ!(inode, iindex, jnode, jindex, sumargexp, M, ρ, prior, model)
        @test M ≈ Mc1
        @test ρ.fwm ≈ ρc1fwm
        @test ρ.bwm ≈ ρcbwm
        SmallCouplingDynamicCavity.update_single_message!(0.0, jnode, iindex, ρ, M, 0.0, -Inf, model)
        @test jnode.cavities[iindex].m ≈ mc1
        @test jnode.cavities[iindex].μ ≈ μc
    # i=2
    inode = nodes[2]
    SmallCouplingDynamicCavity.compute_sumargexp!(inode, nodes, sumargexp, model)
        # j = 1
        jindex = 1
        jnode = nodes[1]
        iindex = jnode.∂_idx[inode.i]
        SmallCouplingDynamicCavity.compute_ρ!(inode, iindex, jnode, jindex, sumargexp, M, ρ, prior, model)
        @test M ≈ Mc23
        @test ρ.fwm ≈ ρc23fwm
        @test ρ.bwm ≈ ρcbwm
        SmallCouplingDynamicCavity.update_single_message!(0.0, jnode, iindex, ρ, M, 0.0, -Inf, model)
        @test jnode.cavities[iindex].m ≈ mc23
        @test jnode.cavities[iindex].μ ≈ μc
    # i=3
    inode = nodes[3]
    SmallCouplingDynamicCavity.compute_sumargexp!(inode, nodes, sumargexp, model)
        # j = 1
        jindex = 1
        jnode = nodes[1]
        iindex = jnode.∂_idx[inode.i]
        SmallCouplingDynamicCavity.compute_ρ!(inode, iindex, jnode, jindex, sumargexp, M, ρ, prior, model)
        @test M ≈ Mc23
        @test ρ.fwm ≈ ρc23fwm
        @test ρ.bwm ≈ ρcbwm
        SmallCouplingDynamicCavity.update_single_message!(0.0, jnode, iindex, ρ, M, 0.0, -Inf, model)
        @test jnode.cavities[iindex].m ≈ mc23
        @test jnode.cavities[iindex].μ ≈ μc
end





##### Test SI model #####
NV = 10 # number of graph vertices
k = 3 # average degree

#genrate an Erdos-Renyi random graph with average connectivity k
rng = Xoshiro(1)
G = erdos_renyi(NV, k/NV, rng=rng)

# define the constants
T = 5 # total time
γ = 1/NV # Patient zero probability
λ₀ = 0.3 # Infection rate

# constant infection probability
λ = zeros(NV, NV, T)
for e in edges(G)
    λ[src(e), dst(e), :] = ones(T) * λ₀
    λ[dst(e), src(e), :] = ones(T) * λ₀
end

# define de epidemic model
infectionmodel = SI(0.0, NV, T)
model = EpidemicModel(infectionmodel, G, T, log.(1 .- λ))

configtest=[1  1  1  1  1  1;
            0  0  0  1  1  1;
            0  0  0  0  0  0;
            0  0  0  0  1  1;
            0  0  0  0  1  1;
            0  0  0  0  0  0;
            0  0  0  0  0  0;
            0  0  0  0  0  1;
            0  0  0  0  0  0;
            0  0  0  0  0  0]

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

@testset "SimSI" begin
    # epidemic simulation
    Random.seed!(rng, 3)
    config = sim_epidemics(model, patient_zero=[1], rng=rng)

    @test config == configtest
end

@testset "inferenceSI" begin
    Random.seed!(rng, 1)
    nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp, μ_cutoff = μ_cutoff, rng=rng)

    marg = zeros(NV,2,T+1)
    for (i,node) in enumerate(nodes)
        marg[i,:,:] = node.marg.m
    end

    margtest = load("data/margSI.jld2", "marg")
    @test marg ≈ margtest
end

########### checking averaging method when non-converged ##########
n_iter_nc = 10
damp_nc = 0.3
@testset "inferenceSI_nc" begin
    Random.seed!(rng, 1)
    nodes_nc = run_SCDC(model, obsprob, γ, 2, epsconv, damp, μ_cutoff=μ_cutoff, n_iter_nc=n_iter_nc, damp_nc=damp_nc, rng=rng)
    Random.seed!(rng, 1)
    nodes = run_SCDC(model, obsprob, γ, 2, epsconv, damp, μ_cutoff=μ_cutoff, rng=rng)
    run_SCDC!(nodes, model, γ, 0, epsconv, damp, μ_cutoff=μ_cutoff, n_iter_nc=n_iter_nc, damp_nc=damp_nc, rng=rng)

    marg = [node.marg.m[2,T+1] for node in nodes]
    marg_nc = [node.marg.m[2,T+1] for node in nodes_nc]

    @test marg ≈ marg_nc
end


############## Test SI model with damping scheme ##############
maxiter = [90, 50]  # max number of iterations scheme
damp = [0.9, 0.5]  # damping factor scheme
@testset "inferenceSIscheme" begin
    Random.seed!(rng, 1)
    nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp, μ_cutoff = μ_cutoff, rng=rng)

    marg = zeros(NV,2,T+1)
    for (i,node) in enumerate(nodes)
        marg[i,:,:] = node.marg.m
    end

    margtestscheme = load("data/margSIscheme.jld2", "marg")
    @test marg ≈ margtestscheme
end

########### checking averaging method when non-converged with damping scheme ##########
n_iter_nc = 10
damp_nc = 0.3
@testset "inferenceSIscheme_nc" begin
    Random.seed!(rng, 1)
    nodes_nc = run_SCDC(model, obsprob, γ, [2,2], epsconv, damp, μ_cutoff=μ_cutoff, n_iter_nc=n_iter_nc, damp_nc=damp_nc, rng=rng)
    Random.seed!(rng, 1)
    nodes = run_SCDC(model, obsprob, γ, [2,2], epsconv, damp, μ_cutoff=μ_cutoff, rng=rng)
    run_SCDC!(nodes, model, γ, [0,0], epsconv, damp, μ_cutoff=μ_cutoff, n_iter_nc=n_iter_nc, damp_nc=damp_nc, rng=rng)

    marg = [node.marg.m[2,T+1] for node in nodes]
    marg_nc = [node.marg.m[2,T+1] for node in nodes_nc]

    @test marg ≈ marg_nc
end



########### checking forward dynamics ##########
##### Test SI model #####
NV = 50 # number of graph vertices
k = 3 # average degree

#genrate an Erdos-Renyi random graph with average connectivity k
rng = Xoshiro(1)
G = random_regular_graph(NV, k, rng=rng)

# define the constants
T = 10 # total time
γ = 1/NV # Patient zero probability
λ₀ = 0.3 # Infection rate

# constant infection probability
λ = zeros(NV, NV, T)
for e in edges(G)
    λ[src(e), dst(e), :] = ones(T) * λ₀
    λ[dst(e), src(e), :] = ones(T) * λ₀
end

# define de epidemic model
infectionmodel = SI(0.0, NV, T)
model = EpidemicModel(infectionmodel, G, T, log.(1 .- λ))

@testset "fwd_dyn_RRG" begin
    #function to run the forward dynamics on regular graphs
    function fwd_regular(γ, K, ν, T, NV)
        cavs = zeros(T+1)
        margs = zeros(T+1)
        cavs[1] = γ
        margs[1] = γ
        @inbounds @fastmath @simd for t in 1:T
            cavs[t+1] = cavs[t] + (1-cavs[t])*(1-exp((K-1)*cavs[t]*ν))
            margs[t+1] = margs[t] + (1-margs[t])*(1-exp(K*cavs[t]*ν))
        end
        return cavs, margs
    end

    # run SCDC forward dynamics
    nodes = run_fwd_dynamics(model, γ)
    
    # run regular forward dynamics
    cavstest, margtest = fwd_regular(γ, k, log(1-λ₀), T, NV)

    # check the results
    for node in nodes
        @test node.marg.m[2,:] ≈ margtest
        for cav in node.cavities
            @test cav.m ≈ cavstest
        end
    end
end