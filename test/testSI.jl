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
    SmallCouplingDynamicCavity.compute_sumargexp!(inode, nodes, sumargexp)
        # j = 2
        jindex = 1
        jnode = nodes[2]
        iindex = jnode.∂_idx[inode.i]
        SmallCouplingDynamicCavity.compute_ρ!(inode, iindex, jnode, jindex, sumargexp, M, ρ, prior, T, infectionmodel)
        @test M ≈ Mc1
        @test ρ.fwm ≈ ρc1fwm
        @test ρ.bwm ≈ ρcbwm
        SmallCouplingDynamicCavity.update_single_message!(0.0, jnode, iindex, ρ, M, 0.0, inode, -Inf)
        @test jnode.cavities[iindex].m ≈ mc1
        @test jnode.cavities[iindex].μ ≈ μc
        # j = 3
        jindex = 2
        jnode = nodes[3]
        iindex = jnode.∂_idx[inode.i]
        SmallCouplingDynamicCavity.compute_ρ!(inode, iindex, jnode, jindex, sumargexp, M, ρ, prior, T, infectionmodel)
        @test M ≈ Mc1
        @test ρ.fwm ≈ ρc1fwm
        @test ρ.bwm ≈ ρcbwm
        SmallCouplingDynamicCavity.update_single_message!(0.0, jnode, iindex, ρ, M, 0.0, inode, -Inf)
        @test jnode.cavities[iindex].m ≈ mc1
        @test jnode.cavities[iindex].μ ≈ μc
    # i=2
    inode = nodes[2]
    SmallCouplingDynamicCavity.compute_sumargexp!(inode, nodes, sumargexp)
        # j = 1
        jindex = 1
        jnode = nodes[1]
        iindex = jnode.∂_idx[inode.i]
        SmallCouplingDynamicCavity.compute_ρ!(inode, iindex, jnode, jindex, sumargexp, M, ρ, prior, T, infectionmodel)
        @test M ≈ Mc23
        @test ρ.fwm ≈ ρc23fwm
        @test ρ.bwm ≈ ρcbwm
        SmallCouplingDynamicCavity.update_single_message!(0.0, jnode, iindex, ρ, M, 0.0, inode, -Inf)
        @test jnode.cavities[iindex].m ≈ mc23
        @test jnode.cavities[iindex].μ ≈ μc
    # i=3
    inode = nodes[3]
    SmallCouplingDynamicCavity.compute_sumargexp!(inode, nodes, sumargexp)
        # j = 1
        jindex = 1
        jnode = nodes[1]
        iindex = jnode.∂_idx[inode.i]
        SmallCouplingDynamicCavity.compute_ρ!(inode, iindex, jnode, jindex, sumargexp, M, ρ, prior, T, infectionmodel)
        @test M ≈ Mc23
        @test ρ.fwm ≈ ρc23fwm
        @test ρ.bwm ≈ ρcbwm
        SmallCouplingDynamicCavity.update_single_message!(0.0, jnode, iindex, ρ, M, 0.0, inode, -Inf)
        @test jnode.cavities[iindex].m ≈ mc23
        @test jnode.cavities[iindex].μ ≈ μc
end





##### Test SI model #####
NV = 10 # number of graph vertices
k = 3 # average degree

#genrate an Erdos-Renyi random graph with average connectivity k
Random.seed!(1)
G = erdos_renyi(NV, k/NV)

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
    Random.seed!(3)
    config = sim_epidemics(model, patient_zero=[1])

    @test config ≈ configtest
end

@testset "inferenceSI" begin
    Random.seed!(1)
    nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp, μ_cutoff = μ_cutoff)

    marg = zeros(NV,2,T+1)
    for (i,node) in enumerate(nodes)
        marg[i,:,:] = node.marg.m
    end

    margtest = load("data/margSI.jld2", "marg")
    @test marg ≈ margtest
end

@testset "inferenceSIscheme" begin
    maxiter = [90, 50]  # max number of iterations scheme
    damp = [0.9, 0.5]  # damping factor scheme

    Random.seed!(1)
    nodes = run_SCDC(model, obsprob, γ, maxiter, epsconv, damp, μ_cutoff = μ_cutoff)

    marg = zeros(NV,2,T+1)
    for (i,node) in enumerate(nodes)
        marg[i,:,:] = node.marg.m
    end

    margtestscheme = load("data/margSIscheme.jld2", "marg")
    @test marg ≈ margtestscheme
end
