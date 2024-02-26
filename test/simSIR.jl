@testset "SimSI" begin
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
    λ = zeros(NV, NV, T+1)
    for e in edges(G)
        λ[src(e), dst(e), :] = ones(T+1) * λ₀
        λ[dst(e), src(e), :] = ones(T+1) * λ₀
    end

    # define de epidemic model
    infectionmodel = SI(ε_autoinf, NV, T)
    model = EpidemicModel(infectionmodel, G, T, log.(1 .- λ))

    # epidemic simulation
    Random.seed!(3)
    config = sim_epidemics(model, patient_zero=[1])

    configtest=[1.0  1.0  1.0  1.0  1.0  1.0;
    0.0  0.0  0.0  1.0  1.0  1.0;
    0.0  0.0  0.0  0.0  0.0  0.0;
    0.0  0.0  0.0  0.0  1.0  1.0;
    0.0  0.0  0.0  0.0  1.0  1.0;
    0.0  0.0  0.0  0.0  0.0  0.0;
    0.0  0.0  0.0  0.0  0.0  0.0;
    0.0  0.0  0.0  0.0  0.0  1.0;
    0.0  0.0  0.0  0.0  0.0  0.0;
    0.0  0.0  0.0  0.0  0.0  0.0]

    @test config ≈ configtest
end