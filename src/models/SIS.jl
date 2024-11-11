"""
    struct SIS <: InfectionModel
        εᵢᵗ::Array{Float64, 2} # Autoinfection probabilities
        rᵢᵗ::Array{Float64, 2} # Recovery probabilities
    end

The `SIS` struct represents the SIS (Susceptible-Infected-Susceptible) infection model.

# Fields
- `εᵢᵗ`: An NVxT array representing the self-infection probabilities over time, where NV is the number of nodes and T is the number of time-steps. Each element εᵢᵗ[i, t] denotes the probability of node i infecting itself at time t.
- `rᵢᵗ`: An NVxT array representing the recovery probabilities over time, where NV is the number of nodes and T is the number of time-steps. Each element rᵢᵗ[i, t] denotes the probability of node i recovering from infection at time t.

"""
struct SIS <: InfectionModel
    εᵢᵗ::Array{Float64, 2} # Autoinfection probabilities
    rᵢᵗ::Array{Float64, 2} # Recovery probabilities

    """
        SIS(
            εᵢᵗ::Union{Float64,Array{Float64,2}},
            rᵢᵗ::Union{Float64,Array{Float64,2}},
            NV::Int,
            T::Int)

    Defines the SIS infection model.

    # Arguments
    - `εᵢᵗ`: Self-infection probability. Can be either a Float64 (constant over all nodes and times) or a NVxT matrix.
    - `rᵢᵗ`: Recovery probability. Can be either a Float64 (constant over all nodes and times) or a NVxT matrix.
    - `NV`: Number of nodes of the contact graph.
    - `T`: Number of time-steps.

    # Returns
    - An instance of the SIS struct representing the SIS infection model.
    """
    function SIS(
        εᵢᵗ::Union{Float64,Array{Float64,2}},
        rᵢᵗ::Union{Float64,Array{Float64,2}},
        NV::Int,
        T::Int)
        if typeof(εᵢᵗ) == Float64
            εᵢᵗ = ones(NV, T) .* εᵢᵗ
        end
        if typeof(rᵢᵗ) == Float64
            rᵢᵗ = ones(NV, T) .* rᵢᵗ
        end

        new(εᵢᵗ, rᵢᵗ)
    end
end

n_states(X::SIS) = 2


function nodes_formatting(
    model::EpidemicModel{SIS,TG}, 
    obsprob::Function) where {TG<:AbstractGraph}

    nodes = Vector{Node{SIS,TG}}()

    for i in 1:model.N
        obs = ones(2, model.T + 1)
        @inbounds @fastmath @simd for t in 1:model.T+1
            obs[1, t] = obsprob(model.obsmat[i, t], 0)
            obs[2, t] = obsprob(model.obsmat[i, t], 1)
        end

        ∂ = neighbors(model.G, i)

        ν∂ = [model.ν[k, i, :] for k in ∂]

        push!(nodes, Node(i, ∂, model.T, ν∂, obs, model))
    end
    model.obsmat = nothing
    return collect(nodes)
end

function nodes_formatting(
    model::EpidemicModel{SIS,TG}, 
    obsprob::Function) where {TG<:Vector{<:AbstractGraph}}

    nodes = Vector{Node{SIS,TG}}()

    for i in 1:model.N
        obs = ones(2, model.T + 1)
        @inbounds @fastmath @simd for t in 1:model.T+1
            obs[1, t] = obsprob(model.obsmat[i, t], 0)
            obs[2, t] = obsprob(model.obsmat[i, t], 1)
        end

        ∂ = Vector{Int}()

        for t in 1:model.T
            ∂ = union(∂, neighbors(model.G[t], i))
        end

        ν∂ = [model.ν[k, i, :] for k in ∂]

        push!(nodes, Node(i, ∂, model.T, ν∂, obs, model))
    end
    model.obsmat = nothing
    return collect(nodes)
end

# function to fill the transfer matrix
function fill_transmat_cav!(
    M::Array{Float64,3},
    inode::Node{SIS,TG},
    iindex::Int,
    jnode::Node{SIS,TG},
    jindex::Int,
    sumargexp::SumM,
    model::EpidemicModel{SIS,TG}) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    @inbounds @fastmath @simd for t in 1:model.T
        M[1, 1, t] = exp(sumargexp.summ[t] - inode.cavities[jindex].m[t] * inode.νs[jindex][t]) * (1 - infectionmodel.εᵢᵗ[inode.i, t]) * inode.obs[1, t]
        M[1, 2, t] = (1 - exp(sumargexp.summ[t] - inode.cavities[jindex].m[t] * inode.νs[jindex][t]) * (1 - infectionmodel.εᵢᵗ[inode.i, t])) * inode.obs[1, t]
        M[2, 1, t] = infectionmodel.rᵢᵗ[inode.i, t] * exp(sumargexp.sumμ[t] - inode.cavities[jindex].μ[t] * jnode.νs[iindex][t]) * inode.obs[2, t]
        M[2, 2, t] = (1 - infectionmodel.rᵢᵗ[inode.i, t]) * exp(sumargexp.sumμ[t] - inode.cavities[jindex].μ[t] * jnode.νs[iindex][t]) * inode.obs[2, t]
    end
end

function fill_transmat_marg!(
    M::Array{Float64,3},
    inode::Node{SIS,TG},
    sumargexp::SumM,
    model::EpidemicModel{SIS,TG}) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    @inbounds @fastmath @simd for t in 1:model.T
        M[1, 1, t] = exp(sumargexp.summ[t]) * (1 - infectionmodel.εᵢᵗ[inode.i, t]) * inode.obs[1, t]
        M[1, 2, t] = (1 - exp(sumargexp.summ[t]) * (1 - infectionmodel.εᵢᵗ[inode.i, t])) * inode.obs[1, t]
        M[2, 1, t] = infectionmodel.rᵢᵗ[inode.i, t] * exp(sumargexp.sumμ[t]) * inode.obs[2, t]
        M[2, 2, t] = (1 - infectionmodel.rᵢᵗ[inode.i, t]) * exp(sumargexp.sumμ[t]) * inode.obs[2, t]
    end
end

"""
    sim_epidemics(
        model::EpidemicModel{SIS,TG};
        patient_zero::Union{Vector{Int},Nothing}=nothing,
        γ::Union{Float64,Nothing}=nothing) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

Simulates an epidemic outbreak using the SIS (Susceptible-Infectious-Susceptible) model.

# Arguments
- `model`: The SIS epidemic model, encapsulating information about the infection dynamics, contact graph, and other parameters.
- `patient_zero`: (Optional) A vector specifying the indices of initial infected individuals. If not provided (default `nothing`), patient zero is selected randomly based on the probability `γ`.
- `γ`: (Optional) The probability of being a patient zero. If `patient_zero` is not specified and `γ` is provided, patient zero is chosen randomly with probability `γ`. If both `patient_zero` and `γ` are not provided (default `nothing`), patient zero is selected randomly with equal probability for each individual.

# Returns
- A matrix representing the epidemic outbreak configuration over time. Each row corresponds to a node, and each column represents a time step. The values in the matrix indicate the state of each node at each time step: 0.0 for Susceptible (S) and 1.0 for Infected (I).
"""
function sim_epidemics(
    model::EpidemicModel{SIS,TG};
    patient_zero::Union{Vector{Int},Nothing}=nothing,
    γ::Union{Float64,Nothing}=nothing) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    inf₀ = false
    if patient_zero === nothing && γ !== nothing
        while !inf₀
            patient_zero = rand(Binomial(1, γ), model.N)
            patient_zero = findall(x -> x == 1, patient_zero)
            inf₀ = !isempty(patient_zero)
        end
    elseif patient_zero === nothing && γ === nothing
        while !inf₀
            patient_zero = rand(Binomial(1, 1 / model.N), model.N)
            patient_zero = findall(x -> x == 1, patient_zero)
            inf₀ = !isempty(patient_zero)
        end
    end

    config = zeros(Int8, model.N, model.T + 1)
    config[patient_zero, 1] .+= 1

    hs = zeros(model.N)
    for t in 1:model.T
        hs .= (config[:, t]' * model.ν[:, :, t])'
        config[:, t+1] .= [x * rand(Bernoulli(1 - r)) + (1 - x) * rand(Bernoulli(1 - exp(h))) for (x, h, r) in zip(config[:, t], hs, model.Disease.rᵢᵗ[:, t])]
    end
    return config
end
