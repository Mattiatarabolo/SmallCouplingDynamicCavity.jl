"""
    struct SIRS <: InfectionModel
        εᵢᵗ::Array{Float64, 2} # Autoinfection probabilities
        rᵢᵗ::Array{Float64, 2} # Recovery probabilities
        σᵢᵗ::Array{Float64, 2} # Loss of immunity probabilities
    end

The `SIRS` struct represents the SIRS (Susceptible-Infected-Recovered-Susceptible) infection model.

# Fields
- `εᵢᵗ`: An NVxT array representing the self-infection probabilities over time, where NV is the number of nodes and T is the number of time-steps. Each element εᵢᵗ[i, t] denotes the probability of node i infecting itself at time t.
- `rᵢᵗ`: An NVxT array representing the recovery probabilities over time, where NV is the number of nodes and T is the number of time-steps. Each element rᵢᵗ[i, t] denotes the probability of node i recovering from infection at time t.
- `σᵢᵗ`: An NVxT array representing the loss of immunity probabilities over time, where NV is the number of nodes and T is the number of time-steps. Each element σᵢᵗ[i, t] denotes the probability of node i losing immunity and becoming susceptible again at time t.

"""
struct SIRS <: InfectionModel
    εᵢᵗ::Array{Float64, 2} # Autoinfection probabilities
    rᵢᵗ::Array{Float64, 2} # Recovery probabilities
    σᵢᵗ::Array{Float64, 2} # Loss of immunity probabilities

    """
        SIRS(
            εᵢᵗ::Union{Float64,Array{Float64,2}},
            rᵢᵗ::Union{Float64,Array{Float64,2}},
            σᵢᵗ::Union{Float64,Array{Float64,2}},
            NV::Int,
            T::Int)

    Defines the SIRS infection model.

    # Arguments
    - `εᵢᵗ`: Self-infection probability. Can be either a Float64 (constant over all nodes and times) or a NVxT matrix.
    - `rᵢᵗ`: Recovery probability. Can be either a Float64 (constant over all nodes and times) or a NVxT matrix.
    - `σᵢᵗ`: Loss of immunity probability. Can be either a Float64 (constant over all nodes and times) or a NVxT matrix.
    - `NV`: Number of nodes of the contact graph.
    - `T`: Number of time-steps.

    # Returns
    - An instance of the SIRS struct representing the SIRS infection model.
    """
    function SIRS(
        εᵢᵗ::Union{Float64,Array{Float64,2}},
        rᵢᵗ::Union{Float64,Array{Float64,2}},
        σᵢᵗ::Union{Float64,Array{Float64,2}},
        NV::Int,
        T::Int)
        if typeof(εᵢᵗ) == Float64
            εᵢᵗ = ones(NV, T) .* εᵢᵗ
        end
        if typeof(rᵢᵗ) == Float64
            rᵢᵗ = ones(NV, T) .* rᵢᵗ
        end
        if typeof(σᵢᵗ) == Float64
            σᵢᵗ = ones(NV, T) .* σᵢᵗ
        end

        new(εᵢᵗ, rᵢᵗ, σᵢᵗ)
    end
end

n_states(X::SIRS) = 3


function nodes_formatting(
    model::EpidemicModel{SIRS,TG}, 
    obsprob::Function) where {TG<:AbstractGraph}

    nodes = Vector{Node{SIRS,TG}}()

    for i in 1:model.N
        obs = ones(3, model.T + 1)
        @inbounds @fastmath @simd for t in 1:model.T+1
            obs[1, t] = obsprob(model.obsmat[i, t], 0)
            obs[2, t] = obsprob(model.obsmat[i, t], 1)
            obs[3, t] = obsprob(model.obsmat[i, t], 2)
        end

        ∂ = neighbors(model.G, i)

        ν∂ = [model.ν[k, i, :] for k in ∂]

        push!(nodes, Node(i, ∂, model.T, ν∂, obs, model))
    end
    return collect(nodes)
end


function nodes_formatting(
    model::EpidemicModel{SIRS,TG}, 
    obsprob::Function) where {TG<:Vector{<:AbstractGraph}}

    nodes = Vector{Node{SIRS,TG}}()

    for i in 1:model.N
        obs = ones(3, model.T + 1)
        @inbounds @fastmath @simd for t in 1:model.T+1
            obs[1, t] = obsprob(model.obsmat[i, t], 0)
            obs[2, t] = obsprob(model.obsmat[i, t], 1)
            obs[3, t] = obsprob(model.obsmat[i, t], 2)
        end

        ∂ = Vector{Int}()

        for t in 1:model.T
            ∂ = union(∂, neighbors(model.G[t], i))
        end

        ν∂ = [model.ν[k, i, :] for k in ∂]

        push!(nodes, Node(i, ∂, model.T, ν∂, obs, model))
    end
    return collect(nodes)
end

# function to fill the transfer matrix
function fill_transmat_cav!(
    M::Array{Float64,3},
    inode::Node{SIRS,TG},
    iindex::Int,
    jnode::Node{SIRS,TG},
    jindex::Int,
    sumargexp::SumM,
    model::EpidemicModel{SIRS,TG}) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    @inbounds @fastmath @simd for t in 1:model.T
        M[1, 1, t] = exp(sumargexp.summ[t] - inode.cavities[jindex].m[t] * inode.νs[jindex][t])*(1 - model.Disease.εᵢᵗ[inode.i, t]) * inode.obs[1, t]
        M[1, 2, t] = (1 - exp(sumargexp.summ[t] - inode.cavities[jindex].m[t] * inode.νs[jindex][t])*(1 - model.Disease.εᵢᵗ[inode.i, t])) * inode.obs[1, t]
        M[2, 2, t] = (1 - model.Disease.rᵢᵗ[inode.i, t]) * exp(sumargexp.sumμ[t] - inode.cavities[jindex].μ[t] * jnode.νs[iindex][t]) * inode.obs[2, t]
        M[2, 3, t] = model.Disease.rᵢᵗ[inode.i, t] * exp(sumargexp.sumμ[t] - inode.cavities[jindex].μ[t] * jnode.νs[iindex][t]) * inode.obs[2, t]
        M[3, 1, t] = model.Disease.σᵢᵗ[inode.i, t] * inode.obs[3, t]
        M[3, 3, t] = (1 - model.Disease.σᵢᵗ[inode.i, t]) * inode.obs[3, t]
    end
end

function fill_transmat_marg!(
    M::Array{Float64,3},
    inode::Node{SIRS,TG},
    sumargexp::SumM,
    model::EpidemicModel{SIRS,TG}) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    @inbounds @fastmath @simd for t in 1:model.T
        M[1, 1, t] = exp(sumargexp.summ[t])*(1 - model.Disease.εᵢᵗ[inode.i, t]) * inode.obs[1, t]
        M[1, 2, t] = (1 - exp(sumargexp.summ[t])*(1 - model.Disease.εᵢᵗ[inode.i, t])) * inode.obs[1, t]
        M[2, 2, t] = (1 - model.Disease.rᵢᵗ[inode.i, t]) * exp(sumargexp.sumμ[t]) * inode.obs[2, t]
        M[2, 3, t] = model.Disease.rᵢᵗ[inode.i, t] * exp(sumargexp.sumμ[t]) * inode.obs[2, t]
        M[3, 1, t] = model.Disease.σᵢᵗ[inode.i, t] * inode.obs[3, t]
        M[3, 3, t] = (1 - model.Disease.σᵢᵗ[inode.i, t]) * inode.obs[3, t]
    end
end

"""
    sim_epidemics(
        model::EpidemicModel{SIRS,TG};
        patient_zero::Union{Vector{Int},Nothing}=nothing,
        γ::Union{Float64,Nothing}=nothing) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

Simulates an epidemic outbreak using the SIRS (Susceptible-Infectious-Recovered-Susceptible) model.

# Arguments
- `model`: The SIRS epidemic model, encapsulating information about the infection dynamics, contact graph, and other parameters.
- `patient_zero`: (Optional) A vector specifying the indices of initial infected individuals. If not provided (default `nothing`), patient zero is selected randomly based on the probability `γ`.
- `γ`: (Optional) The probability of being a patient zero. If `patient_zero` is not specified and `γ` is provided, patient zero is chosen randomly with probability `γ`. If both `patient_zero` and `γ` are not provided (default `nothing`), patient zero is selected randomly with equal probability for each individual.
- `rng`: (Optional) A random number generator. Default is `Xoshiro(1234)`.

# Returns
- A matrix representing the epidemic outbreak configuration over time. Each row corresponds to a node, and each column represents a time step. The values in the matrix indicate the state of each node at each time step: 0.0 for Susceptible (S), 1.0 for Infected (I), and 2.0 for Recovered (R).
"""
function sim_epidemics(
    model::EpidemicModel{SIRS,TG};
    patient_zero::Union{Vector{Int},Nothing}=nothing,
    γ::Union{Float64,Nothing}=nothing,
    rng::AbstractRNG=Xoshiro(1234)) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    inf₀ = false
    if patient_zero === nothing && γ !== nothing
        while !inf₀
            patient_zero = rand(rng, Binomial(1,γ), model.N)
            patient_zero = findall(x->x==1, patient_zero)
            inf₀ = !isempty(patient_zero)
        end
    elseif patient_zero === nothing && γ === nothing
        while !inf₀
            patient_zero = rand(rng, Binomial(1,1/model.N), model.N)
            patient_zero = findall(x->x==1, patient_zero)
            inf₀ = !isempty(patient_zero)
        end
    end

    config = zeros(Int8,model.N, model.T + 1)

    config[patient_zero,1] .+= 1

    function W_SIRS(x, y, h::Float64, r::Float64, σ::Float64)
        if x == 0
            return (y == 0) * exp(h) + (y == 2) * σ
        elseif x == 1
            return (y == 0) * (1 - exp(h)) + (y == 1) * (1 - r)
        elseif x == 2
            return (y == 1) * r + (y == 2) * (1 - σ)
        else
            throw(ArgumentError("Invalid value for y"))
        end
    end
    hs = zeros(model.N)
    for t in 1:model.T
        hs = [Float64(x == 1) for x in config[:, t]]' * model.ν[:, :, t]
        config[:, t+1] = [
            if (u <= W_SIRS(0, x, h, r, σ))
                0
            elseif (W_SIRS(0, x, h, r, σ) < u <= W_SIRS(0, x, h, r, σ) + W_SIRS(1, x, h, r, σ))
                1
            else
                2
            end for (x, h, r, σ, u) in zip(config[:, t], hs, model.Disease.rᵢᵗ[:, t], model.Disease.σᵢᵗ[:, t], rand(rng, Float64, model.N))
        ]
    end
    return config
end
  