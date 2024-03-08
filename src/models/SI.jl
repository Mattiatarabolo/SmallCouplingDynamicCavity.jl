"""
    struct SI <: InfectionModel
        εᵢᵗ::Array{Float64, 2} # Autoinfection probabilities
    end

The `SI` struct represents the SI (Susceptible-Infected) infection model.

# Fields
- `εᵢᵗ`: An NVxT array representing the self-infection probabilities over time, where NV is the number of nodes and T is the number of time-steps. Each element εᵢᵗ[i, t] denotes the probability of node i infecting itself at time t.

"""
struct SI <: InfectionModel
    εᵢᵗ::Array{Float64, 2} # Autoinfection probabilities

    """
        SI(
            εᵢᵗ::Union{Float64,Array{Float64,2}},
            NV::Int,
            T::Int)

    Defines the SI infection model.

    # Arguments
    - `εᵢᵗ`: Self-infection probability. Can be either a Float64 (constant over all nodes and times) or a NVxT matrix.
    - `NV`: Number of nodes of the contact graph.
    - `T`: Number of time-steps.

    # Returns
    - An instance of the SI struct representing the SI infection model.
    """
    function SI(
        εᵢᵗ::Union{Float64, Array{Float64, 2}},
        NV::Int,
        T::Int)
        if typeof(εᵢᵗ) == Float64
            εᵢᵗ = ones(NV, T) .* εᵢᵗ
        end

        new(εᵢᵗ)
    end
end

n_states(X::SI) = 2


function nodes_formatting(
    model::EpidemicModel{SI,TG}, 
    obsprob::Function) where {TG<:AbstractGraph}

    nodes = Vector{Node{SI,TG}}()

    for i in 1:model.N
        obs = ones(2, model.T + 1)
        obs[1, :] = [obsprob(Ob, 0.0) for Ob in model.obsmat[i, :]]
        obs[2, :] = [obsprob(Ob, 1.0) for Ob in model.obsmat[i, :]]

        ∂ = neighbors(model.G, i)

        ν∂ = [model.ν[k, i, :] for k in ∂]

        push!(nodes, Node(i, ∂, model.T, ν∂, obs, model))
        
    end
    return collect(nodes)
end

function nodes_formatting(
    model::EpidemicModel{SI,TG}, 
    obsprob::Function) where {TG<:Vector{<:AbstractGraph}}

    nodes = Vector{Node{SI,TG}}()

    for i in 1:model.N
        obs = ones(2, model.T + 1)
        obs[1, :] = [obsprob(Ob, 0.0) for Ob in model.obsmat[i, :]]
        obs[2, :] = [obsprob(Ob, 1.0) for Ob in model.obsmat[i, :]]

        ∂ = Vector{Int}()

        for t in 1:model.T+1
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
    inode::Node{SI,TG},
    iindex::Int,
    jnode::Node{SI,TG},
    jindex::Int,
    sumargexp::SumM,
    infectionmodel::SI) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    M[1, 1, :] .= (exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1])).*(1 .- infectionmodel.εᵢᵗ[inode.i, :]) .* inode.obs[1, 1:end-1]
    M[1, 2, :] .= (1 .- exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1]).*(1 .- infectionmodel.εᵢᵗ[inode.i, :])) .* inode.obs[1, 1:end-1]
    M[2, 2, :] .= exp.(sumargexp.sumμ .- inode.cavities[jindex].μ .* jnode.νs[iindex][1:end-1]) .* inode.obs[2, 1:end-1]
end

function fill_transmat_marg!(
    M::Array{Float64,3},
    inode::Node{SI,TG},
    sumargexp::SumM,
    infectionmodel::SI) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    M[1, 1, :] .= (exp.(sumargexp.summ[1:end-1])).*(1 .- infectionmodel.εᵢᵗ[inode.i, :]) .* inode.obs[1, 1:end-1]
    M[1, 2, :] .= (1 .- exp.(sumargexp.summ[1:end-1]).*(1 .- infectionmodel.εᵢᵗ[inode.i, :])) .* inode.obs[1, 1:end-1]
    M[2, 2, :] .= exp.(sumargexp.sumμ) .* inode.obs[2, 1:end-1]
end

"""
    sim_epidemics(
        model::EpidemicModel{SI,TG};
        patient_zero::Union{Vector{Int},Nothing}=nothing,
        γ::Union{Float64,Nothing}=nothing) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

Simulates an epidemic outbreak using the SI (Susceptible-Infectious) model.

# Arguments
- `model`: The SI epidemic model, encapsulating information about the infection dynamics, contact graph, and other parameters.
- `patient_zero`: (Optional) A vector specifying the indices of initial infected individuals. If not provided (default `nothing`), patient zero is selected randomly based on the probability `γ`.
- `γ`: (Optional) The probability of being a patient zero. If `patient_zero` is not specified and `γ` is provided, patient zero is chosen randomly with probability `γ`. If both `patient_zero` and `γ` are not provided (default `nothing`), patient zero is selected randomly with equal probability for each individual.

# Returns
- A matrix representing the epidemic outbreak configuration over time. Each row corresponds to a node, and each column represents a time step. The values in the matrix indicate the state of each node at each time step: 0.0 for Susceptible (S) and 1.0 for Infected (I).
"""
function sim_epidemics(
    model::EpidemicModel{SI,TG};
    patient_zero::Union{Vector{Int},Nothing}=nothing,
    γ::Union{Float64,Nothing}=nothing) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    inf₀ = false
    if patient_zero === nothing && γ !== nothing
        while !inf₀
            patient_zero = rand(Binomial(1,γ), model.N)
            patient_zero = findall(x->x==1, patient_zero)
            inf₀ = !isempty(patient_zero)
        end
    elseif patient_zero === nothing && γ === nothing
        while !inf₀
            patient_zero = rand(Binomial(1,1/model.N), model.N)
            patient_zero = findall(x->x==1, patient_zero)
            inf₀ = !isempty(patient_zero)
        end
    end

    config = zeros(model.N, model.T + 1)

    config[patient_zero, 1] .+= 1.0

    hs = zeros(model.N)
    for t in 1:model.T
        hs = config[:, t]' * model.ν[:, :, t]
        config[:, t+1] = [x + (1 - x) * rand(Bernoulli(1 - exp(h))) for (x, h) in zip(config[:, t], hs)]
    end
    return config
end
