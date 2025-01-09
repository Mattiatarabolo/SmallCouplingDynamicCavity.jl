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


function nodes_formatting(model::EpidemicModel{SI,TG}) where {TG<:AbstractGraph}
    nodes = Vector{Node{SI,TG}}()

    for i in 1:model.N
        obs = ones(2, model.T + 1)

        ∂ = neighbors(model.G, i)

        ν∂ = [model.ν[k, i, :] for k in ∂]

        push!(nodes, Node(i, ∂, model.T, ν∂, obs, model))
    end
    return collect(nodes)
end


function nodes_formatting(model::EpidemicModel{SI,TG}) where {TG<:Vector{<:AbstractGraph}}
    nodes = Vector{Node{SI,TG}}()

    for i in 1:model.N
        obs = ones(2, model.T + 1)

        ∂ = Vector{Int}()

        for t in 1:model.T
            ∂ = union(∂, neighbors(model.G[t], i))
        end
             
        ν∂ = [model.ν[k, i, :] for k in ∂]

        push!(nodes, Node(i, ∂, model.T, ν∂, obs, model))
    end
    return collect(nodes)
end


function nodes_formatting(
    model::EpidemicModel{SI,TG}, 
    obsprob::Function) where {TG<:AbstractGraph}

    nodes = Vector{Node{SI,TG}}()

    for i in 1:model.N
        obs = ones(2, model.T + 1)
        @inbounds @fastmath @simd for t in 1:model.T+1
            obs[1, t] = obsprob(model.obsmat[i,t], 0)
            obs[2, t] = obsprob(model.obsmat[i,t], 1)
        end

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
        @inbounds @fastmath @simd for t in 1:model.T+1
            obs[1, t] = obsprob(model.obsmat[i,t], 0)
            obs[2, t] = obsprob(model.obsmat[i,t], 1)
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
    inode::Node{SI,TG},
    iindex::Int,
    jnode::Node{SI,TG},
    jindex::Int,
    sumargexp::SumM,
    model::EpidemicModel{SI,TG}) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    @inbounds @fastmath @simd for t in 1:model.T
        M[1, 1, t] = (exp(sumargexp.summ[t] - inode.cavities[jindex].m[t] * inode.νs[jindex][t]) * (1 - model.Disease.εᵢᵗ[inode.i, t])) * inode.obs[1, t]
        M[1, 2, t] = (1 - exp(sumargexp.summ[t] - inode.cavities[jindex].m[t] * inode.νs[jindex][t]) * (1 - model.Disease.εᵢᵗ[inode.i, t])) * inode.obs[1, t]
        M[2, 2, t] = exp(sumargexp.sumμ[t] - inode.cavities[jindex].μ[t] * jnode.νs[iindex][t]) * inode.obs[2, t]
    end
end

function fill_transmat_marg!(
    M::Array{Float64,3},
    inode::Node{SI,TG},
    sumargexp::SumM,
    model::EpidemicModel{SI,TG}) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    @inbounds @fastmath @simd for t in 1:model.T
        M[1, 1, t] = (exp(sumargexp.summ[t]) * (1 - model.Disease.εᵢᵗ[inode.i, t])) * inode.obs[1, t]
        M[1, 2, t] = (1 - exp(sumargexp.summ[t]) * (1 - model.Disease.εᵢᵗ[inode.i, t])) * inode.obs[1, t]
        M[2, 2, t] = exp(sumargexp.sumμ[t]) * inode.obs[2, t]
    end
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
- `rng`: (Optional) A random number generator. Default is `Xoshiro(1234)`.

# Returns
- A matrix representing the epidemic outbreak configuration over time. Each row corresponds to a node, and each column represents a time step. The values in the matrix indicate the state of each node at each time step: 0.0 for Susceptible (S) and 1.0 for Infected (I).
"""
function sim_epidemics(
    model::EpidemicModel{SI,TG};
    patient_zero::Union{Vector{Int},Nothing}=nothing,
    γ::Union{Float64,Nothing}=nothing,
    reject::Bool=false,
    rng::AbstractRNG=Xoshiro(1234)) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    inf₀ = (patient_zero !== nothing)
    (γ===nothing) && (γ=1/model.N)
    if !inf₀
        if reject
            x₀ = [Int8(rand(rng) < γ) for _ in 1:model.N]
            patient_zero = findall(x->x==1, x₀)
            inf₀ = !isempty(patient_zero)
        else
            while !inf₀
                x₀ = [Int8(rand(rng) < γ) for _ in 1:model.N]
                patient_zero = findall(x->x==1, x₀)
                inf₀ = !isempty(patient_zero)
            end
        end
    end

    config = zeros(Int8, model.N, model.T + 1)

    !inf₀ && return config

    config[patient_zero, 1] .+= 1

    hs = zeros(model.N)
    for t in 1:model.T
        hs .= config[:, t]' * model.ν[:, :, t]
        config[:, t+1] .= [x + (1 - x) * rand(rng, Bernoulli(1 - exp(h))) for (x, h) in zip(config[:, t], hs)]
    end
    return config
end




####################################  Forward dynamics  #######################################

function compute_fwd_sumargexp!(inode::Node{SI,TG}, t::Int) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    sumargexp = 0.0

    @inbounds @fastmath for (kindex, k) in enumerate(inode.∂)
        sumargexp += inode.cavities[kindex].m[t] * inode.νs[kindex][t]
    end

    return sumargexp
end


function fwd_cav_update!(model::EpidemicModel{SI,TG}, nodes::Vector{Node{SI,TG}}) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}


    @inbounds @fastmath for t in 1:model.T
        @inbounds @fastmath for inode in nodes
            # compute sumargexp
            sumargexp = compute_fwd_sumargexp!(inode, t)

            # compute cavities
            @inbounds @fastmath for (jindex, j) in enumerate(inode.∂)
                iindex = nodes[j].∂_idx[inode.i]
                nodes[j].cavities[iindex].m[t+1] = nodes[j].cavities[iindex].m[t] + (1-nodes[j].cavities[iindex].m[t])*(1-(1-model.Disease.εᵢᵗ[inode.i,t])*exp(sumargexp-inode.cavities[jindex].m[t]*inode.νs[jindex][t]))
            end
        end
    end
end


function fwd_marg_update!(model::EpidemicModel{SI,TG}, nodes::Vector{Node{SI,TG}}) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    @inbounds @fastmath for t in 1:model.T
        @inbounds @fastmath for inode in nodes
            sumargexp = compute_fwd_sumargexp!(inode, t)
            inode.marg.m[2,t+1] = inode.marg.m[2,t] + (1-inode.marg.m[2,t])*(1-(1-model.Disease.εᵢᵗ[inode.i,t])*exp(sumargexp))
        end
    end
end


"""
    run_fwd_dynamics(model::EpidemicModel{SI,TG}, γ::Vector{Float64}) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

Run the forward-in-time SCDC algorithm for epidemic modeling.

This function performs SCDC forward-in-time epidemic dynamics on the specified epidemic model, using the provided parameters such as the probability of being a patient zero, etc. It iteratively updates cavity messages until convergence or until the maximum number of iterations is reached.

# Arguments
- `model::EpidemicModel{SI,TG}`: The epidemic model to be used.
- `prior::Vector{Float64}`: Prior probabilities of the nodes being infected at time t=0.
"""
function run_fwd_dynamics(model::EpidemicModel{SI,TG}, prior::Vector{Float64}) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    # Format nodes for inference
    nodes = nodes_formatting(model)

    # Initialize message objects
    for inode in nodes
        inode.marg.m[2, 1] = prior[inode.i]
        @inbounds @fastmath for (jindex, j) in enumerate(inode.∂)
            inode.cavities[jindex].m[1] = prior[nodes[j].i]
        end
    end

    # Compute cavity messages
    fwd_cav_update!(model, nodes)
    
    # Compute final marginal probabilities
    fwd_marg_update!(model, nodes)

    return nodes
end