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
        M[1, 1, t] = exp(sumargexp.summ[t] - inode.cavities[jindex].m[t] * inode.νs[jindex][t]) * (1 - model.Disease.εᵢᵗ[inode.i, t]) * inode.obs[1, t]
        M[1, 2, t] = (1 - exp(sumargexp.summ[t] - inode.cavities[jindex].m[t] * inode.νs[jindex][t]) * (1 - model.Disease.εᵢᵗ[inode.i, t])) * inode.obs[1, t]
        M[2, 1, t] = model.Disease.rᵢᵗ[inode.i, t] * exp(sumargexp.sumμ[t] - inode.cavities[jindex].μ[t] * jnode.νs[iindex][t]) * inode.obs[2, t]
        M[2, 2, t] = (1 - model.Disease.rᵢᵗ[inode.i, t]) * exp(sumargexp.sumμ[t] - inode.cavities[jindex].μ[t] * jnode.νs[iindex][t]) * inode.obs[2, t]
    end
end

function fill_transmat_marg!(
    M::Array{Float64,3},
    inode::Node{SIS,TG},
    sumargexp::SumM,
    model::EpidemicModel{SIS,TG}) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    @inbounds @fastmath @simd for t in 1:model.T
        M[1, 1, t] = exp(sumargexp.summ[t]) * (1 - model.Disease.εᵢᵗ[inode.i, t]) * inode.obs[1, t]
        M[1, 2, t] = (1 - exp(sumargexp.summ[t]) * (1 - model.Disease.εᵢᵗ[inode.i, t])) * inode.obs[1, t]
        M[2, 1, t] = model.Disease.rᵢᵗ[inode.i, t] * exp(sumargexp.sumμ[t]) * inode.obs[2, t]
        M[2, 2, t] = (1 - model.Disease.rᵢᵗ[inode.i, t]) * exp(sumargexp.sumμ[t]) * inode.obs[2, t]
    end
end


function W_SIS(x, y, h::Float64, r::Float64, α::Float64)
    @assert x in 0:1 && y in 0:1 "x=$x, y=$y"
    @assert 0<=r<=1 &&  0<=α<=1 "r=$r, α=$α"
    (x==0) && return (y==0)*α*exp(h) + (y==1)*r
    (x==1) && return (y==0)*(1-α*exp(h)) + (y==1)*(1-r)
end


function sample_single(rng::AbstractRNG, i::Int, t::Int, y::Int8, h::Float64, model::EpidemicModel{SIS,TG}) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    p_S = W_SIS(0, y, h, model.Disease.rᵢᵗ[i,t], 1-model.Disease.εᵢᵗ[i,t])

    u = rand(rng)

    (u<p_S) ? (return Int8(0)) : (return Int8(1))
end