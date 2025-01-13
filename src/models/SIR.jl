"""
    struct SIR <: InfectionModel
        εᵢᵗ::Array{Float64, 2} # Autoinfection probabilities
        rᵢᵗ::Array{Float64, 2} # Recovery probabilities
    end

The `SIR` struct represents the SIR (Susceptible-Infected-Recovered) infection model.

# Fields
- `εᵢᵗ`: An NVxT array representing the self-infection probabilities over time, where NV is the number of nodes and T is the number of time-steps. Each element εᵢᵗ[i, t] denotes the probability of node i infecting itself at time t.
- `rᵢᵗ`: An NVxT array representing the recovery probabilities over time, where NV is the number of nodes and T is the number of time-steps. Each element rᵢᵗ[i, t] denotes the probability of node i recovering from infection at time t.

"""
struct SIR <: InfectionModel
    εᵢᵗ::Array{Float64, 2} # Autoinfection probabilities
    rᵢᵗ::Array{Float64, 2} # Recovery probabilities


    """
        SIR(
            εᵢᵗ::Union{Float64,Array{Float64,2}},
            rᵢᵗ::Union{Float64,Array{Float64,2}},
            NV::Int,
            T::Int)

    Defines the SIR infection model.

    # Arguments
    - `εᵢᵗ`: Self-infection probability. Can be either a Float64 (constant over all nodes and times) or a NVxT matrix.
    - `rᵢᵗ`: Recovery probability. Can be either a Float64 (constant over all nodes and times) or a NVxT matrix.
    - `NV`: Number of nodes of the contact graph.
    - `T`: Number of time-steps.

    # Returns
    - An instance of the SIR struct representing the SIR infection model.
    """
    function SIR(
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

n_states(X::SIR) = 3



function nodes_formatting(
    model::EpidemicModel{SIR,TG}, 
    obsprob::Function) where {TG<:AbstractGraph}

    nodes = Vector{Node{SIR,TG}}()

    for i in 1:model.N
        obs = ones(3, model.T + 1)
        @inbounds @fastmath @simd for t in 1:model.T+1
            obs[1, t] = obsprob(model.obsmat[i,t], 0)
            obs[2, t] = obsprob(model.obsmat[i,t], 1)
            obs[3, t] = obsprob(model.obsmat[i,t], 2)
        end

        ∂ = neighbors(model.G, i)

        ν∂ = [model.ν[k, i, :] for k in ∂]

        push!(nodes, Node(i, ∂, model.T, ν∂, obs, model))
    end
    return collect(nodes)
end


function nodes_formatting(
    model::EpidemicModel{SIR,TG}, 
    obsprob::Function) where {TG<:Vector{<:AbstractGraph}}

    nodes = Vector{Node{SIR,TG}}()

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
    inode::Node{SIR,TG},
    iindex::Int,
    jnode::Node{SIR,TG},
    jindex::Int,
    sumargexp::SumM,
    model::EpidemicModel{SIR,TG}) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    @inbounds @fastmath @simd for t in 1:model.T
        M[1, 1, t] = exp(sumargexp.summ[t] - inode.cavities[jindex].m[t] * inode.νs[jindex][t])*(1 - model.Disease.εᵢᵗ[inode.i, t]) * inode.obs[1, t]
        M[1, 2, t] = (1 - exp(sumargexp.summ[t] - inode.cavities[jindex].m[t] * inode.νs[jindex][t])*(1 - model.Disease.εᵢᵗ[inode.i, t])) * inode.obs[1, t]
        M[2, 2, t] = (1 - model.Disease.rᵢᵗ[inode.i, t]) * exp(sumargexp.sumμ[t] - inode.cavities[jindex].μ[t] * jnode.νs[iindex][t]) * inode.obs[2, t]
        M[2, 3, t] = model.Disease.rᵢᵗ[inode.i, t] * exp(sumargexp.sumμ[t] - inode.cavities[jindex].μ[t] * jnode.νs[iindex][t]) * inode.obs[2, t]
        M[3, 3, t] = inode.obs[3, t]
    end
end

function fill_transmat_marg!(
    M::Array{Float64,3},
    inode::Node{SIR,TG},
    sumargexp::SumM,
    model::EpidemicModel{SIR,TG}) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    @inbounds @fastmath @simd for t in 1:model.T
        M[1, 1, t] = (exp(sumargexp.summ[t])*(1 - model.Disease.εᵢᵗ[inode.i, t])) * inode.obs[1, t]
        M[1, 2, t] = (1 - exp(sumargexp.summ[t])*(1 - model.Disease.εᵢᵗ[inode.i, t])) * inode.obs[1, t]
        M[2, 2, t] = (1 - model.Disease.rᵢᵗ[inode.i, t]) * exp(sumargexp.sumμ[t]) * inode.obs[2, t]
        M[2, 3, t] = model.Disease.rᵢᵗ[inode.i, t] * exp(sumargexp.sumμ[t]) * inode.obs[2, t]
        M[3, 3, t] = inode.obs[3, t]
    end
end

function W_SIR(x, y, h::Float64, r::Float64, α::Float64)
    @assert x in 0:2 && y in 0:2 "x=$x, y=$y"
    @assert 0<=r<=1 && 0<=α<=1 "r=$r, α=$α"
    (x==0) && return (y==0)*α*exp(h)
    (x==1) && return (y==0)*(1-α*exp(h)) + (y==1)*(1-r)
    (x==2) && return (y==1)*r + (y==2)
end


function sample_single(rng::AbstractRNG, i::Int, t::Int, y::Int8, h::Float64, model::EpidemicModel{SIR,TG}) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    p_S = W_SIR(0, y, h, model.Disease.rᵢᵗ[i,t], 1-model.Disease.εᵢᵗ[i,t])
    p_I = W_SIR(1, y, h, model.Disease.rᵢᵗ[i,t], 1-model.Disease.εᵢᵗ[i,t])
    p_R = W_SIR(2, y, h, model.Disease.rᵢᵗ[i,t], 1-model.Disease.εᵢᵗ[i,t])

    pcum = cumsum([p_S, p_I, p_R])

    u = rand(rng)

    (u<pcum[1]) && return Int8(0)
    (pcum[1]<=u<pcum[2]) && return Int8(1)
    (pcum[2]<=u) && return Int8(2)
end