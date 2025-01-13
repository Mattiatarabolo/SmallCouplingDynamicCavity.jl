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


function W_SIRS(x, y, h::Float64, r::Float64, σ::Float64, α::Float64)
    @assert x in 0:2 && y in 0:2 "x=$x, y=$y"
    @assert 0<=r<=1 && 0<=σ<=1 && 0<=α<=1 "r=$r, σ=$σ, α=$α"
    (x==0) && return (y==0)*α*exp(h) + (y==2)*σ
    (x==1) && return (y==0)*(1-α*exp(h)) + (y==1)*(1-r)
    (x==2) && return (y==1)*r + (y==2)*(1-σ)
end


function sample_single(rng::AbstractRNG, i::Int, t::Int, y::Int8, h::Float64, model::EpidemicModel{SIRS,TG}) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    p_S = W_SIRS(0, y, h, model.Disease.rᵢᵗ[i,t], model.Disease.σᵢᵗ[i,t], 1-model.Disease.εᵢᵗ[i,t])
    p_I = W_SIRS(1, y, h, model.Disease.rᵢᵗ[i,t], model.Disease.σᵢᵗ[i,t], 1-model.Disease.εᵢᵗ[i,t])
    p_R = W_SIRS(2, y, h, model.Disease.rᵢᵗ[i,t], model.Disease.σᵢᵗ[i,t], 1-model.Disease.εᵢᵗ[i,t])

    pcum = cumsum([p_S, p_I, p_R])

    u = rand(rng)

    (u<pcum[1]) && return Int8(0)
    (pcum[1]<=u<pcum[2]) && return Int8(1)
    (pcum[2]<=u) && return Int8(2)
end
  