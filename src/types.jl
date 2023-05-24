function TransMat(
    T::Int, 
    nr_states::Int)

    return zeros(nr_states, nr_states, T)
end

struct FBm
    fwm::Array{Float64,2}
    bwm::Array{Float64,2}
end

function FBm(
    T::Int, 
    nr_states::Int)

    return FBm(ones(nr_states, T + 1), ones(nr_states, T + 1))
end

mutable struct SumM
    summ::Vector{Float64}
    sumμ::Vector{Float64}
end

function SumM(T::Int)
    return SumM(zeros(T + 1), zeros(T))
end

mutable struct Updmess
    lognumm::Array{Float64,2}
    lognumμ::Vector{Float64}
    signμ::Vector{Float64}
    logZ::Vector{Float64}
end

function Updmess(
    T::Int, 
    nr_states::Int)
    
    return Updmess(zeros(nr_states, T + 1), zeros(T), ones(T), zeros(T + 1))
end

mutable struct Message
    i::Int
    j::Int
    m::Vector{Float64} #message m_i\j
    μ::Vector{Float64} #message μ_i\j
end

function Message(
    i::Int, 
    j::Int, 
    T::Int)
    return Message(i, j, ones(T + 1) ./ (T + 1), zeros(T))
end

mutable struct Marginal
    i::Int
    m::Array{Float64, 2} #marginal p_i(x_i^t)
    μ::Vector{Float64} #marginal μ_i^t
end

function Marginal(
    i::Int,
    T::Int,
    nr_states::Int)
    return Marginal(i, ones(nr_states, T + 1) ./ (T + 1), zeros(T))
end
mutable struct Node
    i::Int
    ∂::Vector{Int}
    ∂_idx::Dict{Int,Int}
    marg::Marginal
    cavities::Vector{Message}
    ρs::Vector{FBm}
    νs::Vector{Vector{Float64}}
    rᵢᵗ::Union{Vector{Float64},Nothing}
    σᵢᵗ::Union{Vector{Float64},Nothing}
    obs::Array{Float64,2}
end

function Node(
    i::Int, 
    ∂::Vector{Int}, 
    T::Int, 
    νs::Vector{Vector{Float64}}, 
    obs::Array{Float64,2},
    nr_states::Int; 
    rᵢᵗ::Union{Vector{Float64},Nothing}=nothing, 
    σᵢᵗ::Union{Vector{Float64},Nothing}=nothing)

    return Node(
        i,
        ∂,
        Dict(∂[idx] => idx for idx = 1:length(∂)),
        Marginal(i, T, nr_states),
        collect([Message(i, j, T) for j in ∂]),
        collect([FBm(T, nr_states) for _ in 1:length(∂)]),
        νs,
        rᵢᵗ,
        σᵢᵗ,
        obs)
end
