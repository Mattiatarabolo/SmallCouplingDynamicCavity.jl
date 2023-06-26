function TransMat(
    T::Int, 
    infectionmodel::TI) where {TI <:InfectionModel}

    return zeros(n_states(infectionmodel), n_states(infectionmodel), T)
end

struct FBm
    fwm::Array{Float64,2}
    bwm::Array{Float64,2}
end

function FBm(
    T::Int, 
    infectionmodel::TI) where {TI <:InfectionModel}

    return FBm(ones(n_states(infectionmodel), T + 1), ones(n_states(infectionmodel), T + 1))
end

struct SumM
    summ::Vector{Float64}
    sumμ::Vector{Float64}
end

function SumM(T::Int)
    return SumM(zeros(T + 1), zeros(T))
end

struct Updmess
    lognumm::Array{Float64,2}
    lognumμ::Vector{Float64}
    signμ::Vector{Float64}
    logZ::Vector{Float64}
end

function Updmess(
    T::Int, 
    infectionmodel::TI) where {TI <:InfectionModel}
    
    return Updmess(zeros(n_states(infectionmodel), T + 1), zeros(T), ones(T), zeros(T + 1))
end

struct Message
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

struct Marginal
    i::Int
    m::Array{Float64, 2} #marginal p_i(x_i^t)
    μ::Vector{Float64} #marginal μ_i^t
end

function Marginal(
    i::Int,
    T::Int,
    infectionmodel::TI) where {TI <:InfectionModel}
    return Marginal(i, ones(n_states(infectionmodel), T + 1) ./ (T + 1), zeros(T))
end

struct Node{TI <: InfectionModel}
    i::Int
    ∂::Vector{Int}
    ∂_idx::Dict{Int,Int}
    marg::Marginal
    cavities::Vector{Message}
    ρs::Vector{FBm}
    νs::Vector{Vector{Float64}}
    obs::Array{Float64,2}
    infectionmodel::TI
end

function Node(
    i::Int, 
    ∂::Vector{Int}, 
    T::Int, 
    νs::Vector{Vector{Float64}}, 
    obs::Array{Float64,2},
    infectionmodel::TI) where {TI <:InfectionModel}

    return Node{TI}(
        i,
        ∂,
        Dict(∂[idx] => idx for idx = 1:length(∂)),
        Marginal(i, T, infectionmodel),
        collect([Message(i, j, T) for j in ∂]),
        collect([FBm(T, infectionmodel) for _ in 1:length(∂)]),
        νs,
        obs,
        infectionmodel)
end


struct EpidemicModel{TI<:InfectionModel}
    Disease::TI
    G::SimpleGraph{Int64}
    T::Int
    ν::Array{Float64, 3}
    obsmat::Matrix{Float64}
end

function EpidemicModel(infectionmodel::TI, G::SimpleGraph{Int64}, T::Int, ν::Array{Float64, 3}, obs::Matrix{Float64}) where {TI <: InfectionModel}
    return EpidemicModel{TI}(infectionmodel, G, T, ν, obs)
end

function EpidemicModel(infectionmodel::TI, G::SimpleGraph{Int64}, T::Int, ν::Array{Float64, 3}) where {TI <: InfectionModel}
    return EpidemicModel{TI}(infectionmodel, G, T, ν, zeros(nv(G),T+1))
end