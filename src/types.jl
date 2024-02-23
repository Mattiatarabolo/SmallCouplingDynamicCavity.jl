function TransMat(
    T::Int, 
    infectionmodel::TI) where {TI <:InfectionModel}

    return zeros(n_states(infectionmodel), n_states(infectionmodel), T)
end


struct FBm
    fwm::Array{Float64,2}
    bwm::Array{Float64,2}

    function FBm(
        T::Int, 
        infectionmodel::TI) where {TI <:InfectionModel}
    
        new(ones(n_states(infectionmodel), T + 1), ones(n_states(infectionmodel), T + 1))
    end

end


struct SumM
    summ::Vector{Float64}
    sumμ::Vector{Float64}

    function SumM(T::Int)
        new(zeros(T + 1), zeros(T))
    end
end


struct Updmess
    lognumm::Array{Float64,2}
    lognumμ::Vector{Float64}
    signμ::Vector{Float64}
    logZ::Vector{Float64}
    
    function Updmess(
        T::Int, 
        infectionmodel::TI) where {TI <:InfectionModel}
        
        new(zeros(n_states(infectionmodel), T + 1), zeros(T), ones(T), zeros(T + 1))
    end
end


struct Message
    i::Int
    j::Int
    m::Vector{Float64} #message m_i\j
    μ::Vector{Float64} #message μ_i\j
    
    function Message(
        i::Int, 
        j::Int, 
        T::Int)
        new(i, j, ones(T + 1) ./ (T + 1), zeros(T))
    end
end


struct Marginal
    i::Int
    m::Array{Float64, 2} #marginal p_i(x_i^t)
    μ::Vector{Float64} #marginal μ_i^t

    function Marginal(
        i::Int,
        T::Int,
        infectionmodel::TI) where {TI <:InfectionModel}
        new(i, ones(n_states(infectionmodel), T + 1) ./ (T + 1), zeros(T))
    end
end


struct Node{TI<:InfectionModel,TG<:Union{AbstractGraph,Vector{<:AbstractGraph}}}
    i::Int
    ∂::Vector{Int}
    ∂_idx::Dict{Int,Int}
    marg::Marginal
    cavities::Vector{Message}
    ρs::Vector{FBm}
    νs::Vector{Float64}
    obs::Array{Float64,2}
    model::EpidemicModel{TI,TG}

    function Node(
        i::Int, 
        ∂::Vector{Int}, 
        T::Int, 
        νs::Vector{Vector{Float64}}, 
        obs::Array{Float64,2},
        model::EpidemicModel{TI,TG}) where {TI <:InfectionModel,TG<:Union{AbstractGraph,Vector{<:AbstractGraph}}}
    
        new{TI,TG}(
            i,
            ∂,
            Dict(∂[idx] => idx for idx = 1:length(∂)),
            Marginal(i, T, model.Disease),
            collect([Message(i, j, T) for j in ∂]),
            collect([FBm(T, model.Disease) for _ in 1:length(∂)]),
            νs,
            obs,
            model)
    end
end


struct EpidemicModel{TI<:InfectionModel,TG<:Union{AbstractGraph,Vector{<:AbstractGraph}}}
    Disease::TI
    G::TG
    T::Int
    ν::Array{Float64, 3}
    obsmat::Matrix{Float64}

    function EpidemicModel(infectionmodel::TI, G::TG, T::Int, ν::Array{Float64, 3}, obs::Matrix{Float64}) where {TI<:InfectionModel,TG<:Union{AbstractGraph,Vector{<:AbstractGraph}}}
        new{TI,TG}(infectionmodel, G, T, ν, obs)
    end

    function EpidemicModel(infectionmodel::TI, G::TG, T::Int, ν::Array{Float64, 3}) where {TI<:InfectionModel,TG<:Union{AbstractGraph,Vector{<:AbstractGraph}}}
        new{TI,TG}(infectionmodel, G, T, ν, zeros(nv(G),T+1))
    end
end





