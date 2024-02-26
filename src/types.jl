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

nv(G::Vector{<:AbstractGraph}) = Graphs.nv(G[1]) #convenience function for evolving graphs

struct EpidemicModel{TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    Disease::TI
    G::TG
    T::Int
    ν::Array{Float64, 3}
    obsmat::Matrix{Float64}


    """
    EpidemicModel(infectionmodel, G, T::Int, ν::Array{Float64, 3}, obs::Matrix{Float64})

    Defines the epidemic model.

    # Arguments
    * `infectionmodel`: Infection model. Currently are implemented SI, SIR, SIS and SIRS infection models.
    * `G`: Contact graph. Can be either an AbstractGraph (contact graph constant over time) or a T+1 vector of AbstractGraph (time varying contact graph)
    * `T`: Number of time-steps.
    * `ν`: Infection couplings. It is a NVxNVx(T+1) Array where νᵗᵢⱼ=log(1-λᵗᵢⱼ), with λᵗᵢⱼ being the infection probability from individual i to individual j at time t.
    * `obs`: Observations matrix. It is a NVx(T+1) Matrix, where obsᵗᵢ is the observation of individual i at time t: it is equal to -1.0 if not observed, 0.0 if S, 1.0 if I, 2.0 if R (only for SIR and SIRS).
    """
    function EpidemicModel(infectionmodel::TI, G::TG, T::Int, ν::Array{Float64, 3}, obs::Matrix{Float64}) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
        new{TI,TG}(infectionmodel, G, T, ν, obs)
    end

    """
    EpidemicModel(infectionmodel, G, T::Int, ν::Array{Float64, 3})

    Define the epidemic model.

    # Arguments
    * `infectionmodel`: Infection model. Currently are implemented SI, SIR, SIS and SIRS infection models.
    * `G`: Contact graph. Can be either an AbstractGraph (contact graph constant over time) or a T+1 vector of AbstractGraph (time varying contact graph)
    * `T`: Number of time-steps.
    * `ν`: Infection couplings. It is a NVxNVx(T+1) Array where νᵗᵢⱼ=log(1-λᵗᵢⱼ), with λᵗᵢⱼ being the infection probability from individual i to individual j at time t.
    """
    function EpidemicModel(infectionmodel::TI, G::TG, T::Int, ν::Array{Float64, 3}) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
        new{TI,TG}(infectionmodel, G, T, ν, zeros(nv(G),T+1))
    end
end


struct Node{TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    i::Int
    ∂::Vector{Int}
    ∂_idx::Dict{Int,Int}
    marg::Marginal
    cavities::Vector{Message}
    ρs::Vector{FBm}
    νs::Vector{Vector{Float64}}
    obs::Array{Float64,2}
    model::EpidemicModel{TI,TG}

    function Node(
        i::Int, 
        ∂::Vector{Int}, 
        T::Int, 
        νs::Vector{Vector{Float64}}, 
        obs::Array{Float64,2},
        model::EpidemicModel{TI,TG}) where {TI <:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
        new{TI,TG}(
            i,
            ∂,
            Dict(∂[idx] => idx for idx = 1:length(∂)),
            Marginal(i, T, model.Disease),
            collect([Message(i, j, T) for j in ∂]),
            collect([FBm(T, model.Disease) for _ in ∂]),
            νs,
            obs,
            model)
    end
end




