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
    
        new(zeros(n_states(infectionmodel), T + 1), zeros(n_states(infectionmodel), T + 1))
    end
end


struct SumM
    summ::Vector{Float64}
    sumμ::Vector{Float64}

    function SumM(T::Int)
        new(zeros(T), zeros(T))
    end
end


"""
    Message

Cavity messages mᵢⱼ and μᵢⱼ.

# Fields

$(TYPEDFIELDS)

"""
struct Message
    """Index of the node i."""
    i::Int
    """Index of the node j."""
    j::Int
    """T+1 Vector of messages mᵢⱼ over time."""
    m::Vector{Float64} #message m_i\j
    """T+1 Vector of messages μᵢⱼ over time."""
    μ::Vector{Float64} #message μ_i\j
    
    function Message(
        i::Int, 
        j::Int, 
        T::Int;
        zero_mess=false)
        if zero_mess
            new(i, j, zeros(T + 1), zeros(T))
        else
            new(i, j, ones(T + 1), zeros(T))
        end
    end
end

"""
    Marginal

Marginals pᵢ(xᵢ) and μᵢ.

# Fields

$(TYPEDFIELDS)

"""
struct Marginal
    """Index of the node i."""
    i::Int
    """(nstates)x(T+1) Matrix of marginals over time, where nstates is the number of states that the infection model has."""
    m::Array{Float64, 2} #marginal p_i(x_i^t)
    """T+1 Vector of marginals μᵢ over time."""
    μ::Vector{Float64} #marginal μ_i^t

    function Marginal(
        i::Int,
        T::Int,
        infectionmodel::TI) where {TI <:InfectionModel}
        new(i, ones(n_states(infectionmodel), T + 1), zeros(T))
    end
end

"""
    EpidemicModel

Epidemic model containing all the informations of the system.

# Fields

$(TYPEDFIELDS)

"""
struct EpidemicModel{TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    """Infection model. Currently are implemented SI, SIR, SIS and SIRS infection models."""
    Disease::TI
    """Contact graph. It can be either an AbstractGraph (contact graph constant over time) or a Vector of AbstractGraph (time varying contact graph)."""
    G::TG
    """Number of nodes of the contact graph."""
    N::Int
    """Number of time steps."""
    T::Int
    """Infection couplings. It is a NVxNVx(T+1) Array where νᵗᵢⱼ=log(1-λᵗᵢⱼ), with λᵗᵢⱼ being the infection probability from individual i to individual j at time t."""
    ν::Array{Float64, 3}
    """Observations matrix. It is a NVx(T+1) Matrix, where obsᵗᵢ is the observation of individual i at time t: it is equal to -1.0 if not observed, 0.0 if S, 1.0 if I, 2.0 if R (only for SIR and SIRS)."""
    obsmat::Matrix{Int8}

    @doc """
        EpidemicModel(
            infectionmodel::TI, 
            G::TG, T::Int, 
            ν::Array{Float64, 3}, 
            obs::Matrix{Int8}) where {TI<:InfectionModel,TG<:AbstractGraph}

    Defines the epidemic model.

    This function defines an epidemic model based on the provided infection model, contact graph, time steps, infection couplings, and observation matrix.

    # Arguments
    - `infectionmodel`: The infection model. Currently implemented models include SI, SIR, SIS, and SIRS infection models.
    - `G`: The contact graph. It should be an AbstractGraph representing the contact graph, which is time-varying.
    - `T`: The number of time-steps.
    - `ν`: The infection couplings. It should be a 3-dimensional array of size NVxNVx(T+1), where νᵗᵢⱼ=log(1-λᵗᵢⱼ), with λᵗᵢⱼ being the infection probability from individual i to individual j at time t.
    - `obs`: The observations matrix. It should be a NVx(T+1) matrix, where obsᵗᵢ is the observation of individual i at time t: it is equal to -1.0 if not observed, 0.0 if S, 1.0 if I, 2.0 if R (only for SIR and SIRS).

    # Returns
    - `EpidemicModel`: An [`EpidemicModel`](@ref) object representing the defined epidemic model.
    """
    function EpidemicModel(
        infectionmodel::TI, 
        G::TG, T::Int, 
        ν::Array{Float64, 3}, 
        obs::Matrix{Int8}) where {TI<:InfectionModel,TG<:AbstractGraph}
        new{TI,TG}(infectionmodel, G, nv(G), T, ν, obs)
    end

    @doc """
        EpidemicModel(
            infectionmodel::TI, 
            G::TG, T::Int, 
            ν::Array{Float64, 3}, 
            obs::Matrix{Int8}) where {TI<:InfectionModel,TG<:Vector{<:AbstractGraph}}

    Defines the epidemic model.

    This function defines an epidemic model based on the provided infection model, contact graph, time steps, infection couplings, and observation matrix.

        # Arguments
        - `infectionmodel`: The infection model. Currently implemented models include SI, SIR, SIS, and SIRS infection models.
        - `G`: The contact graph. It should be a T+1 vector of AbstractGraph representing the time-varying contact graph.
        - `T`: The number of time-steps.
        - `ν`: The infection couplings. It should be a 3-dimensional array of size NVxNVx(T+1), where νᵗᵢⱼ=log(1-λᵗᵢⱼ), with λᵗᵢⱼ being the infection probability from individual i to individual j at time t.
        - `obs`: The observations obsmatrix. It should be a NVx(T+1) matrix, where obsᵗᵢ is the observation of individual i at time t: it is equal to -1.0 if not observed, 0.0 if S, 1.0 if I, 2.0 if R (only for SIR and SIRS).
    
        # Returns
        - `EpidemicModel`: An [`EpidemicModel`](@ref) object representing the defined epidemic model.
    """
    function EpidemicModel(
        infectionmodel::TI, 
        G::TG, T::Int, 
        ν::Array{Float64, 3}, 
        obs::Matrix{Int8}) where {TI<:InfectionModel,TG<:Vector{<:AbstractGraph}}
        new{TI,TG}(infectionmodel, G, nv(G[1]), T, ν, obs)
    end

    @doc """
        EpidemicModel(
            infectionmodel::TI, 
            G::TG, T::Int, 
            ν::Array{Float64, 3}) where {TI<:InfectionModel,TG<:AbstractGraph}

    Define an epidemic model.

    This function defines an epidemic model based on the provided infection model, contact graph, time steps, and infection couplings. It initializes the observation matrix with zeros.

    # Arguments
    - `infectionmodel`: The infection model. Currently implemented models include SI, SIR, SIS, and SIRS infection models.
    - `G`: The contact graph. It should be an AbstractGraph representing the contact graph, which is constant over time.
    - `T`: The number of time-steps.
    - `ν`: The infection couplings. It should be a 3-dimensional array of size NVxNVx(T+1), where νᵗᵢⱼ=log(1-λᵗᵢⱼ), with λᵗᵢⱼ being the infection probability from individual i to individual j at time t.

    # Returns
    - `EpidemicModel`: An [`EpidemicModel`](@ref) object representing the defined epidemic model.
    """
    function EpidemicModel(
        infectionmodel::TI, 
        G::TG, 
        T::Int, 
        ν::Array{Float64, 3}) where {TI<:InfectionModel,TG<:AbstractGraph}
        new{TI,TG}(infectionmodel, G, nv(G), T, ν, ones(Int8, nv(G),T+1)*Int8(-1))
    end


    @doc """
        EpidemicModel(
            infectionmodel::TI, 
            G::TG, T::Int, 
            ν::Array{Float64, 3}) where {TI<:InfectionModel,TG<:Vector{<:AbstractGraph}}

    Define an epidemic model.

    This function defines an epidemic model based on the provided infection model, time-varying contact graph, time steps, and infection couplings. It initializes the observation matrix with zeros.

    # Arguments
    - `infectionmodel`: The infection model. Currently implemented models include SI, SIR, SIS, and SIRS infection models.
    - `G`: The contact graph. It should be a T+1 vector of AbstractGraph representing the time-varying contact graph.
    - `T`: The number of time-steps.
    - `ν`: The infection couplings. It should be a 3-dimensional array of size NVxNVx(T+1), where νᵗᵢⱼ=log(1-λᵗᵢⱼ), with λᵗᵢⱼ being the infection probability from individual i to individual j at time t.

    # Returns
    - `EpidemicModel`: An [`EpidemicModel`](@ref) object representing the defined epidemic model.
    """
    function EpidemicModel(
        infectionmodel::TI, 
        G::TG, 
        T::Int, 
        ν::Array{Float64, 3}) where {TI<:InfectionModel,TG<:Vector{<:AbstractGraph}}
        new{TI,TG}(infectionmodel, G, nv(G[1]), T, ν, zeros(nv(G[1]),T+1))
    end

end

"""
    Node

Type containing all the informations of a node. 

# Fields

$(TYPEDFIELDS)

"""
struct Node{TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    """Index of the node."""
    i::Int
    """List of neighbours. If the underlying contact graph is varying in time it is the union of all the neighbours over time."""
    ∂::Vector{Int}
    """Only for developers."""
    ∂_idx::Dict{Int,Int}
    """Marginals of the node. It is a [`Marginal`](@ref) type."""
    marg::Marginal
    """Cavities messages entering into the node from its neigbours. It is a vector of [`Message`](@ref), each one corresponding to a neighbour with the same order of ∂."""
    cavities::Vector{Message}
    """Infection couplings of the neighbours against the node."""
    νs::Vector{Vector{Float64}}
    """Observation probability matrix."""
    obs::Matrix{Float64}
    """Epidemic model. It is a [`EpidemicModel`](@ref) type."""
    model::EpidemicModel{TI,TG}

    function Node(
        i::Int, 
        ∂::Vector{Int}, 
        T::Int, 
        νs::Vector{Vector{Float64}}, 
        obs::Matrix{Float64},
        model::EpidemicModel{TI,TG}) where {TI <:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
        new{TI,TG}(
            i,
            ∂,
            Dict(∂[idx] => idx for idx = 1:length(∂)),
            Marginal(i, T, model.Disease),
            collect([Message(j, i, T) for j in ∂]),
            νs,
            obs,
            model)
    end
end




