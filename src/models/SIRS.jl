struct SIRS <: InfectionModel
    εᵢᵗ::Array{Float64,2} # autoinfection
    rᵢᵗ::Array{Float64,2} # I->R
    σᵢᵗ::Array{Float64,2} # R->S
end
n_states(X::SIRS) = 3

"""
    SIRS(
        εᵢᵗ::Union{Float64,Array{Float64,2}},
        rᵢᵗ::Union{Float64,Array{Float64,2}},
        σᵢᵗ::Union{Float64,Array{Float64,2}},
        NV::Int,
        T::Int)

Defines the SIS infection model.

# Arguments
* `εᵢᵗ`: Self-infection probability. Can be either a Float64 (constant over all nodes and times) or a NVxT matrix.
* `rᵢᵗ`: Recovery probability. Can be either a Float64 (constant over all nodes and times) or a NVxT matrix.
* `σᵢᵗ`: Loss of immunity probability. Can be either a Float64 (constant over all nodes and times) or a NVxT matrix.
* `NV`: Number of nodes of the contact graph.
* `T`: Number of time-steps.
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

    return SIRS(εᵢᵗ, rᵢᵗ, σᵢᵗ)
end

function nodes_formatting(
    model::EpidemicModel{SIRS,AbstractGraph}, 
    obsprob::Function)

    nodes = Vector{Node{SIRS,AbstractGraph}}()

    for i in 1:model.N
        obs = ones(3, model.T + 1)
        obs[1, :] = [obsprob(Ob, 0.0) for Ob in model.obsmat[i, :]]
        obs[2, :] = [obsprob(Ob, 1.0) for Ob in model.obsmat[i, :]]
        obs[3, :] = [obsprob(Ob, 2.0) for Ob in model.obsmat[i, :]]

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
        obs[1, :] = [obsprob(Ob, 0.0) for Ob in model.obsmat[i, :]]
        obs[2, :] = [obsprob(Ob, 1.0) for Ob in model.obsmat[i, :]]
        obs[3, :] = [obsprob(Ob, 2.0) for Ob in model.obsmat[i, :]]

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
    inode::Node{SIRS,TG},
    iindex::Int,
    jnode::Node{SIRS,TG},
    jindex::Int,
    sumargexp::SumM,
    infectionmodel::SIRS) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    M[1, 1, :] .= (exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1])).*(1 .- infectionmodel.εᵢᵗ[inode.i, :]) .* inode.obs[1, 1:end-1]
    M[1, 2, :] .= (1 .- exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1]).*(1 .- infectionmodel.εᵢᵗ[inode.i, :])) .* inode.obs[1, 1:end-1]
    M[2, 2, :] .= (1 .- infectionmodel.rᵢᵗ[inode.i, :]) .* exp.(sumargexp.sumμ .- inode.cavities[jindex].μ .* jnode.νs[iindex][1:end-1]) .* inode.obs[2, 1:end-1]
    M[2, 3, :] .= infectionmodel.rᵢᵗ[inode.i, :] .* exp.(sumargexp.sumμ .- inode.cavities[jindex].μ .* jnode.νs[iindex][1:end-1]) .* inode.obs[2, 1:end-1]
    M[3, 1, :] .= infectionmodel.σᵢᵗ[inode.i, :] .* inode.obs[3, 1:end-1]
    M[3, 3, :] .= (1 .- infectionmodel.σᵢᵗ[inode.i, :]) .* inode.obs[3, 1:end-1]
end

function fill_transmat_marg!(
    M::Array{Float64,3},
    inode::Node{SIRS,TG},
    sumargexp::SumM,
    infectionmodel::SIRS) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    M[1, 1, :] .= (exp.(sumargexp.summ[1:end-1])).*(1 .- infectionmodel.εᵢᵗ[inode.i, :]) .* inode.obs[1, 1:end-1]
    M[1, 2, :] .= (1 .- exp.(sumargexp.summ[1:end-1]).*(1 .- infectionmodel.εᵢᵗ[inode.i, :])) .* inode.obs[1, 1:end-1]
    M[2, 2, :] .= (1 .- infectionmodel.rᵢᵗ[inode.i, :]) .* exp.(sumargexp.sumμ) .* inode.obs[2, 1:end-1]
    M[2, 3, :] .= infectionmodel.rᵢᵗ[inode.i, :] .* exp.(sumargexp.sumμ) .* inode.obs[2, 1:end-1]
    M[3, 1, :] .= infectionmodel.σᵢᵗ[inode.i, :] .* inode.obs[3, 1:end-1]
    M[3, 3, :] .= (1 .- infectionmodel.σᵢᵗ[inode.i, :]) .* inode.obs[3, 1:end-1]
end

"""
    sim_epidemics(
        model::EpidemicModel{SIRS,TG};
        patient_zero::Union{Vector{Int},Nothing}=nothing,
        γ::Union{Float64,Nothing}=nothing) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

Simulates the epidemic outbreak given a SIRS model. 

# Arguments
* `model`: The SIRS epidemic model.
* `patient_zero`: Vector of patients zero. Default is "nothing", meaning that the patients zero are chosen at random with probability γ.
* `γ`: Probability of being a patient zero. Default is "nothing", meaning that it is fixed to0 1/NV, where NV is the number of nodes of the contact graph.
"""
function sim_epidemics(
    model::EpidemicModel{SIRS,TG};
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

    config[patient_zero,1] .+= 1.0

    function W_SIRS(x::Float64, y::Float64, h::Float64, r::Float64, σ::Float64)
        if x == 0.0
            return (y == 0.0) * exp(h) + (y == 2.0) * σ
        elseif x == 1.0
            return (y == 0.0) * (1 - exp(h)) + (y == 1.0) * (1 - r)
        elseif x == 2.0
            return (y == 1.0) * r + (y == 2.0) * (1 - σ)
        else
            throw(ArgumentError("Invalid value for y"))
        end
    end
    hs = zeros(model.N)
    for t in 1:model.T
        hs = [Float64(x == 1.0) for x in config[:, t]]' * model.ν[:, :, t]
        config[:, t+1] = [
            if (u <= W_SIRS(0.0, x, h, r, σ))
                0.0
            elseif (W_SIRS(0.0, x, h, r, σ) < u <= W_SIRS(0.0, x, h, r, σ) + W_SIRS(1.0, x, h, r, σ))
                1.0
            else
                2.0
            end for (x, h, r, σ, u) in zip(config[:, t], hs, model.Disease.rᵢᵗ[:, t], model.Disease.σᵢᵗ[:, t], rand(Float64, model.N))
        ]
    end
    return config
end
        