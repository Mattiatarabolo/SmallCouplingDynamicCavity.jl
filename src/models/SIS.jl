struct SIS <: InfectionModel
    εᵢᵗ::Array{Float64,2} # autoinfection
    rᵢᵗ::Array{Float64,2} # I->R
end
n_states(X::SIS) = 2

"""
    SIS(
        εᵢᵗ::Union{Float64,Array{Float64,2}},
        rᵢᵗ::Union{Float64,Array{Float64,2}},
        NV::Int,
        T::Int)

Defines the SIS infection model.

# Arguments
* `εᵢᵗ`: Self-infection probability. Can be either a Float64 (constant over all nodes and times) or a NVxT matrix.
* `rᵢᵗ`: Recovery probability. Can be either a Float64 (constant over all nodes and times) or a NVxT matrix.
* `NV`: Number of nodes of the contact graph.
* `T`: Number of time-steps.
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

    return SIS(εᵢᵗ, rᵢᵗ)
end

function nodes_formatting(
    model::EpidemicModel{SIS,TG}, 
    obsprob::Function) where {TG<:AbstractGraph}

    nodes = Vector{Node{SIS,TG}}()

    for i in 1:model.N
        obs = ones(2, model.T + 1)
        obs[1, :] = [obsprob(Ob, 0.0) for Ob in model.obsmat[i, :]]
        obs[2, :] = [obsprob(Ob, 1.0) for Ob in model.obsmat[i, :]]

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
        obs[1, :] = [obsprob(Ob, 0.0) for Ob in model.obsmat[i, :]]
        obs[2, :] = [obsprob(Ob, 1.0) for Ob in model.obsmat[i, :]]

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
    inode::Node{SIS,TG},
    iindex::Int,
    jnode::Node{SIS,TG},
    jindex::Int,
    sumargexp::SumM,
    infectionmodel::SIS) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    M[1, 1, :] .= (exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1])).*(1 .- infectionmodel.εᵢᵗ[inode.i, :]) .* inode.obs[1, 1:end-1]
    M[1, 2, :] .= (1 .- exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1]).*(1 .- infectionmodel.εᵢᵗ[inode.i, :])) .* inode.obs[1, 1:end-1]
    M[2, 1, :] .= infectionmodel.rᵢᵗ[inode.i, :] .* exp.(sumargexp.sumμ .- inode.cavities[jindex].μ .* jnode.νs[iindex][1:end-1]) .* inode.obs[2, 1:end-1]
    M[2, 2, :] .= (1 .- infectionmodel.rᵢᵗ[inode.i, :]) .* exp.(sumargexp.sumμ .- inode.cavities[jindex].μ .* jnode.νs[iindex][1:end-1]) .* inode.obs[2, 1:end-1]
end

function fill_transmat_marg!(
    M::Array{Float64,3},
    inode::Node{SIS,TG},
    sumargexp::SumM,
    infectionmodel::SIS) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    M[1, 1, :] .= (exp.(sumargexp.summ[1:end-1])).*(1 .- infectionmodel.εᵢᵗ[inode.i, :]) .* inode.obs[1, 1:end-1]
    M[1, 2, :] .= (1 .- exp.(sumargexp.summ[1:end-1]).*(1 .- infectionmodel.εᵢᵗ[inode.i, :])) .* inode.obs[1, 1:end-1]
    M[2, 1, :] .= infectionmodel.rᵢᵗ[inode.i, :] .* exp.(sumargexp.sumμ) .* inode.obs[2, 1:end-1]
    M[2, 2, :] .= (1 .- infectionmodel.rᵢᵗ[inode.i, :]) .* exp.(sumargexp.sumμ) .* inode.obs[2, 1:end-1]
end

"""
    sim_epidemics(
        model::EpidemicModel{SIS,TG};
        patient_zero::Union{Vector{Int},Nothing}=nothing,
        γ::Union{Float64,Nothing}=nothing) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

Simulates the epidemic outbreak given a SIS model. 

# Arguments
* `model`: The SIS epidemic model.
* `patient_zero`: Vector of patients zero. Default is "nothing", meaning that the patients zero are chosen at random with probability γ.
* `γ`: Probability of being a patient zero. Default is "nothing", meaning that it is fixed to0 1/NV, where NV is the number of nodes of the contact graph.
"""
function sim_epidemics(
    model::EpidemicModel{SIS,TG};
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

    hs = zeros(model.N)
    for t in 1:model.T
        hs .= (config[:, t]' * model.ν[:, :, t])'
        config[:, t+1] .= [x * rand(Bernoulli(1 - r)) + (1 - x) * rand(Bernoulli(1 - exp(h))) for (x, h, r) in zip(config[:, t], hs, model.Disease.rᵢᵗ[:, t])]
    end
    return config
end
