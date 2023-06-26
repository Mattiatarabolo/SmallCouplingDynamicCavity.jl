struct SIS <: InfectionModel
    εᵢᵗ::Array{Float64,2} # autoinfection
    rᵢᵗ::Array{Float64,2} # I->R
end
n_states(X::SIS) = 2

# formatting
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
    model::EpidemicModel{SIS}, 
    obsprob::Function)

    nodes = Vector{Node{SIS}}()

    for i in 1:nv(model.G)
        obs = ones(2, model.T + 1)
        obs[1, :] = [obsprob(Ob, 0.0) for Ob in model.obsmat[i, :]]
        obs[2, :] = [obsprob(Ob, 1.0) for Ob in model.obsmat[i, :]]

        ∂ = neighbors(model.G, i)

        ν∂ = [model.ν[k, i, :] for k in ∂]

        push!(nodes, Node(i, ∂, model.T, ν∂, obs, model.Disease))
        
    end
    return collect(nodes)
end

# function to fill the transfer matrix
function fill_transmat_cav!(
    M::Array{Float64,3},
    inode::Node{SIS},
    iindex::Int,
    jnode::Node{SIS},
    jindex::Int,
    sumargexp::SumM,
    infectionmodel::SIS)
    
    M[1, 1, :] .= (exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1])).*(1 .- infectionmodel.εᵢᵗ[inode.i, :]) .* inode.obs[1, 1:end-1]
    M[1, 2, :] .= (1 .- exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1]).*(1 .- infectionmodel.εᵢᵗ[inode.i, :])) .* inode.obs[1, 1:end-1]
    M[2, 1, :] .= infectionmodel.rᵢᵗ[inode.i, :] .* exp.(sumargexp.sumμ .- inode.cavities[jindex].μ .* jnode.νs[iindex][1:end-1]) .* inode.obs[2, 1:end-1]
    M[2, 2, :] .= (1 .- infectionmodel.rᵢᵗ[inode.i, :]) .* exp.(sumargexp.sumμ .- inode.cavities[jindex].μ .* jnode.νs[iindex][1:end-1]) .* inode.obs[2, 1:end-1]
end

function fill_transmat_marg!(
    M::Array{Float64,3},
    inode::Node{SIS},
    sumargexp::SumM,
    infectionmodel::SIS)
    
    M[1, 1, :] .= (exp.(sumargexp.summ[1:end-1])).*(1 .- infectionmodel.εᵢᵗ[inode.i, :]) .* inode.obs[1, 1:end-1]
    M[1, 2, :] .= (1 .- exp.(sumargexp.summ[1:end-1]).*(1 .- infectionmodel.εᵢᵗ[inode.i, :])) .* inode.obs[1, 1:end-1]
    M[2, 1, :] .= infectionmodel.rᵢᵗ[inode.i, :] .* exp.(sumargexp.sumμ) .* inode.obs[2, 1:end-1]
    M[2, 2, :] .= (1 .- infectionmodel.rᵢᵗ[inode.i, :]) .* exp.(sumargexp.sumμ) .* inode.obs[2, 1:end-1]
end

# sample
function sim_epidemics(
    model::EpidemicModel{SIS};
    patient_zero::Union{Vector{Int},Nothing}=nothing,
    γ::Union{Float64,Nothing}=nothing)

    inf₀ = false
    if patient_zero === nothing && γ !== nothing
        while !inf₀
            patient_zero = rand(Binomial(1,γ), nv(model.G))
            patient_zero = findall(x->x==1, patient_zero)
            inf₀ = !isempty(patient_zero)
        end
    elseif patient_zero === nothing && γ !== nothing
        while !inf₀
            patient_zero = rand(Binomial(1,1/nv(model.G)), nv(model.G))
            patient_zero = findall(x->x==1, patient_zero)
            inf₀ = !isempty(patient_zero)
        end
    end

    config = zeros(nv(model.G), model.T + 1)

    config[patient_zero,1] .+= 1.0

    hs = zeros(nv(model.G))
    for t in 1:model.T
        hs .= (config[:, t]' * model.ν[:, :, t])'
        config[:, t+1] .= [x * rand(Bernoulli(1 - r)) + (1 - x) * rand(Bernoulli(1 - exp(h))) for (x, h, r) in zip(config[:, t], hs, model.Disease.rᵢᵗ[:, t])]
    end
    return config
end
