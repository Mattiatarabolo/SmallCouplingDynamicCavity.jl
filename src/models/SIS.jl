struct SIS <: InfectionModel
    εᵢᵗ::Union{Float64,Array{Float64,2}} # autoinfection
    σᵢᵗ::Union{Float64,Array{Float64,2}} # R->S
end
n_states(X::SIS) = 2

# formatting
function SIS(
    εᵢᵗ::Union{Float64,Array{Float64,2}},
    σᵢᵗ::Union{Float64,Array{Float64,2}},
    T::Int)
    if length(εᵢᵗ) == T+1
        εᵢᵗ = εᵢᵗ[1:end-1]
    elseif length(σᵢᵗ) == T+1 
        σᵢᵗ = σᵢᵗ[1:end-1]
    end

    return SIRS(εᵢᵗ, rᵢᵗ, σᵢᵗ)
end

function nodes_formatting(
    model::EpidemicModel{SIS}, 
    obsprob::Function, 
    ν::Array{Float64,3})

    nodes = Vector{Node{SIS}}()

    for i in 1:nv(model.G)
        obs = ones(2, model.T + 1)
        obs[1, :] = [obsprob(Ob, 0.0) for Ob in model.obsmat[i, :]]
        obs[2, :] = [obsprob(Ob, 1.0) for Ob in model.obsmat[i, :]]

        ∂ = neighbors(model.G, i)

        ν∂ = [ν[k, i, :] for k in ∂]

        push!(nodes, Node(i, ∂, model.T, ν∂, obs, model.Disease))
        
    end
    return collect(nodes)
end

# function to fill the transfer matrix
function fill_transmat_cav!(
    M::Array{Float64,3},
    inode::Node,
    iindex::Int,
    jnode::Node,
    jindex::Int,
    sumargexp::SumM,
    infectionmodel::SI)
    
    M[1, 1, :] .= (exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1])).*(1 .- infectionmodel.εᵢᵗ) .* inode.obs[1, 1:end-1]
    M[1, 2, :] .= (1 .- exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1]).*(1 .- infectionmodel.εᵢᵗ)) .* inode.obs[1, 1:end-1]
    M[2, 1, :] .= infectionmodel.rᵢᵗ .* exp.(sumargexp.sumμ .- inode.cavities[jindex].μ .* jnode.νs[iindex][1:end-1]) .* inode.obs[2, 1:end-1]
    M[2, 2, :] .= (1 .- infectionmodel.rᵢᵗ) .* exp.(sumargexp.sumμ .- inode.cavities[jindex].μ .* jnode.νs[iindex][1:end-1]) .* inode.obs[2, 1:end-1]
end

function fill_transmat_marg!(
    M::Array{Float64,3},
    inode::Node,
    sumargexp::SumM,
    infectionmodel::SI)
    
    M[1, 1, :] .= (exp.(sumargexp.summ[1:end-1])).*(1 .- infectionmodel.εᵢᵗ) .* inode.obs[1, 1:end-1]
    M[1, 2, :] .= (1 .- exp.(sumargexp.summ[1:end-1]).*(1 .- infectionmodel.εᵢᵗ)) .* inode.obs[1, 1:end-1]
    M[2, 1, :] .= infectionmodel.rᵢᵗ .* exp.(sumargexp.sumμ) .* inode.obs[2, 1:end-1]
    M[2, 2, :] .= (1 .- infectionmodel.rᵢᵗ) .* exp.(sumargexp.sumμ) .* inode.obs[2, 1:end-1]
end

# sample
function sim_epidemics(
    model::EpidemicModel{SIS};
    patient_zero::Union{Vector{Int},Nothing}=nothing,
    γ::Union{Float64,Nothing}=nothing)

    inf₀ = false
    if patient_zero === nothing && γ != nothing
        while !inf₀
            patient_zero = rand(Binomial(1,γ), nv(model.G))
            patient_zero = findall(x->x==1, patient_zero)
            inf₀ = !isempty(patient_zero)
        end
    elseif patient_zero === nothing && γ != nothing
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
        hs = config[:, t]' * model.ν[:, :, t]
        config[:, t+1] = [x * rand(Bernoulli(1 - r)) + (1 - x) * rand(Bernoulli(1 - exp(h))) for (x, h, r) in zip(config[:, t], hs, model.Disease.σᵢᵗ[:, t])]
    end
    return config
end
