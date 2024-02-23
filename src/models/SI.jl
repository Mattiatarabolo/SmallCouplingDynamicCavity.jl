struct SI <: InfectionModel
    εᵢᵗ::Array{Float64,2} # autoinfection
end
n_states(X::SI) = 2

# formatting
function SI(
    εᵢᵗ::Union{Float64,Array{Float64,2}},
    NV::Int,
    T::Int)
    if typeof(εᵢᵗ) == Float64
        εᵢᵗ = ones(NV, T) .* εᵢᵗ
    end

    return SI(εᵢᵗ)
end

function nodes_formatting(
    model::EpidemicModel{SI,AbstractGraph}, 
    obsprob::Function)

    nodes = Vector{Node{SI,AbstractGraph}}()

    for i in 1:nv(model.G)
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
    model::EpidemicModel{SI,Vector{<:AbstractGraph}}, 
    obsprob::Function)

    nodes = Vector{Node{SI,Vector{<:AbstractGraph}}}()

    for i in 1:nv(model.G)
        obs = ones(2, model.T + 1)
        obs[1, :] = [obsprob(Ob, 0.0) for Ob in model.obsmat[i, :]]
        obs[2, :] = [obsprob(Ob, 1.0) for Ob in model.obsmat[i, :]]

        ∂ = Vector{Int}()

        for t in 1:model.T+1
            ∂ = union(neighbors(model.G[t], i))
        end
             
        ν∂ = [model.ν[k, i, :] for k in ∂]

        push!(nodes, Node(i, ∂, model.T, ν∂, obs, model))
        
    end
    return collect(nodes)
end
 
# function to fill the transfer matrix
function fill_transmat_cav!(
    M::Array{Float64,3},
    inode::Node{SI,TG},
    iindex::Int,
    jnode::Node{SI,TG},
    jindex::Int,
    sumargexp::SumM,
    infectionmodel::SI) where {TG<:Union{AbstractGraph,Vector{<:AbstractGraph}}}
    
    M[1, 1, :] .= (exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1])).*(1 .- infectionmodel.εᵢᵗ[inode.i, :]) .* inode.obs[1, 1:end-1]
    M[1, 2, :] .= (1 .- exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1]).*(1 .- infectionmodel.εᵢᵗ[inode.i, :])) .* inode.obs[1, 1:end-1]
    M[2, 2, :] .= exp.(sumargexp.sumμ .- inode.cavities[jindex].μ .* jnode.νs[iindex][1:end-1]) .* inode.obs[2, 1:end-1]
end

function fill_transmat_marg!(
    M::Array{Float64,3},
    inode::Node{SI,TG},
    sumargexp::SumM,
    infectionmodel::SI) where {TG<:Union{AbstractGraph,Vector{<:AbstractGraph}}}
    
    M[1, 1, :] .= (exp.(sumargexp.summ[1:end-1])).*(1 .- infectionmodel.εᵢᵗ[inode.i, :]) .* inode.obs[1, 1:end-1]
    M[1, 2, :] .= (1 .- exp.(sumargexp.summ[1:end-1]).*(1 .- infectionmodel.εᵢᵗ[inode.i, :])) .* inode.obs[1, 1:end-1]
    M[2, 2, :] .= exp.(sumargexp.sumμ) .* inode.obs[2, 1:end-1]
end

# sample
function sim_epidemics(
    model::EpidemicModel{SI,TG};
    patient_zero::Union{Vector{Int},Nothing}=nothing,
    γ::Union{Float64,Nothing}=nothing) where {TG<:Union{AbstractGraph,Vector{<:AbstractGraph}}}

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

    config[patient_zero, 1] .+= 1.0

    hs = zeros(nv(model.G))
    for t in 1:model.T
        hs = config[:, t]' * model.ν[:, :, t]
        config[:, t+1] = [x + (1 - x) * rand(Bernoulli(1 - exp(h))) for (x, h) in zip(config[:, t], hs)]
    end
    return config
end
