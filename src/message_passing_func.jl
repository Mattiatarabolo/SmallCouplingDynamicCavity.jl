function update_single_message!(
    jnode::Node,
    iindex::Int,
    ρ::FBm,
    M::Array{Float64,3},
    updmess::Updmess,
    newmess::Message,
    damp::Float64,
    sumargexp::SumM,
    inode::Node,
    μ_cutoff::Float64)

    clear!(updmess, newmess)

    updmess.lognumm .= log.(ρ.fwm) .+ log.(ρ.bwm)
    updmess.signμ .= sign.(ρ.bwm[1, 2:end] - ρ.bwm[2, 2:end])
    updmess.lognumμ .= log.(ρ.fwm[1, 1:end-1]) .+ log.(M[1, 1, :]) .+ log.(abs.(ρ.bwm[1, 2:end] - ρ.bwm[2, 2:end]))
    updmess.logZ .= log.(dropdims(sum(ρ.fwm .* ρ.bwm, dims=1), dims=1))

    newmess.m .= exp.(updmess.lognumm[2, :] .- updmess.logZ)
    newmess.μ .= max.(updmess.signμ .* exp.(updmess.lognumμ .- updmess.logZ[1:end-1]), μ_cutoff)

    #=  #DEBUGGING    
    updmess.lognumm = ρ.fwm.*ρ.bwm
    updmess.signμ = sign.(ρ.bwm[1,2:end]-ρ.bwm[2,2:end])
    updmess.lognumμ = ρ.fwm[1,1:end-1].*M[1,1,:].*(ρ.bwm[1,2:end]-ρ.bwm[2,2:end])
    updmess.logZ = dropdims(sum(ρ.fwm .* ρ.bwm, dims = 1), dims = 1)

    newmess.m = updmess.lognumm[2,:]./updmess.logZ
    newmess.μ = updmess.lognumμ./updmess.logZ[1:end-1] 
    =#

    newmess.m .= jnode.cavities[iindex].m.*damp .+ newmess.m.*(1 - damp)
    newmess.μ .= jnode.cavities[iindex].μ.*damp .+ newmess.μ.*(1 - damp)

    ε = max(normupdate(jnode.cavities[iindex].m, newmess.m), normupdate(jnode.cavities[iindex].μ, newmess.μ))

    if isnan(ε)
        #DEBUGGING       
        println("cavity updated i = $(inode.i)-> j = $(jnode.i)")

        for (kindex, k) in enumerate(inode.∂)
            println("\nk = $(k) ∈ ∂i\\ j")
            println("mₖᵢᵗ = $(inode.cavities[kindex].m)")
            println("μₖᵢᵗ = $(inode.cavities[kindex].μ)")
        end
        println("\nsummᵢ = $(sumargexp.summ)")
        println("sumμᵢ = $(sumargexp.sumμ)")
        println("\nnewmᵢⱼ = $(newmess.m)")
        println("newμᵢⱼ = $(newmess.μ)")
        println("logZᵢⱼm = $(updmess.logZ)")
        println("ρ₊ₜⁱʲ = $(ρ.fwm)")
        println("ρₜ₋ⁱʲ = $(ρ.bwm)")
        print("Mⁱʲ = ")
        display(M)
        for t in 1:T
            print("ρ₊$(t-1)ⁱʲ")
            display(ρ.fwm[:, t]')
            display(M[:, :, t])
        end
        for t in 1:T
            print("ρ₋$(T+2-t-1)ⁱʲ")
            display(ρ.bwm[:, T+2-t])
            display(M[:, :, T+1-t])
        end

        throw(DomainError("NaN evaluated"))
    end

    jnode.cavities[iindex].m .= newmess.m
    jnode.cavities[iindex].μ .= newmess.μ 

    return ε
end

function compute_ρ!(
    inode::Node,
    iindex::Int,
    jnode::Node,
    jindex::Int,
    sumargexp::SumM,
    M::Array{Float64,3},
    ρ::FBm,
    prior::Array{Float64,2},
    T::Int,
    ε_autoinf::Float64,
    model::Symbol)

    clear!(M, ρ)

    ρ.fwm[:, 1] .= ρ_norm(prior[:, inode.i])
    ρ.bwm[:, T+1] .= ρ_norm(inode.obs[:, T+1])

    if model == :SI
        # problema numerico di exp e 1-exp
        M[1, 1, :] .= (exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1])).*(1 - ε_autoinf) .* inode.obs[1, 1:end-1]
        M[1, 2, :] .= (ones(T).- exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1]).*(1 - ε_autoinf)) .* inode.obs[1, 1:end-1]
        M[2, 2, :] .= exp.(sumargexp.sumμ .- inode.cavities[jindex].μ .* jnode.νs[iindex][1:end-1]) .* inode.obs[2, 1:end-1]
    elseif model == :SIS
        M[1, 1, :] .= (exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1])).*(1 - ε_autoinf) .* inode.obs[1, 1:end-1]
        M[1, 2, :] .= (ones(T).- exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1]).*(1 - ε_autoinf)) .* inode.obs[1, 1:end-1]
        M[2, 1, :] .= inode.rᵢᵗ[1:end-1] .* exp.(sumargexp.sumμ .- inode.cavities[jindex].μ .* jnode.νs[iindex][1:end-1]) .* inode.obs[2, 1:end-1]
        M[2, 2, :] .= (ones(T) .- inode.rᵢᵗ[1:end-1]) .* exp.(sumargexp.sumμ .- inode.cavities[jindex].μ .* jnode.νs[iindex][1:end-1]) .* inode.obs[2, 1:end-1]
    elseif model == :SIR
        M[1, 1, :] .= (exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1])).*(1 - ε_autoinf) .* inode.obs[1, 1:end-1]
        M[1, 2, :] .= (ones(T).- exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1]).*(1 - ε_autoinf)) .* inode.obs[1, 1:end-1]
        M[2, 2, :] .= (ones(T) .- inode.rᵢᵗ[1:end-1]) .* exp.(sumargexp.sumμ .- inode.cavities[jindex].μ .* jnode.νs[iindex][1:end-1]) .* inode.obs[2, 1:end-1]
        M[2, 3, :] .= inode.rᵢᵗ[1:end-1] .* exp.(sumargexp.sumμ .- inode.cavities[jindex].μ .* jnode.νs[iindex][1:end-1]) .* inode.obs[2, 1:end-1]
        M[3, 3, :] .= inode.obs[3, 1:end-1]
    elseif model == :SIRS
        M[1, 1, :] .= (exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1])).*(1 - ε_autoinf) .* inode.obs[1, 1:end-1]
        M[1, 2, :] .= (ones(T).- exp.(sumargexp.summ[1:end-1] .- inode.cavities[jindex].m[1:end-1] .* inode.νs[jindex][1:end-1]).*(1 - ε_autoinf)) .* inode.obs[1, 1:end-1]
        M[2, 2, :] .= (ones(T) .- inode.rᵢᵗ[1:end-1]) .* exp.(sumargexp.sumμ .- inode.cavities[jindex].μ .* jnode.νs[iindex][1:end-1]) .* inode.obs[2, 1:end-1]
        M[2, 3, :] .= inode.rᵢᵗ[1:end-1] .* exp.(sumargexp.sumμ .- inode.cavities[jindex].μ .* jnode.νs[iindex][1:end-1]) .* inode.obs[2, 1:end-1]
        M[3, 1, :] .= inode.σᵢᵗ[1:end-1] .* inode.obs[3, 1:end-1]
        M[3, 3, :] .= (ones(T) .- inode.σᵢᵗ[1:end-1]) .* inode.obs[3, 1:end-1]
    else
        throw(ArgumentError("The only available models are SI, SIS, and SIR"))
    end

    # fwd-bwd update
    for t in 1:T
        ρ.fwm[:, t+1] .= ρ_norm(collect((ρ.fwm[:, t]' * M[:, :, t])'))
        ρ.bwm[:, T+1-t] .= ρ_norm(M[:, :, T+1-t] * ρ.bwm[:, T+2-t])
    end

    return M, ρ
end

function update_single_marginal!(
    inode::Node, 
    nodes::Vector{Node}, 
    sumargexp::SumM, 
    M::Array{Float64, 3}, 
    ρ::FBm, 
    prior::Array{Float64, 2}, 
    T::Int,
    updmess::Updmess,
    newmarg::Marginal,
    μ_cutoff::Float64,
    ε_autoinf::Float64,
    model::Symbol)
    
    compute_sumargexp!(inode, nodes, sumargexp)

    clear!(M, ρ)

    ρ.fwm[:, 1] .= ρ_norm(prior[:, inode.i])
    ρ.bwm[:, T+1] .= ρ_norm(inode.obs[:, T+1])

    if model == :SI
        # problema numerico di exp e 1-exp
        M[1, 1, :] .= exp.(sumargexp.summ[1:end-1]) .* (1- ε_autoinf).* inode.obs[1, 1:end-1]
        M[1, 2, :] .= (ones(T).- exp.(sumargexp.summ[1:end-1]).* (1- ε_autoinf)) .* inode.obs[1, 1:end-1]
        M[2, 2, :] .= exp.(sumargexp.sumμ) .* inode.obs[2, 1:end-1]
    elseif model == :SIS
        M[1, 1, :] .= exp.(sumargexp.summ[1:end-1]) .* (1- ε_autoinf).* inode.obs[1, 1:end-1]
        M[1, 2, :] .= (ones(T).- exp.(sumargexp.summ[1:end-1]).* (1- ε_autoinf)) .* inode.obs[1, 1:end-1]
        M[2, 1, :] .= inode.rᵢᵗ[1:end-1] .* exp.(sumargexp.sumμ) .* inode.obs[2, 1:end-1]
        M[2, 2, :] .= (ones(T) .- inode.rᵢᵗ[1:end-1]) .* exp.(sumargexp.sumμ) .* inode.obs[2, 1:end-1]
    elseif model == :SIR
        M[1, 1, :] .= exp.(sumargexp.summ[1:end-1]) .* (1- ε_autoinf).* inode.obs[1, 1:end-1]
        M[1, 2, :] .= (ones(T).- exp.(sumargexp.summ[1:end-1]).* (1- ε_autoinf)) .* inode.obs[1, 1:end-1]
        M[2, 2, :] .= (ones(T) .- inode.rᵢᵗ[1:end-1]) .* exp.(sumargexp.sumμ) .* inode.obs[2, 1:end-1]
        M[2, 3, :] .= inode.rᵢᵗ[1:end-1] .* exp.(sumargexp.sumμ) .* inode.obs[2, 1:end-1]
        M[3, 3, :] .= inode.obs[3, 1:end-1]
    elseif model == :SIRS
        M[1, 1, :] .= exp.(sumargexp.summ[1:end-1]) .* (1- ε_autoinf).* inode.obs[1, 1:end-1]
        M[1, 2, :] .= (ones(T).- exp.(sumargexp.summ[1:end-1]).* (1- ε_autoinf)) .* inode.obs[1, 1:end-1]
        M[2, 2, :] .= (ones(T) .- inode.rᵢᵗ[1:end-1]) .* exp.(sumargexp.sumμ) .* inode.obs[2, 1:end-1]
        M[2, 3, :] .= inode.rᵢᵗ[1:end-1] .* exp.(sumargexp.sumμ) .* inode.obs[2, 1:end-1]
        M[3, 1, :] .= inode.σᵢᵗ[1:end-1] .* inode.obs[3, 1:end-1]
        M[3, 3, :] .= (ones(T) .- inode.σᵢᵗ[1:end-1]) .* inode.obs[3, 1:end-1]
    else
        throw(ArgumentError("The only available models are SI, SIS, and SIR"))
    end

    # fwd-bwd update
    for t in 1:T
        ρ.fwm[:, t+1] .= ρ_norm(collect((ρ.fwm[:, t]' * M[:, :, t])'))
        ρ.bwm[:, T+1-t] .= ρ_norm(M[:, :, T+1-t] * ρ.bwm[:, T+2-t])
    end

    clear!(updmess, newmarg)

    updmess.lognumm .= log.(ρ.fwm) .+ log.(ρ.bwm)
    updmess.signμ .= sign.(ρ.bwm[1, 2:end] - ρ.bwm[2, 2:end])
    updmess.lognumμ .= log.(ρ.fwm[1, 1:end-1]) .+ log.(M[1, 1, :]) .+ log.(abs.(ρ.bwm[1, 2:end] - ρ.bwm[2, 2:end]))
    updmess.logZ .= log.(dropdims(sum(ρ.fwm .* ρ.bwm, dims=1), dims=1))

    newmarg.m .= exp.(updmess.lognumm .- updmess.logZ')
    newmarg.μ .= max.(updmess.signμ .* exp.(updmess.lognumμ .- updmess.logZ[1:end-1]), μ_cutoff)

    return newmarg
end

function compute_sumargexp!(
    inode::Node,
    nodes::Vector{Node},
    sumargexp::SumM)

    clear!(sumargexp)

    for (kindex, k) in enumerate(inode.∂)
        iindex = nodes[k].∂_idx[inode.i]
        sumargexp.summ .+= inode.cavities[kindex].m .* inode.νs[kindex]  #chiedere ad anna se è più veloce riga o colonna
        sumargexp.sumμ .+= inode.cavities[kindex].μ .* nodes[k].νs[iindex][1:end-1]
    end

    return sumargexp
end

function update_node!(
    inode::Node, 
    nodes::Vector{Node}, 
    sumargexp::SumM, 
    M::Array{Float64, 3}, 
    ρ::FBm, 
    prior::Array{Float64, 2}, 
    T::Int,
    updmess::Updmess,
    newmess::Message,
    newmarg::Marginal,
    damp::Float64,
    μ_cutoff::Float64,
    ε_autoinf::Float64,
    model::Symbol)

    ε = 0.0

    sumargexp = compute_sumargexp!(inode, nodes, sumargexp)

    for (jindex, j) in enumerate(inode.∂)
        iindex = nodes[j].∂_idx[inode.i]
        M, ρ = compute_ρ!(inode, iindex, nodes[j], jindex, sumargexp, M, ρ, prior, T, ε_autoinf, model)
        ε = max(ε, update_single_message!(nodes[j], iindex, ρ, M, updmess, newmess, damp, sumargexp, inode, μ_cutoff))
    end

    return ε
end

function update_cavities!(
    nodes::Vector{Node},
    sumargexp::SumM,
    M::Array{Float64,3},
    ρ::FBm,
    prior::Array{Float64,2},
    T::Int,
    updmess::Updmess,
    newmess::Message,
    newmarg::Marginal,
    damp::Float64,
    μ_cutoff::Float64,
    ε_autoinf::Float64,
    model::Symbol)

    ε = 0.0

    for inode in nodes
        ε = max(ε, update_node!(inode, nodes, sumargexp, M, ρ, prior, T, updmess, newmess, newmarg, damp, μ_cutoff, ε_autoinf, model))
    end

    return ε
end

function compute_marginals!(
    nodes::Vector{Node},
    sumargexp::SumM,
    M::Array{Float64,3},
    ρ::FBm, 
    T::Int64,
    prior::Array{Float64,2},
    updmess::Updmess,
    newmarg::Marginal,
    μ_cutoff::Float64,
    ε_autoinf::Float64,
    model::Symbol)

    if model == :SI || model == :SIS
        nr_states = 2
    elseif model == :SIR || model == :SIRS
        nr_states = 3
    end

    for inode in nodes
        newmarg = update_single_marginal!(inode, nodes, sumargexp, M, ρ, prior, T, updmess, newmarg, μ_cutoff, ε_autoinf, model)
        inode.marg.m .= newmarg.m
    end

    
end


function run_SCDC(
    G::SimpleGraph{Int64},
    λ::Array{Float64,3},
    obsmat::Array{Float64,2},
    obsprob::Function,
    γ::Float64,
    T::Int64,
    maxiter::Int64,
    epsconv::Float64,
    damp::Float64,
    model::Symbol;
    r::Union{Array{Float64,2},Nothing}=nothing,
    σ::Union{Array{Float64,2},Nothing}=nothing,
    μ_cutoff::Float64 = -Inf,
    ε_autoinf::Float64 = 0.0,
    callback::Function=(x...) -> nothing)

    if model == :SI || model == :SIS
        nr_states = 2
    elseif model == :SIR || model == :SIRS
        nr_states = 3
    else
        throw(ArgumentError("The only available models are SI, SIS, and SIR"))
    end

    # set prior (given an expected mean number of source patients γ)
    prior = zeros(nr_states, nv(G))
    prior[1, :] .= (1 - γ) # x_i = S
    prior[2, :] .= γ # x_i = I

    # compute ν Matrix
    ν = log.(1 .- λ)

    if model == :SI
            nodes = nodes_formatting(G, obsmat, obsprob, ν, T, nr_states)
    elseif model == :SIS || model == :SIR
        if r === nothing
            throw(ArgumentError("r must be non empty for SIS and SIR"))
        end
            nodes = nodes_formatting(G, obsmat, obsprob, ν, T, nr_states, r=r)

    elseif model == :SIRS
        if σ === nothing
            throw(ArgumentError("σ must be non empty for SIS and SIR"))
        end
            nodes = nodes_formatting(G, obsmat, obsprob, ν, T, nr_states, r=r, σ=σ)
    else
        throw(ArgumentError("The only available models are SI, SIS, SIR and SIRS"))
    end

    Mⁱʲ = TransMat(T, nr_states)
    ρⁱʲ = FBm(T, nr_states)
    sumargexp = SumM(T)
    updmess = Updmess(T, nr_states)
    newmess = Message(0, 0, T)
    newmarg = Marginal(0, T, nr_states)

    ε = 0.0

    for iter = 1:maxiter
        #update cavity messages and compute their convergence
        ε = update_cavities!(nodes, sumargexp, Mⁱʲ, ρⁱʲ, prior, T, updmess, newmess, newmarg, damp, μ_cutoff, ε_autoinf, model)

        callback(nodes, iter, ε)

        #check convergence
        if ε < epsconv
            println("Converged after $iter iterations")
            break
        end
    end

    for inode in nodes
        sumargexp = compute_sumargexp!(inode, nodes, sumargexp)
        for (jindex, j) in enumerate(inode.∂)
            iindex = nodes[j].∂_idx[inode.i]
            _, ρ = compute_ρ!(inode, iindex, nodes[j], jindex, sumargexp, Mⁱʲ, ρⁱʲ, prior, T, ε_autoinf, model)
            nodes[j].ρs[iindex].fwm .= ρ.fwm
            nodes[j].ρs[iindex].bwm .= ρ.bwm
        end
    end

    # check unconvergence
    if ε > epsconv
        println("NOT converged after $maxiter iterations")
    end
    

    compute_marginals!(nodes, sumargexp, Mⁱʲ, ρⁱʲ, T, prior, updmess, newmarg, μ_cutoff, ε_autoinf, model)

    return nodes
end