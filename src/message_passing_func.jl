function update_single_message!(
    jnode::Node{TI,TG},
    iindex::Int,
    ρ::FBm,
    M::Array{Float64,3},
    updmess::Updmess,
    newmess::Message,
    damp::Float64,
    sumargexp::SumM,
    inode::Node{TI,TG},
    μ_cutoff::Float64) where {TI<:InfectionModel,TG<:Union{AbstractGraph,Vector{<:AbstractGraph}}}
    
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
        throw(DomainError("NaN evaluated"))
    end   
    #= 
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

 =#    
    jnode.cavities[iindex].m .= newmess.m
    jnode.cavities[iindex].μ .= newmess.μ 

    return ε
end

function compute_ρ!(
    inode::Node{TI,TG},
    iindex::Int,
    jnode::Node{TI,TG},
    jindex::Int,
    sumargexp::SumM,
    M::Array{Float64,3},
    ρ::FBm,
    prior::Array{Float64,2},
    T::Int,
    infectionmodel::TI) where {TI<:InfectionModel,TG<:Union{AbstractGraph,Vector{<:AbstractGraph}}}

    clear!(M, ρ)

    ρ.fwm[:, 1] .= ρ_norm(prior[:, inode.i])
    ρ.bwm[:, T+1] .= ρ_norm(inode.obs[:, T+1])

    fill_transmat_cav!(M, inode, iindex, jnode, jindex, sumargexp, infectionmodel)

    # fwd-bwd update
    for t in 1:T
        ρ.fwm[:, t+1] .= ρ_norm(collect((ρ.fwm[:, t]' * M[:, :, t])'))
        ρ.bwm[:, T+1-t] .= ρ_norm(M[:, :, T+1-t] * ρ.bwm[:, T+2-t])
    end

    return M, ρ
end

function update_single_marginal!(
    inode::Node{TI,TG}, 
    nodes::Vector{Node{TI,TG}}, 
    sumargexp::SumM, 
    M::Array{Float64, 3}, 
    ρ::FBm, 
    prior::Array{Float64, 2}, 
    T::Int,
    updmess::Updmess,
    newmarg::Marginal,
    μ_cutoff::Float64,
    infectionmodel::TI) where {TI<:InfectionModel,TG<:Union{AbstractGraph,Vector{<:AbstractGraph}}}
    
    compute_sumargexp!(inode, nodes, sumargexp)

    clear!(M, ρ)

    ρ.fwm[:, 1] .= ρ_norm(prior[:, inode.i])
    ρ.bwm[:, T+1] .= ρ_norm(inode.obs[:, T+1])

    fill_transmat_marg!(M, inode, sumargexp, infectionmodel)

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
    inode::Node{TI,TG},
    nodes::Vector{Node{TI,TG}},
    sumargexp::SumM) where {TI<:InfectionModel,TG<:Union{AbstractGraph,Vector{<:AbstractGraph}}}

    clear!(sumargexp)

    for (kindex, k) in enumerate(inode.∂)
        iindex = nodes[k].∂_idx[inode.i]
        sumargexp.summ .+= inode.cavities[kindex].m .* inode.νs[kindex]  #chiedere ad anna se è più veloce riga o colonna
        sumargexp.sumμ .+= inode.cavities[kindex].μ .* nodes[k].νs[iindex][1:end-1]
    end

    return sumargexp
end

function update_node!(
    inode::Node{TI,TG}, 
    nodes::Vector{Node{TI,TG}}, 
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
    infectionmodel::TI) where {TI<:InfectionModel,TG<:Union{AbstractGraph,Vector{<:AbstractGraph}}}

    ε = 0.0

    sumargexp = compute_sumargexp!(inode, nodes, sumargexp)

    for (jindex, j) in enumerate(inode.∂)
        iindex = nodes[j].∂_idx[inode.i]
        M, ρ = compute_ρ!(inode, iindex, nodes[j], jindex, sumargexp, M, ρ, prior, T, infectionmodel)
        ε = max(ε, update_single_message!(nodes[j], iindex, ρ, M, updmess, newmess, damp, sumargexp, inode, μ_cutoff))
    end

    return ε
end

function update_cavities!(
    nodes::Vector{Node{TI,TG}},
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
    infectionmodel::TI) where {TI<:InfectionModel,TG<:Union{AbstractGraph,Vector{<:AbstractGraph}}}

    ε = 0.0

    for inode in nodes
        ε = max(ε, update_node!(inode, nodes, sumargexp, M, ρ, prior, T, updmess, newmess, newmarg, damp, μ_cutoff, infectionmodel))
    end

    return ε
end

function compute_marginals!(
    nodes::Vector{Node{TI,TG}},
    sumargexp::SumM,
    M::Array{Float64,3},
    ρ::FBm, 
    T::Int64,
    prior::Array{Float64,2},
    updmess::Updmess,
    newmarg::Marginal,
    μ_cutoff::Float64,
    infectionmodel::TI) where {TI<:InfectionModel,TG<:Union{AbstractGraph,Vector{<:AbstractGraph}}}

    for inode in nodes
        newmarg = update_single_marginal!(inode, nodes, sumargexp, M, ρ, prior, T, updmess, newmarg, μ_cutoff, infectionmodel)
        inode.marg.m .= newmarg.m
    end

    
end


function run_SCDC(
    model::EpidemicModel{TI,TG},
    obsprob::Function,
    γ::Float64,
    maxiter::Int64,
    epsconv::Float64,
    damp::Float64;
    μ_cutoff::Float64 = -Inf,
    callback::Function=(x...) -> nothing) where {TI<:InfectionModel,TG<:Union{AbstractGraph,Vector{<:AbstractGraph}}}

    # set prior (given an expected mean number of source patients γ)
    prior = zeros(n_states(model.Disease), nv(model.G))
    prior[1, :] .= (1 - γ) # x_i = S
    prior[2, :] .= γ # x_i = I

    nodes = nodes_formatting(model, obsprob)

    Mⁱʲ = TransMat(model.T, model.Disease)
    ρⁱʲ = FBm(model.T, model.Disease)
    sumargexp = SumM(model.T)
    updmess = Updmess(model.T, model.Disease)
    newmess = Message(0, 0, model.T)
    newmarg = Marginal(0, model.T, model.Disease)

    ε = 0.0

    for iter = 1:maxiter
        #update cavity messages and compute their convergence
        ε = update_cavities!(nodes, sumargexp, Mⁱʲ, ρⁱʲ, prior, model.T, updmess, newmess, newmarg, damp, μ_cutoff, model.Disease)

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
            _, ρ = compute_ρ!(inode, iindex, nodes[j], jindex, sumargexp, Mⁱʲ, ρⁱʲ, prior, model.T, model.Disease)
            nodes[j].ρs[iindex].fwm .= ρ.fwm
            nodes[j].ρs[iindex].bwm .= ρ.bwm
        end
    end

    # check unconvergence
    if ε > epsconv
        println("NOT converged after $maxiter iterations")
    end
    

    compute_marginals!(nodes, sumargexp, Mⁱʲ, ρⁱʲ, model.T, prior, updmess, newmarg, μ_cutoff, model.Disease)

    return nodes
end