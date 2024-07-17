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
    μ_cutoff::Float64) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    clear!(updmess, newmess)

    updmess.lognumm .= log.(ρ.fwm) .+ log.(ρ.bwm)
    updmess.signμ .= sign.(ρ.bwm[1, 2:end] - ρ.bwm[2, 2:end])
    updmess.lognumμ .= log.(ρ.fwm[1, 1:end-1]) .+ log.(M[1, 1, :]) .+ log.(abs.(ρ.bwm[1, 2:end] - ρ.bwm[2, 2:end]))
    updmess.logZ .= log.(dropdims(sum(ρ.fwm .* ρ.bwm, dims=1), dims=1))

    newmess.m .= exp.(updmess.lognumm[2, :] .- updmess.logZ)
    newmess.μ .= max.(updmess.signμ .* exp.(updmess.lognumμ .- updmess.logZ[1:end-1]), μ_cutoff)

    newmess.m .= jnode.cavities[iindex].m.*damp .+ newmess.m.*(1 - damp)
    newmess.μ .= jnode.cavities[iindex].μ.*damp .+ newmess.μ.*(1 - damp)

    if any(!isfinite, newmess.m) || any(!isfinite, newmess.μ)
        throw(DomainError("NaN evaluated when updating message!"))
    end

    ε = normupdate(jnode.cavities[iindex].m, newmess.m)#, normupdate(jnode.cavities[iindex].μ, newmess.μ))

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
    infectionmodel::TI) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    clear!(M, ρ)

    ρ.fwm[:, 1] .= prior[:, inode.i]
    ρ.bwm[:, T+1] .= inode.obs[:, T+1]

    fill_transmat_cav!(M, inode, iindex, jnode, jindex, sumargexp, infectionmodel)

    # fwd-bwd update
    for t in 1:T
        ρ.fwm[:, t+1] .= (ρ.fwm[:, t]' * M[:, :, t])'
        ρ.bwm[:, T+1-t] .= M[:, :, T+1-t] * ρ.bwm[:, T+2-t]
    end

    if any(!isfinite, ρ.fwm) || any(!isfinite, ρ.bwm)
        throw(DomainError("NaN evaluated when computing ρ!"))
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
    infectionmodel::TI) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    compute_sumargexp!(inode, nodes, sumargexp)

    clear!(M, ρ)

    ρ.fwm[:, 1] .= prior[:, inode.i]
    ρ.bwm[:, T+1] .= inode.obs[:, T+1]

    fill_transmat_marg!(M, inode, sumargexp, infectionmodel)

    # fwd-bwd update
    for t in 1:T
        ρ.fwm[:, t+1] .= (ρ.fwm[:, t]' * M[:, :, t])'
        ρ.bwm[:, T+1-t] .= M[:, :, T+1-t] * ρ.bwm[:, T+2-t]
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
    sumargexp::SumM) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    clear!(sumargexp)

    for (kindex, k) in enumerate(inode.∂)
        iindex = nodes[k].∂_idx[inode.i]
        sumargexp.summ .+= inode.cavities[kindex].m[1:end-1] .* inode.νs[kindex]  #chiedere ad anna se è più veloce riga o colonna
        sumargexp.sumμ .+= inode.cavities[kindex].μ .* nodes[k].νs[iindex]
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
    infectionmodel::TI) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

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
    infectionmodel::TI) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    ε = 0.0

    for inode in shuffle(nodes)
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
    infectionmodel::TI) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    for inode in nodes
        newmarg = update_single_marginal!(inode, nodes, sumargexp, M, ρ, prior, T, updmess, newmarg, μ_cutoff, infectionmodel)
        inode.marg.m .= newmarg.m
    end
end

"""
    run_SCDC(
        model::EpidemicModel{TI,TG},
        obsprob::Function,
        γ::Float64,
        maxiter::Int64,
        epsconv::Float64,
        damp::Float64;
        μ_cutoff::Float64 = -Inf,
        callback::Function=(x...) -> nothing) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

Runs the Small Coupling Dynamic Cavity (SCDC) inference algorithm.

This function performs SCDC inference on the specified epidemic model, using the provided evidence (likelihood) probability function, and other parameters such as the probability of being a patient zero, maximum number of iterations, convergence threshold, damping factor, etc. It iteratively updates cavity messages until convergence or until the maximum number of iterations is reached.

# Arguments
- `model`: An [`EpidemicModel`](@ref) representing the epidemic model.
- `obsprob`: A function representing the evidence (likelihood) probability p(O|x) of an observation O given the planted state x.
- `γ`: The probability of being a patient zero.
- `maxiter`: The maximum number of iterations.
- `epsconv`: The convergence threshold of the algorithm.
- `damp`: The damping factor of the algorithm.
- `μ_cutoff`: (Optional) Lower cut-off for the values of μ.
- `n_iter_nc`: (Optional) Number of iterations for non-converged messages. The messages are averaged over this number of iterations.
- `damp_nc`: (Optional) Damping factor for non-converged messages.
- `callback`: (Optional) A callback function to monitor the progress of the algorithm.

# Returns
- `nodes`: An array of [`Node`](@ref) objects representing the updated node states after inference.

"""
function run_SCDC(
    model::EpidemicModel{TI,TG},
    obsprob::Function,
    γ::Float64,
    maxiter::Int64,
    epsconv::Float64,
    damp::Float64;
    μ_cutoff::Float64 = -Inf,
    n_iter_nc::Int64 = 1,
    damp_nc::Float64 = 0.0,
    callback::Function=(x...) -> nothing) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    # Initialize prior probabilities based on the expected mean number of source patients (γ)
    prior = zeros(n_states(model.Disease), model.N)
    prior[1, :] .= (1 - γ) # x_i = S
    prior[2, :] .= γ # x_i = I

    # Format nodes for inference
    nodes = nodes_formatting(model, obsprob)

    # Initialize message objects
    M = TransMat(model.T, model.Disease)
    ρ = FBm(model.T, model.Disease)
    sumargexp = SumM(model.T)
    updmess = Updmess(model.T, model.Disease)
    newmess = Message(0, 0, model.T)
    newmarg = Marginal(0, model.T, model.Disease)

    ε = 0.0

    # Iteratively update cavity messages until convergence or maximum iterations reached
    for iter = 1:maxiter
        ε = update_cavities!(nodes, sumargexp, M, ρ, prior, model.T, updmess, newmess, newmarg, damp, μ_cutoff, model.Disease)
        callback(nodes, iter, ε)

        # Check for convergence
        if ε < epsconv
            println("Converged after $iter iterations")
            break
        end
    end

    # Check if convergence not achieved
    if ε > epsconv
        println("NOT converged after $maxiter iterations")

        avg_mess = [[Message(node.i, j, model.T; zero_mess=true) for j in node.∂] for node in nodes]

        for iter in 1:n_iter_nc
            # compute average messages
            for inode in nodes
                sumargexp = compute_sumargexp!(inode, nodes, sumargexp)
                for (jindex, j) in enumerate(inode.∂)
                    iindex = nodes[j].∂_idx[inode.i]
                    M, ρ = compute_ρ!(inode, iindex, nodes[j], jindex, sumargexp, M, ρ, prior, model.T, model.Disease)
                    clear!(updmess, newmess)
                    updmess.lognumm .= log.(ρ.fwm) .+ log.(ρ.bwm)
                    updmess.signμ .= sign.(ρ.bwm[1, 2:end] - ρ.bwm[2, 2:end])
                    updmess.lognumμ .= log.(ρ.fwm[1, 1:end-1]) .+ log.(M[1, 1, :]) .+ log.(abs.(ρ.bwm[1, 2:end] - ρ.bwm[2, 2:end]))
                    updmess.logZ .= log.(dropdims(sum(ρ.fwm .* ρ.bwm, dims=1), dims=1))

                    newmess.m .= exp.(updmess.lognumm[2, :] .- updmess.logZ)
                    newmess.μ .= max.(updmess.signμ .* exp.(updmess.lognumμ .- updmess.logZ[1:end-1]), μ_cutoff)

                    newmess.m .= nodes[j].cavities[iindex].m.*damp_nc .+ newmess.m.*(1 - damp_nc)
                    newmess.μ .= nodes[j].cavities[iindex].μ.*damp_nc .+ newmess.μ.*(1 - damp_nc)
                    
                    avg_mess[j][iindex].m .+= newmess.m
                    avg_mess[j][iindex].μ .+= newmess.μ

                    nodes[j].cavities[iindex].m .= newmess.m
                    nodes[j].cavities[iindex].μ .= newmess.μ 
                end
            end
        end
        
        if n_iter_nc != 0
            # compute average messages
            for inode in nodes
                for (_, j) in enumerate(inode.∂)
                    iindex = nodes[j].∂_idx[inode.i]
                    nodes[j].cavities[iindex].m .= avg_mess[j][iindex].m ./ n_iter_nc
                    nodes[j].cavities[iindex].μ .= avg_mess[j][iindex].μ ./ n_iter_nc
                end
            end
        end
    end
    

    # Update messages between nodes
    for inode in nodes
        sumargexp = compute_sumargexp!(inode, nodes, sumargexp)
        for (jindex, j) in enumerate(inode.∂)
            iindex = nodes[j].∂_idx[inode.i]
            _, ρ = compute_ρ!(inode, iindex, nodes[j], jindex, sumargexp, M, ρ, prior, model.T, model.Disease)
            nodes[j].ρs[iindex].fwm .= ρ.fwm
            nodes[j].ρs[iindex].bwm .= ρ.bwm
        end
    end

    # Compute final marginal probabilities
    compute_marginals!(nodes, sumargexp, M, ρ, model.T, prior, updmess, newmarg, μ_cutoff, model.Disease)

    return nodes
end


"""
    run_SCDC(
        model::EpidemicModel{TI,TG},
        obsprob::Function,
        γ::Float64,
        maxiter::Vector{Int64},
        epsconv::Float64,
        damp::Vector{Float64};
        μ_cutoff::Float64 = -Inf,
        callback::Function=(x...) -> nothing) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

Runs the Small Coupling Dynamic Cavity (SCDC) inference algorithm.

This function performs SCDC inference on the specified epidemic model, using the provided evidence (likelihood) probability function, and other parameters such as the probability of being a patient zero, maximum number of iterations, convergence threshold, damping factor, etc. It iteratively updates cavity messages until convergence or until the maximum number of iterations is reached. It implements a dumping schedule for the damping factor, where the dumping factor is changed after a certain number of iterations, specified by the `maxiter` and `damp` arguments.

# Arguments
- `model`: An [`EpidemicModel`](@ref) representing the epidemic model.
- `obsprob`: A function representing the evidence (likelihood) probability p(O|x) of an observation O given the planted state x.
- `γ`: The probability of being a patient zero.
- `maxiter`: Vector of maximum number of iterations for each damping factor. 
- `epsconv`: The convergence threshold of the algorithm.
- `damp`: Vector of damping factors used in the damping schedule.
- `μ_cutoff`: (Optional) Lower cut-off for the values of μ.
- `n_iter_nc`: (Optional) Number of iterations for non-converged messages. The messages are averaged over this number of iterations.
- `damp_nc`: (Optional) Damping factor for non-converged messages.
- `callback`: (Optional) A callback function to monitor the progress of the algorithm.

# Returns
- `nodes`: An array of [`Node`](@ref) objects representing the updated node states after inference.

"""
function run_SCDC(
    model::EpidemicModel{TI,TG},
    obsprob::Function,
    γ::Float64,
    maxiter::Vector{Int64},
    epsconv::Float64,
    damp::Float64;
    μ_cutoff::Vector{Float64} = [-Inf for _ in 1::length(maxiter)],
    n_iter_nc::Int64 = 1,
    damp_nc::Float64 = 0.0,
    callback::Function=(x...) -> nothing) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    # Debugging
    if length(maxiter) != length(damp)
        throw(DomainError("Length of maxiter and damp vectors must be the same!"))
    end

    # Initialize prior probabilities based on the expected mean number of source patients (γ)
    prior = zeros(n_states(model.Disease), model.N)
    prior[1, :] .= (1 - γ) # x_i = S
    prior[2, :] .= γ # x_i = I

    # Format nodes for inference
    nodes = nodes_formatting(model, obsprob)

    # Initialize message objects
    M = TransMat(model.T, model.Disease)
    ρ = FBm(model.T, model.Disease)
    sumargexp = SumM(model.T)
    updmess = Updmess(model.T, model.Disease)
    newmess = Message(0, 0, model.T)
    newmarg = Marginal(0, model.T, model.Disease)

    ε = 0.0

    # Iteratively update cavity messages until convergence or maximum iterations reached
    iter = 0
    for (mi, d) in Iterators.zip(maxiter, damp)
        for i in 1:mi
            ε = update_cavities!(nodes, sumargexp, M, ρ, prior, model.T, updmess, newmess, newmarg, d, μ_cutoff, model.Disease)

            iter += 1
            callback(nodes, iter, ε)
            
            # Check for convergence
            if ε < epsconv
                println("Converged after $iter iterations")
                break
            end
        end
    end

    # Check if convergence not achieved
    if ε > epsconv
        println("NOT converged after $maxiter iterations")

        avg_mess = [[Message(node.i, j, model.T; zero_mess=true) for j in node.∂] for node in nodes]

        for iter in 1:n_iter_nc
            # compute average messages
            for inode in nodes
                sumargexp = compute_sumargexp!(inode, nodes, sumargexp)
                for (jindex, j) in enumerate(inode.∂)
                    iindex = nodes[j].∂_idx[inode.i]
                    M, ρ = compute_ρ!(inode, iindex, nodes[j], jindex, sumargexp, M, ρ, prior, model.T, model.Disease)
                    clear!(updmess, newmess)
                    updmess.lognumm .= log.(ρ.fwm) .+ log.(ρ.bwm)
                    updmess.signμ .= sign.(ρ.bwm[1, 2:end] - ρ.bwm[2, 2:end])
                    updmess.lognumμ .= log.(ρ.fwm[1, 1:end-1]) .+ log.(M[1, 1, :]) .+ log.(abs.(ρ.bwm[1, 2:end] - ρ.bwm[2, 2:end]))
                    updmess.logZ .= log.(dropdims(sum(ρ.fwm .* ρ.bwm, dims=1), dims=1))

                    newmess.m .= exp.(updmess.lognumm[2, :] .- updmess.logZ)
                    newmess.μ .= max.(updmess.signμ .* exp.(updmess.lognumμ .- updmess.logZ[1:end-1]), μ_cutoff)

                    newmess.m .= nodes[j].cavities[iindex].m.*damp_nc .+ newmess.m.*(1 - damp_nc)
                    newmess.μ .= nodes[j].cavities[iindex].μ.*damp_nc .+ newmess.μ.*(1 - damp_nc)
                    
                    avg_mess[j][iindex].m .+= newmess.m
                    avg_mess[j][iindex].μ .+= newmess.μ

                    nodes[j].cavities[iindex].m .= newmess.m
                    nodes[j].cavities[iindex].μ .= newmess.μ 
                end
            end
        end
        
        if n_iter_nc != 0
            # compute average messages
            for inode in nodes
                for (_, j) in enumerate(inode.∂)
                    iindex = nodes[j].∂_idx[inode.i]
                    nodes[j].cavities[iindex].m .= avg_mess[j][iindex].m ./ n_iter_nc
                    nodes[j].cavities[iindex].μ .= avg_mess[j][iindex].μ ./ n_iter_nc
                end
            end
        end
    end
    

    # Update messages between nodes
    for inode in nodes
        sumargexp = compute_sumargexp!(inode, nodes, sumargexp)
        for (jindex, j) in enumerate(inode.∂)
            iindex = nodes[j].∂_idx[inode.i]
            _, ρ = compute_ρ!(inode, iindex, nodes[j], jindex, sumargexp, M, ρ, prior, model.T, model.Disease)
            nodes[j].ρs[iindex].fwm .= ρ.fwm
            nodes[j].ρs[iindex].bwm .= ρ.bwm
        end
    end

    # Compute final marginal probabilities
    compute_marginals!(nodes, sumargexp, M, ρ, model.T, prior, updmess, newmarg, μ_cutoff, model.Disease)

    return nodes
end