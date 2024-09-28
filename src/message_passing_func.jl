function update_single_message!(
    ε::Float64,
    jnode::Node{TI,TG},
    iindex::Int,
    ρ::FBm,
    M::Array{Float64,3},
    normmess::Float64,
    newmess::Message,
    damp::Float64,
    inode::Node{TI,TG},
    μ_cutoff::Float64) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    clear!(newmess)

    @inbounds @fastmath for t in 1:inode.model.T
        normmess = 0.0
        @inbounds @fastmath for x in 1:n_states(inode.model.Disease)
            normmess += ρ.fwm[x,t] * ρ.bwm[x,t]
        end
        newmess.m[t] =ρ.fwm[2,t] * ρ.bwm[2,t+1] / normmess
        newmess.μ[t] =  max(ρ.fwm[1,t] * M[1,1,t] * (ρ.bwm[1,t+1] - ρ.bwm[2,t+1]) / normmes, μ_cutoff)

        newmess.m[t] = jnode.cavities[iindex].m[t]*damp + newmess.m[t]*(1 - damp)
        newmess.μ[t] = jnode.cavities[iindex].μ[t]*damp + newmess.μ[t]*(1 - damp)

        ε = max(ε, abs(newmess.m[t] - jnode.cavities[iindex].m[t]))

        jnode.cavities[iindex].m[t] = newmess.m[t]
        jnode.cavities[iindex].μ[t] = newmess.μ[t]
    end

    # t = T+1
    normmess = 0.0
    @inbounds @fastmath for x in 1:n_states(inode.model.Disease)
        normmess += ρ.fwm[x,inode.model.T+1] * ρ.bwm[x,inode.model.T+1]
    end
    newmess.m[inode.model.T+1] = ρ.fwm[2,inode.model.T+1] * ρ.bwm[2,inode.model.T+1] / normmess
    newmess.m[inode.model.T+1] = jnode.cavities[iindex].m[inode.model.T+1]*damp + newmess.m[inode.model.T+1]*(1 - damp)
    ε = max(ε, abs(newmess.m[inode.model.T+1] - jnode.cavities[iindex].m[inode.model.T+1]))
    jnode.cavities[iindex].m[inode.model.T+1] = newmess.m[inode.model.T+1]

    if any(!isfinite, newmess.m) || any(!isfinite, newmess.μ)
        throw(DomainError("NaN evaluated when updating message!"))
    end
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

    @inbounds @fastmath for x in 1:n_states(infectionmodel)
        ρ.fwm[x, 1] = prior[x, inode.i]
        ρ.bwm[x, T+1] = inode.obs[x, T+1]
    end

    fill_transmat_cav!(M, inode, iindex, jnode, jindex, sumargexp, infectionmodel)

    # fwd-bwd update
    @inbounds @fastmath for t in 1:T
        @inbounds @fastmath for x1 in 1:n_states(infectionmodel)
            @inbounds @fastmath for x2 in 1:n_states(infectionmodel)
                ρ.fwm[x1, t+1] += ρ.fwm[x2, t] * M[x2, x1, t]
                ρ.bwm[x1, T+1-t] += ρ.bwm[x2, T+2-t] * M[x1, x2, T+1-t]
            end
        end
    end

    if any(!isfinite, ρ.fwm) || any(!isfinite, ρ.bwm)
        throw(DomainError("NaN evaluated when computing ρ!"))
    end
end


function update_single_marginal!(
    inode::Node{TI,TG}, 
    nodes::Vector{Node{TI,TG}}, 
    sumargexp::SumM, 
    M::Array{Float64, 3}, 
    ρ::FBm, 
    prior::Array{Float64, 2}, 
    T::Int,
    normmarg::Float64,
    infectionmodel::TI) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    
    compute_sumargexp!(inode, nodes, sumargexp)

    clear!(M, ρ)

    @inbounds @fatsmath for x in 1:n_states(infectionmodel)
        ρ.fwm[x, 1] = prior[x, inode.i]
        ρ.bwm[x, T+1] = inode.obs[x, T+1]
    end

    fill_transmat_marg!(M, inode, sumargexp, infectionmodel)

    # fwd-bwd update
    @inbounds @fastmath for t in 1:T
        @inbounds @fastmath for x1 in 1:n_states(infectionmodel)
            @inbounds @fastmath for x2 in 1:n_states(infectionmodel)
                ρ.fwm[x1, t+1] += ρ.fwm[x2, t] * M[x2, x1, t]
                ρ.bwm[x1, T+1-t] += ρ.bwm[x2, T+2-t] * M[x1, x2, T+1-t]
            end
        end
    end

    @inbounds @fastmath for t in 1:inode.model.T
        normmarg = 0.0
        @inbounds @fastmath for x in 1:n_states(infectionmodel)
            normmarg += ρ.fwm[x,t] * ρ.bwm[x,t]
        end
        @inbounds @fastmath for x in 1:n_states(infectionmodel)
            inode.marg.m[x,t] = ρ.fwm[x,t] * ρ.bwm[x,t] / normmarg
        end
    end

    # t = T+1
    normmarg = 0.0
    @inbounds @fastmath for x in 1:n_states(infectionmodel)
        normmarg += ρ.fwm[x,T+1] * ρ.bwm[x,T+1]
    end
    @inbounds @fastmath for x in 1:n_states(infectionmodel)
        inode.marg.m[x,T+1] = ρ.fwm[x,T+1] * ρ.bwm[x,T+1] / normmarg
    end
end


function compute_sumargexp!(
    inode::Node{TI,TG},
    nodes::Vector{Node{TI,TG}},
    sumargexp::SumM) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    clear!(sumargexp)

    for (kindex, k) in enumerate(inode.∂)
        iindex = nodes[k].∂_idx[inode.i]
        @inbounds @fastmath for t in 1:inode.model.T
            sumargexp.summ[t] += inode.cavities[kindex].m[t] * inode.νs[kindex][t]  #chiedere ad anna se è più veloce riga o colonna
            sumargexp.sumμ[t] += inode.cavities[kindex].μ[t] * nodes[k].νs[iindex][t]
        end
    end
end


function update_node!(
    ε::Float64,
    inode::Node{TI,TG}, 
    nodes::Vector{Node{TI,TG}}, 
    sumargexp::SumM, 
    M::Array{Float64, 3}, 
    ρ::FBm, 
    prior::Array{Float64, 2}, 
    T::Int,
    norm::Float64,
    newmess::Message,
    damp::Float64,
    μ_cutoff::Float64,
    infectionmodel::TI) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    compute_sumargexp!(inode, nodes, sumargexp)

    for (jindex, j) in enumerate(inode.∂)
        iindex = nodes[j].∂_idx[inode.i]
        compute_ρ!(inode, iindex, nodes[j], jindex, sumargexp, M, ρ, prior, T, infectionmodel)
        update_single_message!(ε, nodes[j], iindex, ρ, M, norm, newmess, damp, inode, μ_cutoff)
    end
end


function update_cavities!(
    ε::Float64,
    nodes::Vector{Node{TI,TG}},
    sumargexp::SumM,
    M::Array{Float64,3},
    ρ::FBm,
    prior::Array{Float64,2},
    T::Int,
    norm::Float64,
    newmess::Message,
    damp::Float64,
    μ_cutoff::Float64,
    infectionmodel::TI,
    rng::AbstractRNG) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    ε = 0.0

    for inode in shuffle(rng, nodes)
        update_node!(ε, inode, nodes, sumargexp, M, ρ, prior, T, norm, newmess, damp, μ_cutoff, infectionmodel)
    end
end


function compute_marginals!(
    nodes::Vector{Node{TI,TG}},
    sumargexp::SumM,
    M::Array{Float64,3},
    ρ::FBm, 
    T::Int64,
    prior::Array{Float64,2},
    norm::Float64,
    infectionmodel::TI,
    rng::AbstractRNG) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    for inode in shuffle(rng, nodes)
        update_single_marginal!(inode, nodes, sumargexp, M, ρ, prior, T, norm, infectionmodel)
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
        n_iter_nc::Int64 = 1,
        damp_nc::Float64 = 0.0,
        callback::Function=(x...) -> nothing
        rng::AbstractRNG=Xoshiro()) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

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
- `rng`: (Optional) Random number generator.

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
    callback::Function=(x...) -> nothing,
    rng::AbstractRNG=Xoshiro()) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    # Initialize prior probabilities based on the expected mean number of source patients (γ)
    prior = zeros(n_states(model.Disease), model.N)
    @inbounds @fastmath for i in 1:model.N
        prior[1, i] = (1 - γ) # x_i = S
        prior[2, i] = γ # x_i = I
    end

    # Format nodes for inference
    nodes = nodes_formatting(model, obsprob)

    # Initialize message objects
    M = TransMat(model.T, model.Disease)
    ρ = FBm(model.T, model.Disease)
    sumargexp = SumM(model.T)
    norm = 0.0
    newmess = Message(0, 0, model.T)
    newmarg = Marginal(0, model.T, model.Disease)

    ε = 0.0

    # Iteratively update cavity messages until convergence or maximum iterations reached
    for iter = 1:maxiter
        update_cavities!(ε, nodes, sumargexp, M, ρ, prior, model.T, norm, newmess, damp, μ_cutoff, model.Disease, rng)
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
            for inode in shuffle(rng, nodes)
                sumargexp = compute_sumargexp!(inode, nodes, sumargexp)
                for (jindex, j) in enumerate(inode.∂)
                    iindex = nodes[j].∂_idx[inode.i]
                    compute_ρ!(inode, iindex, nodes[j], jindex, sumargexp, M, ρ, prior, model.T, model.Disease)
                    clear!(newmess)
                    @inbounds @fastmath for t in 1:model.T
                        norm = 0.0
                        @inbounds @fastmath for x in 1:n_states(model.Disease)
                            norm += ρ.fwm[x,t] * ρ.bwm[x,t]
                        end
                        newmess.m[t] =ρ.fwm[2,t] * ρ.bwm[2,t+1] / norm
                        newmess.μ[t] =  max(ρ.fwm[1,t] * M[1,1,t] * (ρ.bwm[1,t+1] - ρ.bwm[2,t+1]) / norm, μ_cutoff)
                        newmess.m[t] = nodes[j].cavities[iindex].m[t]*damp_nc + newmess.m[t]*(1 - damp_nc)
                        newmess.μ[t] = nodes[j].cavities[iindex].μ[t]*damp_nc + newmess.μ[t]*(1 - damp_nc)
                        avg_mess[j][iindex].m[t] += newmess.m[t]
                        avg_mess[j][iindex].μ[t] += newmess.μ[t]
                        nodes[j].cavities[iindex].m[t] = newmess.m[t]
                        nodes[j].cavities[iindex].μ[t] = newmess.μ[t]
                    end
                    # t = T+1
                    norm = 0.0
                    @inbounds @fastmath for x in 1:n_states(model.Disease)
                        norm += ρ.fwm[x,model.T+1] * ρ.bwm[x,model.T+1]
                    end
                    newmess.m[model.T+1] = ρ.fwm[2,model.T+1] * ρ.bwm[2,model.T+1] / norm
                    newmess.m[model.T+1] = nodes[j].cavities[iindex].m[model.T+1]*damp_nc + newmess.m[model.T+1]*(1 - damp_nc)
                    avg_mess[j][iindex].m[model.T+1] += newmess.m[model.T+1]
                    nodes[j].cavities[iindex].m[model.T+1] = newmess.m[model.T+1]
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

    # Compute final marginal probabilities
    compute_marginals!(nodes, sumargexp, M, ρ, model.T, prior, norm, model.Disease, rng)

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
        n_iter_nc::Int64 = 1,
        damp_nc::Float64 = 0.0,
        callback::Function=(x...) -> nothing,
        rng::AbstractRNG=Xoshiro()) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

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
- `rng`: (Optional) Random number generator.

# Returns
- `nodes`: An array of [`Node`](@ref) objects representing the updated node states after inference.

"""
function run_SCDC(
    model::EpidemicModel{TI,TG},
    obsprob::Function,
    γ::Float64,
    maxiter::Vector{Int64},
    epsconv::Float64,
    damp::Vector{Float64};
    μ_cutoff::Float64 = -Inf,
    n_iter_nc::Int64 = 1,
    damp_nc::Float64 = 0.0,
    callback::Function=(x...) -> nothing,
    rng::AbstractRNG=Xoshiro()) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    # Debugging
    if length(maxiter) != length(damp)
        throw(DomainError("Length of maxiter and damp vectors must be the same!"))
    end

    # Initialize prior probabilities based on the expected mean number of source patients (γ)
    prior = zeros(n_states(model.Disease), model.N)
    @inbounds @fastmath for i in 1:model.N
        prior[1, i] = (1 - γ) # x_i = S
        prior[2, i] = γ # x_i = I
    end

    # Format nodes for inference
    nodes = nodes_formatting(model, obsprob)

    # Initialize message objects
    M = TransMat(model.T, model.Disease)
    ρ = FBm(model.T, model.Disease)
    sumargexp = SumM(model.T)
    norm = 0.0
    newmess = Message(0, 0, model.T)
    newmarg = Marginal(0, model.T, model.Disease)

    ε = 0.0

    # Iteratively update cavity messages until convergence or maximum iterations reached
    iter = 0
    check_convergence = false
    for (mi, d) in Iterators.zip(maxiter, damp)
        for _ in 1:mi
            update_cavities!(ε, nodes, sumargexp, M, ρ, prior, model.T, norm, newmess, damp, μ_cutoff, model.Disease, rng)
            iter += 1
            callback(nodes, iter, ε)
            
            # Check for convergence
            if ε < epsconv
                println("Converged after $iter iterations")
                check_convergence = true
                break
            end
        end

        if check_convergence
            break
        end
    end

    # Check if convergence not achieved
    if ε > epsconv
        println("NOT converged after $maxiter iterations")

        avg_mess = [[Message(node.i, j, model.T; zero_mess=true) for j in node.∂] for node in nodes]

        for iter in 1:n_iter_nc
            # compute average messages
            for inode in shuffle(rng, nodes)
                sumargexp = compute_sumargexp!(inode, nodes, sumargexp)
                for (jindex, j) in enumerate(inode.∂)
                    iindex = nodes[j].∂_idx[inode.i]
                    compute_ρ!(inode, iindex, nodes[j], jindex, sumargexp, M, ρ, prior, model.T, model.Disease)
                    clear!(newmess)
                    @inbounds @fastmath for t in 1:model.T
                        norm = 0.0
                        @inbounds @fastmath for x in 1:n_states(model.Disease)
                            norm += ρ.fwm[x,t] * ρ.bwm[x,t]
                        end
                        newmess.m[t] =ρ.fwm[2,t] * ρ.bwm[2,t+1] / norm
                        newmess.μ[t] =  max(ρ.fwm[1,t] * M[1,1,t] * (ρ.bwm[1,t+1] - ρ.bwm[2,t+1]) / norm, μ_cutoff)
                        newmess.m[t] = nodes[j].cavities[iindex].m[t]*damp_nc + newmess.m[t]*(1 - damp_nc)
                        newmess.μ[t] = nodes[j].cavities[iindex].μ[t]*damp_nc + newmess.μ[t]*(1 - damp_nc)
                        avg_mess[j][iindex].m[t] += newmess.m[t]
                        avg_mess[j][iindex].μ[t] += newmess.μ[t]
                        nodes[j].cavities[iindex].m[t] = newmess.m[t]
                        nodes[j].cavities[iindex].μ[t] = newmess.μ[t]
                    end
                    # t = T+1
                    norm = 0.0
                    @inbounds @fastmath for x in 1:n_states(model.Disease)
                        norm += ρ.fwm[x,model.T+1] * ρ.bwm[x,model.T+1]
                    end
                    newmess.m[model.T+1] = ρ.fwm[2,model.T+1] * ρ.bwm[2,model.T+1] / norm
                    newmess.m[model.T+1] = nodes[j].cavities[iindex].m[model.T+1]*damp_nc + newmess.m[model.T+1]*(1 - damp_nc)
                    avg_mess[j][iindex].m[model.T+1] += newmess.m[model.T+1]
                    nodes[j].cavities[iindex].m[model.T+1] = newmess.m[model.T+1]
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

    # Compute final marginal probabilities
    compute_marginals!(nodes, sumargexp, M, ρ, model.T, prior, norm, model.Disease, rng)

    return nodes
end

"""
    run_SCDC(nodes::Vector{Node{TI,TG}}, model::EpidemicModel{TI,TG}, γ::Float64, maxiter::Vector{Int64}, epsconv::Float64, damp::Vector{Float64}; μ_cutoff::Float64 = -Inf, n_iter_nc::Int64 = 1, damp_nc::Float64 = 0.0, callback::Function=(x...) -> nothing, rng::AbstractRNG=Xoshiro())

Run the SCDC algorithm for epidemic modeling. The algorithm resumes the message-passing iterations from the current state of the nodes.

This function performs SCDC inference on the specified epidemic model, using the provided evidence (likelihood) probability function, and other parameters such as the probability of being a patient zero, maximum number of iterations, convergence threshold, damping factor, etc. It iteratively updates cavity messages until convergence or until the maximum number of iterations is reached. It implements a dumping schedule for the damping factor, where the dumping factor is changed after a certain number of iterations, specified by the `maxiter` and `damp` arguments.

# Arguments
- `nodes::Vector{Node{TI,TG}}`: Vector of nodes in the epidemic model.
- `model::EpidemicModel{TI,TG}`: The epidemic model to be used.
- `γ::Float64`: A parameter for the algorithm (e.g., infection rate).
- `maxiter::Vector{Int64}`: Maximum number of iterations for the algorithm.
- `epsconv::Float64`: Convergence threshold for the algorithm.
- `damp::Vector{Float64}`: Damping factors for the algorithm.

# Keyword Arguments
- `μ_cutoff::Float64`: Cutoff value for some parameter μ (default is -Inf).
- `n_iter_nc::Int64`: Number of iterations for non-converging cases (default is 1).
- `damp_nc::Float64`: Damping factor for non-converging cases (default is 0.0).
- `callback::Function`: Callback function to be called during iterations (default does nothing).
- `rng::AbstractRNG`: Random number generator (default is Xoshiro).

# Returns
- `Vector{Node{TI,TG}}`: The updated vector of nodes after running the algorithm.
"""
function run_SCDC(
    nodes::Vector{Node{TI,TG}},
    model::EpidemicModel{TI,TG},
    γ::Float64,
    maxiter::Int64,
    epsconv::Float64,
    damp::Float64;
    μ_cutoff::Float64 = -Inf,
    n_iter_nc::Int64 = 1,
    damp_nc::Float64 = 0.0,
    callback::Function=(x...) -> nothing,
    rng::AbstractRNG=Xoshiro()) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    # Initialize prior probabilities based on the expected mean number of source patients (γ)
    prior = zeros(n_states(model.Disease), model.N)
    @inbounds @fastmath for i in 1:model.N
        prior[1, i] = (1 - γ) # x_i = S
        prior[2, i] = γ # x_i = I
    end

    # Initialize message objects
    M = TransMat(model.T, model.Disease)
    ρ = FBm(model.T, model.Disease)
    sumargexp = SumM(model.T)
    norm = 0.0
    newmess = Message(0, 0, model.T)
    newmarg = Marginal(0, model.T, model.Disease)

    ε = 0.0

    # Iteratively update cavity messages until convergence or maximum iterations reached
    for iter = 1:maxiter
        update_cavities!(ε, nodes, sumargexp, M, ρ, prior, model.T, norm, newmess, damp, μ_cutoff, model.Disease, rng)
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
            for inode in shuffle(rng, nodes)
                sumargexp = compute_sumargexp!(inode, nodes, sumargexp)
                for (jindex, j) in enumerate(inode.∂)
                    iindex = nodes[j].∂_idx[inode.i]
                    compute_ρ!(inode, iindex, nodes[j], jindex, sumargexp, M, ρ, prior, model.T, model.Disease)
                    clear!(newmess)
                    @inbounds @fastmath for t in 1:model.T
                        norm = 0.0
                        @inbounds @fastmath for x in 1:n_states(model.Disease)
                            norm += ρ.fwm[x,t] * ρ.bwm[x,t]
                        end
                        newmess.m[t] =ρ.fwm[2,t] * ρ.bwm[2,t+1] / norm
                        newmess.μ[t] =  max(ρ.fwm[1,t] * M[1,1,t] * (ρ.bwm[1,t+1] - ρ.bwm[2,t+1]) / norm, μ_cutoff)
                        newmess.m[t] = nodes[j].cavities[iindex].m[t]*damp_nc + newmess.m[t]*(1 - damp_nc)
                        newmess.μ[t] = nodes[j].cavities[iindex].μ[t]*damp_nc + newmess.μ[t]*(1 - damp_nc)
                        avg_mess[j][iindex].m[t] += newmess.m[t]
                        avg_mess[j][iindex].μ[t] += newmess.μ[t]
                        nodes[j].cavities[iindex].m[t] = newmess.m[t]
                        nodes[j].cavities[iindex].μ[t] = newmess.μ[t]
                    end
                    # t = T+1
                    norm = 0.0
                    @inbounds @fastmath for x in 1:n_states(model.Disease)
                        norm += ρ.fwm[x,model.T+1] * ρ.bwm[x,model.T+1]
                    end
                    newmess.m[model.T+1] = ρ.fwm[2,model.T+1] * ρ.bwm[2,model.T+1] / norm
                    newmess.m[model.T+1] = nodes[j].cavities[iindex].m[model.T+1]*damp_nc + newmess.m[model.T+1]*(1 - damp_nc)
                    avg_mess[j][iindex].m[model.T+1] += newmess.m[model.T+1]
                    nodes[j].cavities[iindex].m[model.T+1] = newmess.m[model.T+1]
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

    # Compute final marginal probabilities
    compute_marginals!(nodes, sumargexp, M, ρ, model.T, prior, norm, model.Disease, rng)

    return nodes
end


"""
    run_SCDC(nodes::Vector{Node{TI,TG}}, model::EpidemicModel{TI,TG}, γ::Float64, maxiter::Vector{Int64}, epsconv::Float64, damp::Vector{Float64}; μ_cutoff::Float64 = -Inf, n_iter_nc::Int64 = 1, damp_nc::Float64 = 0.0, callback::Function=(x...) -> nothing, rng::AbstractRNG=Xoshiro())

Run the SCDC algorithm for epidemic modeling. The algorithm resumes the message-passing iterations from the current state of the nodes.

This function performs SCDC inference on the specified epidemic model, using the provided evidence (likelihood) probability function, and other parameters such as the probability of being a patient zero, maximum number of iterations, convergence threshold, damping factor, etc. It iteratively updates cavity messages until convergence or until the maximum number of iterations is reached. It implements a dumping schedule for the damping factor, where the dumping factor is changed after a certain number of iterations, specified by the `maxiter` and `damp` arguments.

# Arguments
- `nodes::Vector{Node{TI,TG}}`: Vector of nodes in the epidemic model.
- `model::EpidemicModel{TI,TG}`: The epidemic model to be used.
- `γ::Float64`: A parameter for the algorithm (e.g., infection rate).
- `maxiter::Vector{Int64}`: Maximum number of iterations for the algorithm.
- `epsconv::Float64`: Convergence threshold for the algorithm.
- `damp::Vector{Float64}`: Damping factors for the algorithm.

# Keyword Arguments
- `μ_cutoff::Float64`: Cutoff value for some parameter μ (default is -Inf).
- `n_iter_nc::Int64`: Number of iterations for non-converging cases (default is 1).
- `damp_nc::Float64`: Damping factor for non-converging cases (default is 0.0).
- `callback::Function`: Callback function to be called during iterations (default does nothing).
- `rng::AbstractRNG`: Random number generator (default is Xoshiro).

# Returns
- `Vector{Node{TI,TG}}`: The updated vector of nodes after running the algorithm.
"""
function run_SCDC(
    nodes::Vector{Node{TI,TG}},
    model::EpidemicModel{TI,TG},
    γ::Float64,
    maxiter::Vector{Int64},
    epsconv::Float64,
    damp::Vector{Float64};
    μ_cutoff::Float64 = -Inf,
    n_iter_nc::Int64 = 1,
    damp_nc::Float64 = 0.0,
    callback::Function=(x...) -> nothing,
    rng::AbstractRNG=Xoshiro()) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}

    # Debugging
    if length(maxiter) != length(damp)
        throw(DomainError("Length of maxiter and damp vectors must be the same!"))
    end

    # Initialize prior probabilities based on the expected mean number of source patients (γ)
    prior = zeros(n_states(model.Disease), model.N)
    @inbounds @fastmath for i in 1:model.N
        prior[1, i] = (1 - γ) # x_i = S
        prior[2, i] = γ # x_i = I
    end

    # Initialize message objects
    M = TransMat(model.T, model.Disease)
    ρ = FBm(model.T, model.Disease)
    sumargexp = SumM(model.T)
    norm = 0.0
    newmess = Message(0, 0, model.T)
    newmarg = Marginal(0, model.T, model.Disease)

    ε = 0.0

    # Iteratively update cavity messages until convergence or maximum iterations reached
    iter = 0
    check_convergence = false
    for (mi, d) in Iterators.zip(maxiter, damp)
        for _ in 1:mi
            update_cavities!(ε, nodes, sumargexp, M, ρ, prior, model.T, norm, newmess, damp, μ_cutoff, model.Disease, rng)
            iter += 1
            callback(nodes, iter, ε)
            
            # Check for convergence
            if ε < epsconv
                println("Converged after $iter iterations")
                check_convergence = true
                break
            end
        end

        if check_convergence
            break
        end
    end

    # Check if convergence not achieved
    if ε > epsconv
        println("NOT converged after $maxiter iterations")

        avg_mess = [[Message(node.i, j, model.T; zero_mess=true) for j in node.∂] for node in nodes]

        for iter in 1:n_iter_nc
            # compute average messages
            for inode in shuffle(rng, nodes)
                sumargexp = compute_sumargexp!(inode, nodes, sumargexp)
                for (jindex, j) in enumerate(inode.∂)
                    iindex = nodes[j].∂_idx[inode.i]
                    compute_ρ!(inode, iindex, nodes[j], jindex, sumargexp, M, ρ, prior, model.T, model.Disease)
                    clear!(newmess)
                    @inbounds @fastmath for t in 1:model.T
                        norm = 0.0
                        @inbounds @fastmath for x in 1:n_states(model.Disease)
                            norm += ρ.fwm[x,t] * ρ.bwm[x,t]
                        end
                        newmess.m[t] =ρ.fwm[2,t] * ρ.bwm[2,t+1] / norm
                        newmess.μ[t] =  max(ρ.fwm[1,t] * M[1,1,t] * (ρ.bwm[1,t+1] - ρ.bwm[2,t+1]) / norm, μ_cutoff)
                        newmess.m[t] = nodes[j].cavities[iindex].m[t]*damp_nc + newmess.m[t]*(1 - damp_nc)
                        newmess.μ[t] = nodes[j].cavities[iindex].μ[t]*damp_nc + newmess.μ[t]*(1 - damp_nc)
                        avg_mess[j][iindex].m[t] += newmess.m[t]
                        avg_mess[j][iindex].μ[t] += newmess.μ[t]
                        nodes[j].cavities[iindex].m[t] = newmess.m[t]
                        nodes[j].cavities[iindex].μ[t] = newmess.μ[t]
                    end
                    # t = T+1
                    norm = 0.0
                    @inbounds @fastmath for x in 1:n_states(model.Disease)
                        norm += ρ.fwm[x,model.T+1] * ρ.bwm[x,model.T+1]
                    end
                    newmess.m[model.T+1] = ρ.fwm[2,model.T+1] * ρ.bwm[2,model.T+1] / norm
                    newmess.m[model.T+1] = nodes[j].cavities[iindex].m[model.T+1]*damp_nc + newmess.m[model.T+1]*(1 - damp_nc)
                    avg_mess[j][iindex].m[model.T+1] += newmess.m[model.T+1]
                    nodes[j].cavities[iindex].m[model.T+1] = newmess.m[model.T+1]
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

    # Compute final marginal probabilities
    compute_marginals!(nodes, sumargexp, M, ρ, model.T, prior, norm, model.Disease, rng)

    return nodes
end