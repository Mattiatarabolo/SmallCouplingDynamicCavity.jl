function clear!(
    M::Array{Float64,3}, 
    ρ::FBm)
    fill!(M, 0.0)
    fill!(ρ.fwm, 1.0)
    fill!(ρ.bwm, 1.0)
end

function clear!(
    M::Array{Float64,3}, 
    ρ::FBm, 
    updmess::Updmess)

    fill!(M, 0.0)
    fill!(ρ.fwm, 1.0)
    fill!(ρ.bwm, 1.0)
    fill!(updmess.lognumm, 0.0)
    fill!(updmess.lognumμ, 0.0)
    fill!(updmess.signμ, 1.0)
    fill!(updmess.logZ, 0.0)
end

function clear!(
    updmess::Updmess, 
    newmess::Message)

    fill!(updmess.lognumm, 0.0)
    fill!(updmess.lognumμ, 0.0)
    fill!(updmess.signμ, 1.0)
    fill!(updmess.logZ, 0.0)
    fill!(newmess.m, 1.0)
    fill!(newmess.μ, 0.0)
end

function clear!(
    updmess::Updmess, 
    newmarg::Marginal)

    fill!(updmess.lognumm, 0.0)
    fill!(updmess.lognumμ, 0.0)
    fill!(updmess.signμ, 1.0)
    fill!(updmess.logZ, 0.0)
    fill!(newmarg.m, 1.0)
    fill!(newmarg.μ, 0.0)
end

function clear!(SumM::SumM)
    fill!(SumM.summ, 0.0)
    fill!(SumM.sumμ, 0.0)
end

function normupdate(
    oldmess::Vector{Float64}, 
    newmess::Vector{Float64})
    return maximum(abs.(oldmess .- newmess))
end

function ρ_norm(ρ::Vector{Float64})
    return ρ ./ sum(ρ)
end

function nodes_formatting(
    G::SimpleGraph{Int64}, 
    obsmat::Array{Float64,2}, 
    obsprob::Function, 
    ν::Array{Float64,3}, 
    T::Int64,
    nr_states::Int; 
    r::Union{Array{Float64,2},Nothing}=nothing, 
    σ::Union{Array{Float64,2},Nothing}=nothing)

    nodes = Vector{Node}()

    for i in 1:nv(G)
        obs = ones(nr_states, T + 1)
        obs[1, :] = [obsprob(Ob, 0.0) for Ob in obsmat[i, :]]
        obs[2, :] = [obsprob(Ob, 1.0) for Ob in obsmat[i, :]]
        if nr_states == 3
            obs[3, :] = [obsprob(Ob, 2.0) for Ob in obsmat[i, :]]
        end

        ∂ = neighbors(G, i)

        ν∂ = [ν[k, i, :] for k in ∂]

        if r === nothing && σ === nothing
            push!(nodes, Node(i, ∂, T, ν∂, obs, nr_states))
        elseif typeof(r) == Array{Float64,2} && σ === nothing
            rᵢᵗ = r[i, :]
            push!(nodes, Node(i, ∂, T, ν∂, obs, nr_states, rᵢᵗ=rᵢᵗ))
        else
            rᵢᵗ = r[i, :]
            σᵢᵗ = σ[i, :]
            push!(nodes, Node(i, ∂, T, ν∂, obs, nr_states, rᵢᵗ=rᵢᵗ, σᵢᵗ=σᵢᵗ))
        end
    end
    return collect(nodes)
end

function sim_epidemics(
    G::SimpleGraph{Int64}, 
    ν::Array{Float64,3}, 
    T::Int64,
    model::Symbol;
    r::Union{Array{Float64,2},Nothing}=nothing, 
    σ::Union{Array{Float64,2},Nothing}=nothing,
    patient_zero::Union{Vector{Int},Nothing}=nothing,
    γ::Union{Float64,Nothing}=nothing)

    inf₀ = false
    if patient_zero === nothing && γ != nothing
        while !inf₀
            patient_zero = rand(Binomial(1,γ), nv(G))
            patient_zero = findall(x->x==1, patient_zero)
            inf₀ = !isempty(patient_zero)
        end
    elseif patient_zero === nothing && γ != nothing
        while !inf₀
            patient_zero = rand(Binomial(1,\/nv(G)), nv(G))
            patient_zero = findall(x->x==1, patient_zero)
            inf₀ = !isempty(patient_zero)
        end
    end

    config = zeros(nv(G), T + 1)

    config[patient_zero,1] .+= 1.0

    if model == :SI
        hs = zeros(nv(G))
        for t in 1:T
            hs = config[:, t]' * ν[:, :, t]
            config[:, t+1] = [x + (1 - x) * rand(Bernoulli(1 - exp(h))) for (x, h) in zip(config[:, t], hs)]
        end
    elseif model == :SIS
        hs = zeros(nv(G))
        for t in 1:T
            hs = config[:, t]' * ν[:, :, t]
            config[:, t+1] = [x * rand(Bernoulli(1 - r)) + (1 - x) * rand(Bernoulli(1 - exp(h))) for (x, h, r) in zip(config[:, t], hs, r[:, t])]
        end
    elseif model == :SIR
        function W_SIR(x::Float64, y::Float64, h::Float64, r::Float64)
            if x == 0.0
                return (y == 0.0) * exp(h)
            elseif x == 1.0
                return (y == 0.0) * (1 - exp(h)) + (y == 1.0) * (1 - r)
            elseif x == 2.0
                return (y == 1.0) * r + (y == 2.0)
            else
                throw(ArgumentError("Invalid value for y"))
            end
        end
        hs = zeros(nv(G))
        for t in 1:T
            hs = [Float64(x == 1.0) for x in config[:, t]]' * ν[:, :, t]
            config[:, t+1] = [
                if (u <= W_SIR(0.0, x, h, r))
                    0.0
                elseif (W_SIR(0.0, x, h, r) < u <= W_SIR(0.0, x, h, r) + W_SIR(1.0, x, h, r))
                    1.0
                else
                    2.0
                end for (x, h, r, u) in zip(config[:, t], hs, r[:, t], rand(Float64, nv(G)))
            ]
        end
    elseif model == :SIRS
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
        hs = zeros(nv(G))
        for t in 1:T
            hs = [Float64(x == 1.0) for x in config[:, t]]' * ν[:, :, t]
            config[:, t+1] = [
                if (u <= W_SIRS(0.0, x, h, r, σ))
                    0.0
                elseif (W_SIRS(0.0, x, h, r, σ) < u <= W_SIRS(0.0, x, h, r, σ) + W_SIRS(1.0, x, h, r, σ))
                    1.0
                else
                    2.0
                end for (x, h, r, σ, u) in zip(config[:, t], hs, r[:, t], σ[:, t], rand(Float64, nv(G)))
            ]
        end
    else
        throw(ArgumentError("The only available models are SI, SIS, SIR and SIRS"))
    end
    return config
end

function bethe_lattice(z::Int,tmax::Int,startfrom1::Bool)
    # trivial case of tmax = 0, return only 1 node and empty set of edges
    tmax == 0 && return V = [0], Vector{Vector{Int}}(undef,0)


    V = collect(0:z)
    E = collect([[0,i] for i=1:z])
    tmax ==1 && return V,E
    shell_nodes(t) = z*(z-1)^(t-1)
    leaves = V[2:end]

    for t = 2:tmax
        Nt = shell_nodes(t)
        newnodes = collect(V[end]+1:V[end]+Nt)
        append!(V,newnodes)
        nleaves = length(leaves)
        for j = 1:nleaves
            newedges = collect([[leaves[j],newnodes[k]] for k = 1+ (z-1)*(j-1): (z-1)*j] )
            append!(E,newedges)
        end
        leaves = copy(newnodes)

    end
    startfrom1 && return V+ones(Int,length(V)), E + [ones(Int,2) for _=1:length(E)]
    !startfrom1 && return V,E

end

function ROC_curve(xₜ::Vector{Float64},x::Vector{Float64},N::Int64;normalize::Bool = true)
    if sum(isnan.(x))!=0
    	return zeros(N+1),zeros(N+1),NaN
    end
    xₜ=xₜ[sortperm(abs.(x),rev=true)]
    tp,fp=zeros(N+1),zeros(N+1)
    s=0.0
    for i=1:N
        if xₜ[i]!=0.0
            fp[i+1] , tp[i+1] = fp[i], tp[i]+1
        else
            fp[i+1] , tp[i+1] = fp[i]+1, tp[i]
            s+=tp[i]
        end
    end
	s/=(tp[end]*fp[end])
    if normalize
		fp/=maximum(fp)
		tp/=maximum(tp)
	end
	return fp,tp,s
end