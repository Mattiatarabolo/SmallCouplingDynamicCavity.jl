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

"""
    bethe_lattice(z, tmax, startfrom1)

Generates a Bethe lattice (tree) with degree z and depth tmax. If startfrom1 = true the center of the tree is vertex 1.

It returns a list where the first element is a list of vertices, and the second element is the list of edges.
"""


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