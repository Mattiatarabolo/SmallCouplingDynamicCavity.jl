function clear!(SumM::SumM)
    fill!(SumM.summ, 0.0)
    fill!(SumM.sumμ, 0.0)
end


"""
    bethe_lattice(z::Int, tmax::Int, root1::Bool)

Generate a Bethe lattice (tree) with a specified degree and depth.

# Arguments
- `z::Int`: The degree of the Bethe lattice.
- `tmax::Int`: The maximum depth of the Bethe lattice.

# Returns
- `V::Vector{Int}`: A list of vertices in the Bethe lattice.
- `E::Vector{Vector{Int}}`: A list of edges in the Bethe lattice.

# Description
This function generates a Bethe lattice (tree) with a specified degree (`z`) and maximum depth (`tmax`). The Bethe lattice is constructed by iteratively adding nodes and edges according to the specified parameters.

If `root1` is `true`, the root/center of the tree is vertex 1. Otherwise, the tree is constructed starting from vertex 0.

The function returns a tuple where the first element (`V`) is a list of vertices, and the second element (`E`) is a list of edges in the Bethe lattice.
"""
function bethe_lattice(z::Int, tmax::Int)
    @assert z > 1 "Degree z must be greater than 1."
    @assert tmax >= 1 "Maximum depth tmax must be at least 1."
    
    # Initialize vertex and edge lists
    V = [1]  # Start with root node as vertex 1
    E = Vector{Vector{Int}}()
    
    # Helper function to recursively add vertices and edges
    function add_children(node, depth, next_vertex)
        if depth > tmax
            return next_vertex  # Stop if max depth is reached
        end
        children_count = z - 1  # Number of children each node has, except root
        if node == 1 children_count = z end  # Root has z children

        for i in 1:children_count
            next_vertex += 1  # Increment to get new vertex
            push!(V, next_vertex)
            push!(E, [node, next_vertex])
            # Recurse to add children of this new node
            next_vertex = add_children(next_vertex, depth + 1, next_vertex)
        end
        return next_vertex
    end
    
    # Start recursion from the root node at depth 1
    add_children(1, 1, 1)
    
    return V, E
end



"""
    ROC_curve(marg::Vector{Float64}, x::Vector{TI}) where {TI<:Integer}

Compute the Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC) based on inferred marginal probabilities and true configurations.

This function computes the ROC curve by sorting the marginal probabilities (`marg`) in decreasing order and sorting the true configuration (`x`) correspondingly. It then iterates over the sorted values to calculate the True Positive Rate (TPR) and False Positive Rate (FPR) at different thresholds, and computes the AUC.

# Arguments
- `marg`: A vector containing the marginal probabilities of each node being infected at a fixed time.
- `x`: The true configuration of the nodes at the same fixed time.

# Returns
- `fp_rates`: Array of false positive rates corresponding to each threshold.
- `tp_rates`: Array of true positive rates corresponding to each threshold.
- `auc`: The Area Under the Curve (AUC) of the ROC curve.
"""
function ROC_curve(marg::Vector{Float64}, x::Vector{TI}) where {TI<:Integer}
    # Sort marg in decreasing order and reorder x correspondingly
    sorted_indices = sortperm(marg, rev=true)
    sorted_marg = marg[sorted_indices]
    sorted_x = x[sorted_indices]

    # Initialize arrays to store true positive rates (TPR) and false positive rates (FPR)
    tp_rates = Float64[]
    fp_rates = Float64[]

    # Initialize counts for true positives (TP) and false positives (FP)
    TP = 0
    FP = 0

    # Total number of positive instances (infected nodes)
    total_positive_instances = sum(x)

    # Total number of negative instances (non-infected nodes)
    total_negative_instances = length(x) - total_positive_instances

    # Initialize AUC
    auc = 0.0

    # Iterate over sorted marg and x to compute TPR, FPR, and AUC
    for i in 1:length(sorted_marg)
        # Update TP and FP counts
        if sorted_x[i] == 1
            TP += 1
        else
            FP += 1
            auc += TP
        end

        # Compute TPR and FPR
        TPR = TP / total_positive_instances
        FPR = FP / total_negative_instances

        # Store TPR and FPR
        push!(tp_rates, TPR)
        push!(fp_rates, FPR)
    end

    # Compute AUC
    auc /= total_positive_instances * total_negative_instances

    return fp_rates, tp_rates, auc
end


function check_mess(m::Float64, μ::Float64, norm::Float64, t::Int)
    if !isfinite(m) || !isfinite(μ)
        println("t = $t: m = $m, μ = $μ, norm = $norm")
        throw(DomainError("NaN evaluated when updating message!"))
    end
end


function check_ρ(inode::Node{TI,TG}, ρ::FBm, M::Array{Float64,3}, t::Int, T::Int) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}
    if !isfinite(ρ.fwm[1,t+1]) || !isfinite(ρ.fwm[2,t+1]) || !isfinite(ρ.bwm[1,T+1-t]) || !isfinite(ρ.bwm[2,T+1-t]) 
        throw(DomainError("NaN evaluated when computing ρ!"))
    end

    #=
    if ρ.fwm[:,t+1]==[0.0,0.0] || ρ.bwm[:,T+1-t]==[0.0,0.0]
        throw(DomainError("0.0 evaluated when computing ρ!"))
    end
    =#
end
