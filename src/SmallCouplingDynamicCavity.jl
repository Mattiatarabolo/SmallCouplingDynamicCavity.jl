"""
Module for "SmallCouplingDynamicCavity.jl" -- A package for a Small Coupling expansion of the Dynamic Cavity method for epidemic inference.

# Exports

$(EXPORTS)

"""
module SmallCouplingDynamicCavity

    using Graphs, Distributions, DocStringExtensions, Random

    export sim_epidemics, run_SCDC, ROC_curve, bethe_lattice, SI, SIR, SIS, SIRS, EpidemicModel

    abstract type InfectionModel
        # the infection probability is included into the temporal graph
    end

    include("types.jl")
    include("utils.jl")

    include("models/SI.jl")
    include("models/SIR.jl")
    include("models/SIS.jl")
    include("models/SIRS.jl")

    include("message_passing_func.jl")

end