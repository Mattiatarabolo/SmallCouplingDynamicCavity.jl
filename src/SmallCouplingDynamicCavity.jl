"""
Module for "SmallCouplingDynamicCavity.jl" -- A package for a Small Coupling expansion of the Dynamic Cavity method for epidemic inference.

# Exports

$(EXPORTS)

"""
module SmallCouplingDynamicCavity

    using Graphs, DocStringExtensions, Random, StatsBase

    export sim_epidemics, sim_epidemics!, run_SCDC, run_SCDC!, run_fwd_dynamics, ROC_curve, bethe_lattice, SI, SIR, SIS, SIRS, EpidemicModel

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