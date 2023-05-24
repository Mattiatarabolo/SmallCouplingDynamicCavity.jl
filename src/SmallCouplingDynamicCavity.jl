module SmallCouplingDynamicCavity

using Graphs, Distributions

export sim_epidemics, run_SCDC, run_SCDC_MF, ROC_curve, bethe_lattice

include("types.jl")
include("utils.jl")
include("message_passing_func.jl")

end