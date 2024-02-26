using SmallCouplingDynamicCavity
using Graphs
using Random
using Test
using Aqua

@testset "Aqua" begin
    Aqua.test_all(SmallCouplingDynamicCavity, ambiguities=false)
    Aqua.test_ambiguities(SmallCouplingDynamicCavity)
end

include("simSI.jl")
#=include("simSIR.jl")
include("simSIS.jl")
include("simSIRS.jl")
include("inferenceSI.jl")
include("inferenceSIR.jl")
include("inferenceSISjl")
include("inferenceSIRS.jl")=#

nothing