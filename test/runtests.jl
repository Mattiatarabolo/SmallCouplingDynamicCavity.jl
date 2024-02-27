using SmallCouplingDynamicCavity
using Graphs
using Random
using JLD2
using Test
using Aqua

@testset "Aqua" begin
    Aqua.test_all(SmallCouplingDynamicCavity, ambiguities=false)
    Aqua.test_ambiguities(SmallCouplingDynamicCavity)
end

include("testSI.jl")
include("testSIR.jl")
include("testSIS.jl")
include("testSIRS.jl")
include("testSI_timevarying.jl")
include("testSIR_timevarying.jl")
include("testSIS_timevarying.jl")
include("testSIRS_timevarying.jl")

#nothing