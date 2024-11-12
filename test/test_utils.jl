V = collect(1:10)
E = [[1, 2], [2, 3], [2, 4], [1, 5], [5, 6], [5, 7], [1, 8], [8, 9], [8, 10]]

@testset "bethe_lattice" begin
    @test bethe_lattice(3, 2) == (V, E)
end

@testset "ROC_curve" begin
    @test ROC_curve([0.1,0.5,0.3,0.7], [1,0,0,1]) == ([0.0,0.5,1.0,1.0], [0.5,0.5, 0.5, 1.0], 0.5)
end