using SampleStats
using Test
using Aqua

@testset "SampleStats.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(SampleStats)
    end
    # Write your tests here.
end
