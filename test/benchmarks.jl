module BenchmarkingSamplStats

using BenchmarkTools
using SampleStats
using Statistics
using StatsBase
using Test
using TypeUtils

function runtests(;T::Type=Float32, n::Int=10_000)
    x = rand(T, n) .- T(1//4)
    println("Tests for n=$n observations of type T=$T")
    print("+ length(x)                ");  @btime length($x);
    print("+ reduce(SampleCount, x)   ");  @btime reduce(SampleCount, $x);
    print("+ mean(x)                  ");  @btime mean($x);
    print("+ reduce(SampleMean, x)    ");  @btime reduce(SampleMean, $x);
    print("+ var(x; corrected=false)  ");  @btime var($x; corrected=false);
    print("+ reduce(SampleVariance, x)");  @btime reduce(SampleVariance, $x);
    nothing
end

end # module
