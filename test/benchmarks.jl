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
    print("+ length(x) ---------------->"); @btime length($x);
    print("+ SampleCount(x) ----------->"); @btime SampleCount($x);
    print("+ mean(x) ------------------>"); @btime mean($x);
    print("+ SampleMean(x) ------------>"); @btime SampleMean($x);
    print("+ var(x; corrected=false) -->"); @btime var($x; corrected=false);
    print("+ SampleVariance(x) -------->"); @btime SampleVariance($x);
    print("+ SampleStat{3}(x) --------->"); @btime SampleStat{3}($x);
    print("+ SampleStat{4}(x) --------->"); @btime SampleStat{4}($x);
    print("+ SampleStat{5}(x) --------->"); @btime SampleStat{5}($x);
    print("+ SampleStat{6}(x) --------->"); @btime SampleStat{6}($x);
    print("+ SampleStat{7}(x) --------->"); @btime SampleStat{7}($x);
    print("+ SampleStat{8}(x) --------->"); @btime SampleStat{8}($x);
    print("+ SampleStat{9}(x) --------->"); @btime SampleStat{9}($x);
    nothing
end

end # module
