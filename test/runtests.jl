using Aqua
using SampleStats
using Statistics
using StatsBase
using Test
using TypeUtils

@testset "SampleStats" begin
    @testset "Code quality (Aqua)" begin
        Aqua.test_all(SampleStats)
    end
    @testset "Sample statistics  " begin
        # Generate data.
        T = Float32 # single precision
        T² = typeof(zero(T)^2)
        Tp = Float64 # double precision, so that conversion T -> Tp is exact
        Tp² = typeof(zero(Tp)^2)
        x = rand(T, 10_000) .- T(1//3) # offset is to avoid "centered" variables
        mean_x = mean(x)
        var_x = var(x; corrected=true)
        var_x_biased = var(x; corrected=false)

        # Empty sample statistics.
        e0 = @inferred(SampleCount{T}())
        @test typeof(e0) === SampleStat{0,T,Tuple{}}
        @test order(e0) === 0
        @test count(e0) == 0
        @test isempty(e0)
        @test @inferred(SampleStat{0,T}()) === e0
        @test @inferred(empty(SampleStat{0,T})) === e0
        @test @inferred(typeof(e0)()) === e0
        @test @inferred(empty(e0)) === e0
        @test @inferred(empty(typeof(e0))) === e0
        #
        e1 = @inferred(SampleMean{T}())
        @test typeof(e1) === SampleStat{1,T,Tuple{T}}
        @test order(e1) === 1
        @test count(e1) == 0
        @test isempty(e1)
        @test @inferred(SampleStat{1,T}()) === e1
        @test @inferred(empty(SampleStat{1,T})) === e1
        @test @inferred(typeof(e1)()) === e1
        @test @inferred(empty(e1)) === e1
        @test @inferred(empty(typeof(e1))) === e1
        #
        e2 = @inferred(SampleVariance{T}())
        @test typeof(e2) === SampleStat{2,T,Tuple{T,T²}}
        @test order(e2) === 2
        @test count(e2) == 0
        @test isempty(e2)
        @test @inferred(SampleVariance{T}()) === e2
        @test @inferred(empty(SampleVariance{T})) === e2
        @test @inferred(SampleStat{2,T}()) === e2
        @test @inferred(empty(SampleStat{2,T})) === e2
        @test @inferred(typeof(e2)()) === e2
        @test @inferred(empty(e2)) === e2
        @test @inferred(empty(typeof(e2))) === e2

        # Build sample statistics by reduction.
        s0 = @inferred(reduce(SampleCount, x))
        @test typeof(s0) === SampleStat{0,T,Tuple{}}
        @test @inferred(order(s0)) === 0
        @test @inferred(count(s0)) == length(x)
        @test @inferred(nobs(s0)) == count(s0)
        @test !isempty(s0)
        @test @inferred(empty(s0)) === e0
        s1 = @inferred(reduce(SampleMean, x))
        @test typeof(s1) === SampleStat{1,T,Tuple{T}}
        @test @inferred(order(s1)) === 1
        @test @inferred(count(s1)) == length(x)
        @test @inferred(nobs(s1)) == count(s1)
        @test !isempty(s1)
        @test @inferred(empty(s1)) === e1
        @test mean(s1) ≈ mean_x
        @test @inferred(moment(s1, 1)) == moments(s1)[1]
        s2 = @inferred(reduce(SampleVariance, x))
        @test typeof(s2) === SampleStat{2,T,Tuple{T,T²}}
        @test @inferred(order(s2)) === 2
        @test @inferred(count(s2)) == length(x)
        @test @inferred(nobs(s2)) == count(s2)
        @test !isempty(s2)
        @test @inferred(empty(s2)) === e2
        @test mean(s2) ≈ mean_x
        @test var(s2) ≈ var_x
        @test var(s2; corrected=true) ≈ var_x
        @test var(s2; corrected=false) ≈ var_x_biased
        @test @inferred(moment(s2, 1)) == moments(s2)[1]
        @test @inferred(moment(s2, 2)) == moments(s2)[2]

        # Show.
        s = sprint((io, x) -> show(io, x), s0)
        @test startswith(s, "SampleStat{0,")
        s = sprint((io, x) -> show(io, MIME"text/plain"(), x), s0)
        @test startswith(s, "SampleCount{")
        s = sprint((io, x) -> show(io, x), s1)
        @test startswith(s, "SampleStat{1,")
        s = sprint((io, x) -> show(io, MIME"text/plain"(), x), s1)
        @test startswith(s, "SampleMean{")
        s = sprint((io, x) -> show(io, x), s2)
        @test startswith(s, "SampleStat{2,")
        s = sprint((io, x) -> show(io, MIME"text/plain"(), x), s2)
        @test startswith(s, "SampleVariance{")

        # Comparison.
        s0r = @inferred(SampleStat{0,T}(length(x), ()))
        s1r = @inferred(SampleStat{1,T}(length(x), (mean_x,)))
        s2r = @inferred(SampleStat{2,T}(length(x), (mean_x, var_x_biased)))
        @test s0 == s0r
        @test s1 == s1 # NOTE due to rounding errors, s1r = s1 may no hold
        @test s2 == s2 # NOTE due to rounding errors, s2r = s2 may no hold
        @test s0 == s0r
        @test s0 ≈ s0r
        @test s1 ≈ s1r
        @test s2 ≈ s2r

        # Conversions.
        #
        # - change observation type:
        s0p = @inferred(SampleCount{Tp}(count(s0), moments(s0)))
        @test order(s0p) === 0
        @test obstype(s0p) === Tp
        @test count(s0p) == count(s0)
        @test moments(s0p) === adapt_precision(Tp, moments(s0))
        @test s0p !== s0
        @test s0p == s0
        @test s0p ≈ s0
        s1p = @inferred(SampleMean{Tp}(count(s1), moments(s1)))
        @test order(s1p) === 1
        @test obstype(s1p) === Tp
        @test count(s1p) == count(s1)
        @test moments(s1p) === adapt_precision(Tp, moments(s1))
        @test s1p !== s1
        @test s1p == s1
        @test s1p ≈ s1
        s2p = @inferred(SampleVariance{Tp}(count(s2), moments(s2)))
        @test order(s2p) === 2
        @test obstype(s2p) === Tp
        @test count(s2p) == count(s2)
        @test moments(s2p) === adapt_precision(Tp, moments(s2))
        @test s2p !== s2
        @test s2p == s2
        @test s2p ≈ s2
        #
        # - SampleStat(x)
        @test @inferred(SampleStat(s0)) === s0
        @test @inferred(SampleStat(s1)) === s1
        @test @inferred(SampleStat(s2)) === s2
        #
        @test @inferred(convert(SampleStat, s0)) === s0
        @test @inferred(convert(SampleStat, s1)) === s1
        @test @inferred(convert(SampleStat, s2)) === s2
        #
        # - SampleCount(x)
        @test @inferred(SampleCount(s0)) === s0
        @test @inferred(SampleCount(s1)) === s0
        @test @inferred(SampleCount(s2)) === s0
        @test @inferred(SampleCount{T}(s0)) === s0
        @test @inferred(SampleCount{T}(s1)) === s0
        @test @inferred(SampleCount{T}(s2)) === s0
        @test @inferred(SampleCount{Tp}(s0)) === s0p
        @test @inferred(SampleCount{Tp}(s1)) === s0p
        @test @inferred(SampleCount{Tp}(s2)) === s0p
        #
        @test @inferred(convert(SampleCount, s0)) === s0
        @test @inferred(convert(SampleCount, s1)) === s0
        @test @inferred(convert(SampleCount, s2)) === s0
        @test @inferred(convert(SampleCount{T}, s0)) === s0
        @test @inferred(convert(SampleCount{T}, s1)) === s0
        @test @inferred(convert(SampleCount{T}, s2)) === s0
        @test @inferred(convert(SampleCount{Tp}, s0)) === s0p
        @test @inferred(convert(SampleCount{Tp}, s1)) === s0p
        @test @inferred(convert(SampleCount{Tp}, s2)) === s0p
        #
        # - SampleMean(x)
        @test_throws Exception SampleMean(s0)
        @test @inferred(SampleMean(s1)) === s1
        @test @inferred(SampleMean(s2)) === s1
        @test_throws Exception SampleMean{T}(s0)
        @test @inferred(SampleMean{T}(s1)) === s1
        @test @inferred(SampleMean{T}(s2)) === s1
        @test_throws Exception SampleMean{Tp}(s0)
        @test @inferred(SampleMean{Tp}(s1)) === s1p
        @test @inferred(SampleMean{Tp}(s2)) === s1p
        #
        @test_throws Exception convert(SampleMean, s0)
        @test @inferred(convert(SampleMean, s1)) === s1
        @test @inferred(convert(SampleMean, s2)) === s1
        @test_throws Exception convert(SampleMean{T}, s0)
        @test @inferred(convert(SampleMean{T}, s1)) === s1
        @test @inferred(convert(SampleMean{T}, s2)) === s1
        @test_throws Exception convert(SampleMean{Tp}, s0)
        @test @inferred(convert(SampleMean{Tp}, s1)) === s1p
        @test @inferred(convert(SampleMean{Tp}, s2)) === s1p
        #
        # - SampleVariance(x)
        @test_throws Exception SampleVariance(s0)
        @test_throws Exception SampleVariance(s1)
        @test @inferred(SampleVariance(s2)) === s2
        @test_throws Exception SampleVariance{T}(s0)
        @test_throws Exception SampleVariance{T}(s1)
        @test @inferred(SampleVariance{T}(s2)) === s2
        @test_throws Exception SampleVariance{Tp}(s0)
        @test_throws Exception SampleVariance{Tp}(s1)
        @test @inferred(SampleVariance{Tp}(s2)) === s2p
        #
        @test_throws Exception convert(SampleVariance, s0)
        @test_throws Exception convert(SampleVariance, s1)
        @test @inferred(convert(SampleVariance, s2)) === s2
        @test_throws Exception convert(SampleVariance{T}, s0)
        @test_throws Exception convert(SampleVariance{T}, s1)
        @test @inferred(convert(SampleVariance{T}, s2)) === s2
        @test_throws Exception convert(SampleVariance{Tp}, s0)
        @test_throws Exception convert(SampleVariance{Tp}, s1)
        @test @inferred(convert(SampleVariance{Tp}, s2)) === s2p
        #
        # - SampleStat{0}(x)
        @test @inferred(SampleStat{0}(s0)) === s0
        @test @inferred(SampleStat{0}(s1)) === s0
        @test @inferred(SampleStat{0}(s2)) === s0
        @test @inferred(SampleStat{0,T}(s0)) === s0
        @test @inferred(SampleStat{0,T}(s1)) === s0
        @test @inferred(SampleStat{0,T}(s2)) === s0
        @test @inferred(SampleStat{0,T,Tuple{}}(s0)) === s0
        @test @inferred(SampleStat{0,T,Tuple{}}(s1)) === s0
        @test @inferred(SampleStat{0,T,Tuple{}}(s2)) === s0
        @test @inferred(SampleStat{0,Tp}(s0)) === s0p
        @test @inferred(SampleStat{0,Tp}(s1)) === s0p
        @test @inferred(SampleStat{0,Tp}(s2)) === s0p
        @test @inferred(SampleStat{0,Tp,Tuple{}}(s0)) === s0p
        @test @inferred(SampleStat{0,Tp,Tuple{}}(s1)) === s0p
        @test @inferred(SampleStat{0,Tp,Tuple{}}(s2)) === s0p
        #
        @test @inferred(convert(SampleStat{0}, s0)) === s0
        @test @inferred(convert(SampleStat{0}, s1)) === s0
        @test @inferred(convert(SampleStat{0}, s2)) === s0
        @test @inferred(convert(SampleStat{0,T}, s0)) === s0
        @test @inferred(convert(SampleStat{0,T}, s1)) === s0
        @test @inferred(convert(SampleStat{0,T}, s2)) === s0
        @test @inferred(convert(SampleStat{0,T,Tuple{}}, s0)) === s0
        @test @inferred(convert(SampleStat{0,T,Tuple{}}, s1)) === s0
        @test @inferred(convert(SampleStat{0,T,Tuple{}}, s2)) === s0
        @test @inferred(convert(SampleStat{0,Tp}, s0)) === s0p
        @test @inferred(convert(SampleStat{0,Tp}, s1)) === s0p
        @test @inferred(convert(SampleStat{0,Tp}, s2)) === s0p
        @test @inferred(convert(SampleStat{0,Tp,Tuple{}}, s0)) === s0p
        @test @inferred(convert(SampleStat{0,Tp,Tuple{}}, s1)) === s0p
        @test @inferred(convert(SampleStat{0,Tp,Tuple{}}, s2)) === s0p
        #
        # - SampleStat{1}(x)
        @test_throws Exception SampleStat{1}(s0)
        @test @inferred(SampleStat{1}(s1)) === s1
        @test @inferred(SampleStat{1}(s2)) === s1
        @test_throws Exception SampleStat{1,T}(s0)
        @test @inferred(SampleStat{1,T}(s1)) === s1
        @test @inferred(SampleStat{1,T}(s2)) === s1
        @test_throws Exception SampleStat{1,T,Tuple{T}}(s0)
        @test @inferred(SampleStat{1,T,Tuple{T}}(s1)) === s1
        @test @inferred(SampleStat{1,T,Tuple{T}}(s2)) === s1
        @test_throws Exception SampleStat{1,Tp}(s0)
        @test @inferred(SampleStat{1,Tp}(s1)) === s1p
        @test @inferred(SampleStat{1,Tp}(s2)) === s1p
        @test_throws Exception SampleStat{1,Tp,Tuple{Tp}}(s0)
        @test @inferred(SampleStat{1,Tp,Tuple{Tp}}(s1)) === s1p
        @test @inferred(SampleStat{1,Tp,Tuple{Tp}}(s2)) === s1p
        #
        @test_throws Exception convert(SampleStat{1}, s0)
        @test @inferred(convert(SampleStat{1}, s1)) === s1
        @test @inferred(convert(SampleStat{1}, s2)) === s1
        @test_throws Exception convert(SampleStat{1,T}, s0)
        @test @inferred(convert(SampleStat{1,T}, s1)) === s1
        @test @inferred(convert(SampleStat{1,T}, s2)) === s1
        @test_throws Exception convert(SampleStat{1,T,Tuple{T}}, s0)
        @test @inferred(convert(SampleStat{1,T,Tuple{T}}, s1)) === s1
        @test @inferred(convert(SampleStat{1,T,Tuple{T}}, s2)) === s1
        @test_throws Exception convert(SampleStat{1,Tp}, s0)
        @test @inferred(convert(SampleStat{1,Tp}, s1)) === s1p
        @test @inferred(convert(SampleStat{1,Tp}, s2)) === s1p
        @test_throws Exception convert(SampleStat{1,Tp,Tuple{Tp}}, s0)
        @test @inferred(convert(SampleStat{1,Tp,Tuple{Tp}}, s1)) === s1p
        @test @inferred(convert(SampleStat{1,Tp,Tuple{Tp}}, s2)) === s1p
        #
        # - SampleStat{2}(x)
        @test_throws Exception SampleStat{2}(s0)
        @test_throws Exception SampleStat{2}(s1)
        @test @inferred(SampleStat{2}(s2)) === s2
        @test_throws Exception SampleStat{2,T}(s0)
        @test_throws Exception SampleStat{2,T}(s1)
        @test @inferred(SampleStat{2,T}(s2)) === s2
        @test_throws Exception SampleStat{2,T,Tuple{T,T²}}(s0)
        @test_throws Exception SampleStat{2,T,Tuple{T,T²}}(s1)
        @test @inferred(SampleStat{2,T,Tuple{T,T²}}(s2)) === s2
        @test_throws Exception SampleStat{2,Tp}(s0)
        @test_throws Exception SampleStat{2,Tp}(s1)
        @test @inferred(SampleStat{2,Tp}(s2)) === s2p
        @test_throws Exception SampleStat{2,Tp,Tuple{Tp,Tp²}}(s0)
        @test_throws Exception SampleStat{2,Tp,Tuple{Tp,Tp²}}(s1)
        @test @inferred(SampleStat{2,Tp,Tuple{Tp,Tp²}}(s2)) === s2p
        #
        @test_throws Exception convert(SampleStat{2}, s0)
        @test_throws Exception convert(SampleStat{2}, s1)
        @test @inferred(convert(SampleStat{2}, s2)) === s2
        @test_throws Exception convert(SampleStat{2,T}, s0)
        @test_throws Exception convert(SampleStat{2,T}, s1)
        @test @inferred(convert(SampleStat{2,T}, s2)) === s2
        @test_throws Exception convert(SampleStat{2,T,Tuple{T,T²}}, s0)
        @test_throws Exception convert(SampleStat{2,T,Tuple{T,T²}}, s1)
        @test @inferred(convert(SampleStat{2,T,Tuple{T,T²}}, s2)) === s2
        @test_throws Exception convert(SampleStat{2,Tp}, s0)
        @test_throws Exception convert(SampleStat{2,Tp}, s1)
        @test @inferred(convert(SampleStat{2,Tp}, s2)) === s2p
        @test_throws Exception convert(SampleStat{2,Tp,Tuple{Tp,Tp²}}, s0)
        @test_throws Exception convert(SampleStat{2,Tp,Tuple{Tp,Tp²}}, s1)
        @test @inferred(convert(SampleStat{2,Tp,Tuple{Tp,Tp²}}, s2)) === s2p

        # Common errors.
        # TODO @test_throws AssertionError empty(SampleStat{0}) # missing type T
        # TODO @test_throws AssertionError empty(SampleStat{0x00,T}) # invalid type for M
        # TODO @test_throws AssertionError empty(SampleStat{-1,T}) # invalid value for M
        @test_throws AssertionError empty(SampleStat{1,Int}) # T is not floating-point

        # Merge statistics.
        xa = view(x, 1:div(length(x),3))
        na = length(xa)
        xb = view(x, na+1:length(x))
        nb = length(xb)
        s0a = @inferred(reduce(SampleCount, xa))
        @test count(s0a) == na
        s0b = @inferred(reduce(SampleCount, xb))
        @test count(s0b) == nb
        s0ab = @inferred(merge(s0a, s0b))
        @test count(s0ab) == count(s0)
        @test s0ab ≈ s0
        s1a = @inferred(reduce(SampleMean, xa))
        @test count(s1a) == na
        s1b = @inferred(reduce(SampleMean, xb))
        @test count(s1b) == nb
        s1ab = @inferred(merge(s1a, s1b))
        @test count(s1ab) == count(s1)
        @test s1ab ≈ s1
        s2a = @inferred(reduce(SampleVariance, xa))
        @test count(s2a) == na
        s2b = @inferred(reduce(SampleVariance, xb))
        @test count(s2b) == nb
        s2ab = @inferred(merge(s2a, s2b))
        @test count(s2ab) == count(s2)
        @test s2ab ≈ s2

        # TypeUtils methods.
        #
        @test @inferred(get_precision(s0)) === get_precision(T)
        @test @inferred(get_precision(s1)) === get_precision(T)
        @test @inferred(get_precision(s2)) === get_precision(T)
        @test @inferred(get_precision(s0p)) === get_precision(Tp)
        @test @inferred(get_precision(s1p)) === get_precision(Tp)
        @test @inferred(get_precision(s2p)) === get_precision(Tp)
        #
        @test @inferred(adapt_precision(Tp, s0)) === s0p
        @test @inferred(adapt_precision(Tp, s1)) === s1p
        @test @inferred(adapt_precision(Tp, s2)) === s2p
        @test @inferred(adapt_precision(T, s0p)) === s0
        @test typeof(adapt_precision(T, s0p)) === typeof(s0)
        @test @inferred(adapt_precision(T, s1p)) ≈ s1
        @test typeof(adapt_precision(T, s1p)) === typeof(s1)
        @test @inferred(adapt_precision(T, s2p)) ≈ s2
        @test typeof(adapt_precision(T, s2p)) === typeof(s2)
    end
end
