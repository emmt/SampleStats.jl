using Aqua
using SampleStats
using Statistics
using StatsBase
using Test
using TypeUtils

@testset "SampleStats" begin
    str(u::TypeUtils.NoUnits) = "none"
    str(u::Any) = string(u)
    brief(::Type{SampleStat{M,T}}) where {M,T} =
        "order: $M, precision: $(get_precision(T)), units: $(str(units_of(T)))"
    @testset "Code quality (Aqua)" begin
        Aqua.test_all(SampleStats)
    end
    @testset "Sample statistics ($(brief(SampleStat{M,T})))" for (M,T) in (
        (0, Float32), (1, Float32), (2, Float32), (3, Float32), (4, Float32),
        )

        # Generate data.
        Tp = adapt_precision(get_precision(T) != Float32 ? Float32 : Float64, T)
        n = 123
        V = typeof(ntuple(k -> zero(T)^k, Val(M)))
        Vp = typeof(ntuple(k -> zero(Tp)^k, Val(M)))
        Tp = adapt_precision(Float64, T) # double precision, so that conversion T -> Tp is exact
        Tp² = typeof(zero(Tp)^2)
        Tp³ = typeof(zero(Tp)^3)
        x = rand(T, n) .- T(1//3) # offset is to avoid "centered" variables

        # Empty statistics.
        ms = ntuple(k -> zero(T)^k, Val(M))
        e = @inferred(SampleStat{M,T}())
        @test typeof(e) === SampleStat{M,T,V}
        @test M === @inferred(order(e))
        @test T === @inferred(obstype(e))
        @test 0 === @inferred(count(e))
        @test 0 === @inferred(nobs(e))
        @test e === @inferred(          typeof(e)())
        @test_throws ErrorException SampleStat{M}()
        @test_throws ErrorException SampleStat(   )
        @test isempty(e)
        @test e === @inferred(      empty(e))
        @test e === @inferred(      empty(typeof(e)))
        @test e === @inferred(      empty(SampleStat{M,T}))
        @test_throws ErrorException empty(SampleStat{M})
        @test_throws ErrorException empty(SampleStat)
        @test ms === @inferred(moments(e))
        @test typeof(ms) === V
        @test e === @inferred(                typeof(e)(0, ms))
        @test e === @inferred(          SampleStat{M,T}(0, ms))
        if M == 0
            @test_throws ErrorException SampleStat{M}(  0, ms)
            @test_throws ErrorException SampleStat(     0, ms)
        else
            @test e === @inferred(      SampleStat{M}(  0, ms))
            @test e === @inferred(      SampleStat(     0, ms))
        end
        if M == 0
            @test e === @inferred(            SampleCount{T}())
            @test_throws ErrorException       SampleCount(   )
            @test e === @inferred(            SampleCount{T}(0, ms))
            @test_throws ErrorException       SampleCount(   0, ms)
            @test e === @inferred(      empty(SampleCount{T}))
            @test_throws ErrorException empty(SampleCount)
        elseif M == 1
            @test e === @inferred(            SampleMean{T}())
            @test_throws ErrorException       SampleMean(   )
            @test e === @inferred(            SampleMean{T}(0, ms))
            @test e === @inferred(            SampleMean(   0, ms))
            @test e === @inferred(      empty(SampleMean{T}))
            @test_throws ErrorException empty(SampleMean)
        elseif M == 2
            @test e === @inferred(            SampleVariance{T}())
            @test_throws ErrorException       SampleVariance(   )
            @test e === @inferred(            SampleVariance{T}(0, ms))
            @test e === @inferred(            SampleVariance(   0, ms))
            @test e === @inferred(      empty(SampleVariance{T}))
            @test_throws ErrorException empty(SampleVariance)
        end

        # Initialize sample statistics with a unique number.
        x1 = first(x)
        ms = ntuple(k -> k == 1 ? convert(T, x1) : zero(T)^k, Val(M))
        #
        u = @inferred(SampleStat{M,T}(x1))
        @test typeof(u) === SampleStat{M,T,V}
        @test M === @inferred(order(u))
        @test T === @inferred(obstype(u))
        @test 1 === @inferred(count(u))
        @test 1 === @inferred(nobs(u))
        @test u === @inferred(          typeof(u)(x1))
        @test u === @inferred(      SampleStat{M}(x1))
        @test_throws ErrorException SampleStat(   x1)
        @test !isempty(u)
        @test ms === @inferred(moments(u))
        @test typeof(ms) === V
        @test u === @inferred(                typeof(u)(1, ms))
        @test u === @inferred(          SampleStat{M,T}(1, ms))
        if M == 0
            @test_throws ErrorException SampleStat{M}(  1, ms)
            @test_throws ErrorException SampleStat(     1, ms)
        else
            @test u === @inferred(      SampleStat{M}(  1, ms))
            @test u === @inferred(      SampleStat(     1, ms))
        end
        if M == 0
            @test u === @inferred(      SampleCount{T}(x1))
            @test u === @inferred(      SampleCount(   x1))
            @test u === @inferred(      SampleCount{T}(1, ms))
            @test_throws ErrorException SampleCount(   1, ms)
        elseif M == 1
            @test u === @inferred(      SampleMean(   x1))
            @test u === @inferred(      SampleMean{T}(x1))
            @test u === @inferred(      SampleMean(   1, ms))
            @test u === @inferred(      SampleMean{T}(1, ms))
        elseif M == 2
            @test u === @inferred(      SampleVariance(   x1))
            @test u === @inferred(      SampleVariance{T}(x1))
            @test u === @inferred(      SampleVariance(   1, ms))
            @test u === @inferred(      SampleVariance{T}(1, ms))
        end

        # Compute sample statistics from iterable object.
        moments_x = ()
        mean_x = zero(T)
        var_x = zero(T)
        var_x_biased = zero(T)
        if M ≥ 1
            μ = sum(x)/n # sample mean
            xc = x .- μ # centered observations
            moments_x = ntuple(k -> k == 1 ? μ : sum(xc.^k)/n, Val(M))
            mean_x = mean(x)
            @test moments_x[1] ≈ mean_x
            if M ≥ 2
                var_x = var(x; corrected=true)
                var_x_biased = var(x; corrected=false)
                @test moments_x[2] ≈ var_x_biased
            end
        end
        @test typeof(moments_x) === V
        #
        s = @inferred(SampleStat{M,T}(x))
        @test typeof(s) === SampleStat{M,T,V}
        @test M === @inferred(order(s))
        @test T === @inferred(obstype(s))
        @test n === @inferred(count(s))
        @test n === @inferred(nobs(s))
        @test s === @inferred(          typeof(s)(x))
        @test s === @inferred(      SampleStat{M}(x))
        @test_throws ErrorException SampleStat(   x)
        @test !isempty(s)
        ms = @inferred(moments(s))
        @test ms === ntuple(k -> s[k], Val(M))
        @test ms === ntuple(k -> moment(s, k), Val(M))
        @test typeof(ms) === V
        if M == 0
            @test ms === moments_x
        else
            for k in 1:M
                @test ms[k] ≈ moments_x[k]
            end
        end
        @test s === @inferred(                typeof(s)(n, ms))
        @test s === @inferred(          SampleStat{M,T}(n, ms))
        if M == 0
            @test_throws ErrorException SampleStat{M}(  n, ms)
            @test_throws ErrorException SampleStat(     n, ms)
        else
            @test s === @inferred(      SampleStat{M}(  n, ms))
            @test s === @inferred(      SampleStat(     n, ms))
        end
        if M == 0
            @test s === @inferred(      SampleCount(   x))
            @test s === @inferred(      SampleCount{T}(x))
            @test s === @inferred(      SampleCount{T}(n, ms))
            @test_throws ErrorException SampleCount(   n, ms)
        elseif M == 1
            @test s === @inferred(      SampleMean(   x))
            @test s === @inferred(      SampleMean{T}(x))
            @test s === @inferred(      SampleMean{T}(n, ms))
            @test s === @inferred(      SampleMean(   n, ms))
        elseif M == 2
            @test s === @inferred(      SampleVariance(   x))
            @test s === @inferred(      SampleVariance{T}(x))
            @test s === @inferred(      SampleVariance(   n, ms))
            @test s === @inferred(      SampleVariance{T}(n, ms))
        end
        #
        @test e === @inferred(SampleStat{M}(T[]))
        @test u === @inferred(SampleStat{M}([x1]))

        # Compare computed moments with those from `Statistics` or `StatsBase`.
        if M ≥ 1
            @test @inferred(mean(s)) ≈ mean_x
        end
        if M ≥ 2
            @test @inferred(var(s)) ≈ var_x
            @test @inferred(var(s; corrected=true)) ≈ var_x
            @test @inferred(var(s; corrected=false)) ≈ var_x_biased
        end

        # Call `reduce` to compute statistics.
        @test s === @inferred(      reduce(typeof(s),     x))
        @test s === @inferred(      reduce(SampleStat{M}, x))
        @test_throws ErrorException reduce(SampleStat,    x)
        if M == 0
            @test s === @inferred(  reduce(SampleCount,    x))
            @test s === @inferred(  reduce(SampleCount{T}, x))
        elseif M == 1
            @test s === @inferred(  reduce(SampleMean,    x))
            @test s === @inferred(  reduce(SampleMean{T}, x))
        elseif M == 2
            @test s === @inferred(  reduce(SampleVariance,    x))
            @test s === @inferred(  reduce(SampleVariance{T}, x))
        end

        # Show.
        q = sprint((io, x) -> show(io, x), s)
        @test startswith(q, "SampleStat{$M,")
        q = sprint((io, x) -> show(io, MIME"text/plain"(), x), s)
        if M == 0
            @test startswith(q, "SampleCount{")
        elseif M == 1
            @test startswith(q, "SampleMean{")
        elseif M == 2
            @test startswith(q, "SampleVariance{")
        else
            @test startswith(q, "SampleStat{$M,")
        end

        # Change precision.
        sp = @inferred(SampleStat{M,Tp}(count(s), moments(s)))
        @test @inferred(order(sp)) === M
        @test @inferred(order(typeof(sp))) === M
        @test @inferred(obstype(sp)) === Tp
        @test @inferred(obstype(typeof(sp))) === Tp
        @test @inferred(count(sp)) === n
        @test @inferred(moments(sp)) === adapt_precision(Tp, moments(s))
        @test sp !== s

        # Comparison.
        sr = @inferred(SampleStat{M,T}(length(x), moments_x))
        @test typeof(sr) === typeof(s)
        if M == 0
            # For M = 0, there are no possible rounding errors; hence, compare exactly.
            @test s === sr
            @test s == sr
            @test s ≈ sr
            @test s ≈ sp rtol=1e-5
        else
            # For M > 0, due to rounding errors, only compare approximately.
            @test s == s
            @test s ≈ sr
            @test s ≈ sp rtol=1e-5
        end

        # Conversions.
        #
        # - change nothing:
        @test s === @inferred(        typeof(s)(       s))
        @test s === @inferred(convert(typeof(s),       s))
        @test s === @inferred(        SampleStat{M,T}( s))
        @test s === @inferred(convert(SampleStat{M,T}, s))
        @test s === @inferred(        SampleStat{M}(   s))
        @test s === @inferred(convert(SampleStat{M},   s))
        @test s === @inferred(        SampleStat(      s))
        @test s === @inferred(convert(SampleStat,      s))
        if M == 0
            @test s === @inferred(        SampleCount{T}( s))
            @test s === @inferred(convert(SampleCount{T}, s))
            @test s === @inferred(        SampleCount(    s))
            @test s === @inferred(convert(SampleCount,    s))
        elseif M == 1
            @test s === @inferred(        SampleMean{T}( s))
            @test s === @inferred(convert(SampleMean{T}, s))
            @test s === @inferred(        SampleMean(    s))
            @test s === @inferred(convert(SampleMean,    s))
        elseif M == 2
            @test s === @inferred(        SampleVariance{T}( s))
            @test s === @inferred(convert(SampleVariance{T}, s))
            @test s === @inferred(        SampleVariance(    s))
            @test s === @inferred(convert(SampleVariance,    s))
        end
        #
        # - change observation type:
        @test sp === @inferred(        typeof(sp)(       s))
        @test sp === @inferred(convert(typeof(sp),       s))
        @test sp === @inferred(        SampleStat{M,Tp}( s))
        @test sp === @inferred(convert(SampleStat{M,Tp}, s))
        if M == 0
            @test sp === @inferred(        SampleCount{Tp}( s))
            @test sp === @inferred(convert(SampleCount{Tp}, s))
        elseif M == 1
            @test sp === @inferred(        SampleMean{Tp}( s))
            @test sp === @inferred(convert(SampleMean{Tp}, s))
        elseif M == 2
            @test sp === @inferred(        SampleVariance{Tp}( s))
            @test sp === @inferred(convert(SampleVariance{Tp}, s))
        end
        #
        # - change statistics order:
        ms = @inferred(moments(s))
        msp = map(a->convert(Tp, a), ms)
        if M ≥ 0
            @test @inferred(convert(SampleCount,     s)) === SampleStat{0,T }(n, ())
            @test @inferred(convert(SampleCount{T},  s)) === SampleStat{0,T }(n, ())
            @test @inferred(convert(SampleCount{Tp}, s)) === SampleStat{0,Tp}(n, ())
        end
        if M ≥ 1
            @test @inferred(convert(SampleMean,     s)) === SampleStat(n, (ms[1],))
            @test @inferred(convert(SampleMean{T},  s)) === SampleStat(n, (ms[1],))
            @test @inferred(convert(SampleMean{Tp}, s)) === SampleStat(n, (msp[1],))
        end
        if M ≥ 2
            @test @inferred(convert(SampleVariance,     s)) === SampleStat(n, (ms[1],ms[2]))
            @test @inferred(convert(SampleVariance{T }, s)) === SampleStat(n, (ms[1],ms[2]))
            @test @inferred(convert(SampleVariance{Tp}, s)) === SampleStat(n, (msp[1],msp[2]))
        end

        # TypeUtils methods.
        #
        @test get_precision(T) === @inferred(get_precision(s))
        @test get_precision(T) === @inferred(get_precision(typeof(s)))
        @test get_precision(Tp) === @inferred(get_precision(sp))
        @test get_precision(Tp) === @inferred(get_precision(typeof(sp)))
        #
        @test sp === @inferred(adapt_precision(Tp, s))

        # Merge statistics.
        if M ≤ 2 # TODO Implement for M > 2.
            xa = view(x, 1:div(length(x),3))
            na = length(xa)
            xb = view(x, na+1:length(x))
            nb = length(xb)
            sa = @inferred(SampleStat{M,T}(xa))
            @test count(sa) == na
            sb = @inferred(SampleStat{M,T}(xb))
            @test count(sb) == nb
            @test @inferred(merge(sa, sb)) ≈ s
            @test @inferred(merge(sa, xb)) ≈ s
            @test @inferred(merge(sb, sa)) ≈ s
            @test @inferred(merge(sb, xa)) ≈ s
            stat = @inferred(typeof(s)())
            for xᵢ in x
                stat = @inferred(merge(stat, xᵢ))
            end
            @test stat ≈ s
        else
            @test_throws ErrorException merge(s, x1)
        end

        # Common errors.
        if M == 0
            @test_throws ErrorException empty(SampleStat{M}) # missing type T
            # TODO @test_throws AssertionError empty(SampleStat{0x00,T}) # invalid type for M
            # TODO @test_throws AssertionError empty(SampleStat{-1,T}) # invalid value for M
        end
        @test_throws AssertionError empty(SampleStat{M,Int}) # T is not floating-point
    end
end
