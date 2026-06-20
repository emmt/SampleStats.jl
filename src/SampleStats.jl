"""

`SampleStats` provides immutable objects to store sample statistics of independent and
identically distributed (i.i.d.) variables. Sample statistics can be updated to account for
new observations using recursive formulae as in *online statistics*.

Comparison with [`OnlineStats`](https://github.com/joshday/OnlineStats.jl):

- `SampleStat` objects are immutable.

- Computing sample statistics with `SampleStats` requires no allocations and is as fast or
  faster than with `OnlineStats`. However, since, one-pass recursive formulae are used,
  both are slower than calling `mean`, `var`, etc. from `Statistics`.

- Computing moments of order ≥ 2 with `SampleStats` is done with centered observations which
  avoids overflows and insures that moments of even order are non-negative.

- The moments computed with `SampleStats` are not bias corrected (i.e., they are biased).

- `SampleCount{T} = SampleStat{0,T}` is the counterpart of `OnlineStats.Count`,
  `SampleMean{T} = SampleStat{1,T}` is the counterpart of `OnlineStats.mean{T}`, and
  `SampleVariance{T} = SampleStat{2,T}` is the counterpart of `OnlineStats.Variance{T}`.

Comparison with [`OnlineSampleStatistics`](https://github.com/FerreolS/OnlineSampleStatistics.jl):

Comparison with [`MultiVariateOnlineStatistics`](https://github.com/emmt/MultiVariateOnlineStatistics.jl):

"""
module SampleStats

export
    SampleCount,
    SampleMean,
    SampleStat,
    SampleVariance,
    count,
    mean,
    moment,
    moments,
    nobs,
    obstype,
    order,
    std,
    var

# TODO Merge 2 sample statistics objects of order > 2.
# TODO Optimize reduce for 0-, 1-, and 2-order statistics.

using Statistics

using StatsBase

using TypeUtils
using TypeUtils:
    @public,
    Precision

using Base: @propagate_inbounds

import SharedArrays

"""
    SampleStat{M,T,V}

Type of order `M` sample statistics of independent and identically distributed (i.i.d.)
observations. Parameter `T` is the type of a single observation, it may have units but must
be floating-point point. Parameter `V` is the tuple type of the `M` first moments. Directly
calling `SampleStat{M,T,V}` as a constructor is discouraged, other constructors are provided
which take care of having `V` consistent with `M` and `T`.

For an instance `stat` of `SampleStat`, the syntax `stat[k]` yields `μₖ`, the `k`-th
statistical moment stored by `stat`, while `count(stat)` and `nobs(stat)` both yield the
number `n` of observations. For observations `(x₁, …, xₙ)`, the moments are given by:

```
μ₁ = (1/n) Σᵢ xᵢ
μₖ = (1/n) Σᵢ (xᵢ - μ₁)ᵏ    (for k > 1)
```

Hence, the first moment is the sample mean, the second one is the (biased) sample variance,
etc.

# Merging statistics

A sample statistics object can be updated by *merging* with a new observation or a
collection of observations assuming all observations are mutually independent. For example,
if `A` is a sample of i.i.d. observations, the `M` first moments can be computed by:

```julia
stat = SampleStat{M}(eltype(A))
for x in A
    stat = merge(stat, x)
end
```

A *lazy filter* can be used to only merge valid observations:

```julia
stat = SampleStat{M}(eltype(A))
for x in Iterators.filter(isfinite, A)
    stat = merge(stat, x)
end
```

These examples, can be simplified by calling `reduce`:

```julia
stat = reduce(SampleStat{M}, A)
stat = reduce(SampleStat{M}, Iterators.filter(isfinite, A))
```

"""
struct SampleStat{M,T,V<:NTuple{M,Any}}
    count::Int
    moments::V

    # Inner constructor.
    function SampleStat{M,T,V}(count::Integer,
                               moments::NTuple{M,Any}) where {M,T,V<:NTuple{M,Any}}
        return new{M,T,V}(count, moments)
    end
end

"""
    cnt = SampleCount{T}(args...)
    cnt = SampleStat{0,T}(args...)

Return a 0-th order sample statistics object built according to the arguments `args...` and
which only stores the number of observations. `T` is the (floating-point) type of a single
observation and can be omitted if it can be inferred from the arguments.

"""
const SampleCount{T} = SampleStat{0,T}

"""
    avg = SampleMean{T}(args...)
    avg = SampleStat{1,T}(args...)

Return a 1-st order sample statistics object built according to the arguments `args...` and
which stores the number of observations and the sample mean. `T` is the (floating-point)
type of a single observation and can be omitted if it can be inferred from the arguments.

"""
const SampleMean{T} = SampleStat{1,T}

"""
    var = SampleVariance{T}(args...)
    var = SampleStat{2,T}(args...)

Return a 2-nd order sample statistics object built according to the arguments `args...` and
which stores the number of observations, the sample mean, and the sample variance. `T` is
the (floating-point) type of a single observation and can be omitted if it can be inferred
from the arguments.

"""
const SampleVariance{T} = SampleStat{2,T}

# Accessors.
Base.count(A::SampleStat) = getfield(A, :count)
moments(A::SampleStat) = getfield(A, :moments)

# Traits.
#
# Number of moments.
order(A::SampleStat) = order(typeof(A))
order(A::Type{<:SampleStat{M,T}}) where {M,T} = M
#
check_order(M::Int) = M ≥ 0
check_order(M::Any) = false
#
@noinline throw_bad_order(M::Int) = throw(AssertionError(
    "statistics order must be non-negative, got `M = $M`"))
@noinline throw_bad_order(M::Any) = throw(AssertionError(
    "statistics order must be an `Int`, got `typeof(M) = $(typeof(M))`"))
#
# Type of observations.
obstype(A::SampleStat) = obstype(typeof(A))
obstype(A::Type{<:SampleStat{M,T}}) where {M,T} = T
#
check_obstype(::Type{T}) where {T} =
    isconcretetype(T) && isconcretetype(get_precision(T))
#
@noinline throw_bad_obstype(T::Type) = throw(AssertionError(
    "observation type parameter `T = $T` is not concrete floating-point"))

# Indexing.
@propagate_inbounds function Base.getindex(A::SampleStat, k::Integer)
    return moments(A)[Int(k)::Int]
end

function Base.show(io::IO, x::SampleStat{M,T}) where {M,T<:Number}
    print(io, "SampleStat{", M, ", ")
    show(io, T)
    print(io, "}(", count(x), ", (")
    flag = true
    for k in 1:M
        if flag
            flag = false
        else
            print(io, ", ")
        end
        print(io, x[k])
    end
    print(io, M == 1 ? ",))" : "))")
end

function Base.show(io::IO, mime::MIME"text/plain", x::SampleStat{M,T}) where {M,T<:Number}
    if M == 0
        print(io, "SampleCount{")
    elseif M == 1
        print(io, "SampleMean{")
    elseif M == 2
        print(io, "SampleVariance{")
    else
        print(io, "SampleStat{", M, ", ")
    end
    show(io, mime, T)
    print(io, "}: n=", count(x))
    for k in 1:M
        print(io, ", μ")
        print_index(io, k)
        print(io, "=", x[k])
    end
    return nothing
end

function print_index(io::IO, k::Int)
    k ≥ 0 || error("index must be ≥ 0")
    _print_index(io, k)
end
function _print_index(io::IO, k::Int)
    if k ≥ 10
        n, k = divrem(k, 10)
        _print_index(io, n)
    end
    print(io, '₀' + k)
end

# NOTE `ntuple` is much faster than unrolling to compute the `M` first powers of `x`.
@inline powers(x::Number, M::Int) = powers(x, Val(M))
@inline powers(x::Number, ::Val{0}) = ()
@inline powers(x::Number, ::Val{M}) where {M} = ntuple(n -> x^n, Val(M))

"""
    stat = SampleStat{M,T}()
    stat = SampleStat{M}(T) -> SampleStat{M,float(T)}()

Return an empty sample statistics object for `M` moments of i.i.d. variables of type `T`. If
specified as a type parameter, `T` must be be floating-point; if specified as an argument,
it is automatically converted to floating-point. The returned object has no observations and
all moments equal to zero (with proper units).

"""
SampleStat{M,T,V}() where {M,T<:Number,V} = SampleStat{M,T}()::SampleStat{M,T,V}
SampleStat{M,T}() where {M,T<:Number} = _empty(SampleStat{M,T})
SampleStat{M}(::Type{T}) where {M,T<:Number} = empty(SampleStat{M,float(T)})

@noinline SampleStat{M}() where {M} = error("missing observation type")
@noinline SampleStat(::Type{T}) where {T<:Number} = error("missing statistic order")
@noinline SampleStat() = error("missing statistic order and observation type")

@generated function _empty(::Type{SampleStat{M,T}}) where {M,T}
    check_order(M) || return :(throw_bad_order(M))
    check_obstype(T) || return :(throw_bad_obstype(T))
    moments = powers(zero(T), Val(M))
    quote
        $(Expr(:meta, :inline))
        return SampleStat{M,T,$(typeof(moments))}(0, $moments)
    end
end

"""
    isempty(stat::SampleStat)

Return whether sample statistics `stat` has no observations.

"""
Base.isempty(x::SampleStat) = iszero(count(x))

"""
    empty(stat::SampleStat)
    empty(typeof(stat::SampleStat))

Return a sample statistics object of same type as `stat` but with no observations.

"""
Base.empty(x::SampleStat) = empty(typeof(x))
Base.empty(::Type{S}) where {S<:SampleStat} = S()

"""
    stat = SampleStat(n::Integer, moments::Tuple{Vararg{Number}})

Return a sample statistics object for `n` i.i.d. observations and their computed first
`moments`. The caller is responsible of the consistency of the arguments.

Type parameters `M` (the order of the sample statistics) and `T` (the type of a single
observation) are inferred from `moments` except for `0`-th order statistics (i.e.,
`SampleCount`) for which `moments` is an empty tuple and, hence, `T` must be specified. For
example:

```julia
n = 123 # number of observations
cnt = SampleCount{Float32}(n, ()) # same as SampleStat{0,Float32}(n, ())
```

"""
SampleStat(n::Integer, moments::NTuple{M,Any}) where {M} = SampleStat{M}(n, moments)

# Infer observation type `T` if not specified.
@inline function SampleStat{M}(n::Integer, moments::NTuple{M,Any}) where {M}
    M > 0 || error("observation type `T` must be provided for `SampleCount` statistics")
    T = adapt_precision(default_precision(moments), typeof(moments[1]))
    return SampleStat{M,T}(n, moments)
end

@inline function SampleStat{M,T}(n::Integer, moments::NTuple{M,Number}) where {M,T<:Number}
    check_obstype(T) || throw_bad_obstype(T)
    n ≥ 0 || throw_bad_count(n)
    V = typeof(powers(zero(T), Val(M)))
    return SampleStat{M,T,V}(n, convert(V, moments)::V)
end

@noinline throw_bad_count(n::Integer) =
    throw(ArgumentError("number of observations must be nonnegative, got $n"))

"""
    stat = SampleStat{M}(x::Number)
    stat = SampleStat{M,T}(x::Number)

Return a sample statistics object for `M` moments of i.i.d. observations of the same type as
`x` and initialized with the observation `x`. Hence, the returned object has a single
observation, a mean equal to `x`, and all other moments equal to zero (with proper units).
Optional type parameter `T` is the (floating-point) type of an observation assumed for the
statistics; by default, `T = float(typeof(x))`.

"""
SampleStat{M}(x::Number) where {M} = SampleStat{M,float(typeof(x))}(x)

function SampleStat{M,T}(x::Number) where {M,T}
    check_order(M) || throw_bad_order(M)
    check_obstype(T) || throw_bad_obstype(T)
    return _init(SampleStat{M,T}, x)
end

@inline function _init(::Type{SampleStat{0,T}}, x::Number) where {T}
    return SampleStat{0,T,Tuple{}}(1, ())
end

@generated function _init(::Type{SampleStat{M,T}}, x::Number) where {M,T}
    moments = Expr(:tuple, :(convert(T, x)::T), [zero(T)^k for k in 2:M]...)
    quote
        $(Expr(:meta, :inline))
        moments = $moments
        return SampleStat{M,T,typeof(moments)}(1, moments)
    end
end

"""
    stat = SampleStat{M}(iter)
    stat = SampleStat{M,T}(iter)

Return a sample statistics object for `M` moments of i.i.d. observations of the same type as
the elements of `iter` and initialized with the observations in `iter`. Hence, the returned
object has `length(iter)` observations. Optional type parameter `T` is the (floating-point)
type of an observation assumed for the statistics; by default, `T = float(eltype(iter))`.

"""
function SampleStat{M}(iter) where {M}
    Base.IteratorEltype(iter) isa Base.HasEltype || throw(ArgumentError(
        "`SampleStat{M}(iter)` requires that iterator has known element type, this may be overcome by `SampleStat{M,T}(iter)` "))
    T = float(eltype(iter))
    return SampleStat{M,T}(iter)
end

function SampleStat{M,T}(iter) where {M,T}
    check_order(M) || throw_bad_order(M)
    check_obstype(T) || throw_bad_obstype(T)
    return _reduce(SampleStat{M,T}, iter)
end

# Fallback methods.
SampleStat(x) = error("missing statistics order")
SampleStat{M,T,V}(x) where {M,T,V} = SampleStat{M,T}(x)::SampleStat{M,T,V} # FIXME args...

# `_reduce(S::Type{SampleStat{M,T}, x)` is like `reduce(S, x)` but assuming that `M`
# and `T` have already been checked.

@inline function _reduce(::Type{SampleCount{T}}, iter) where {T}
    if Base.IteratorSize(iter) isa Union{Base.HasLength, Base.HasShape}
        n = Int(length(iter)::Integer)::Int
    else
        n = 0
        @inbounds for x in iter
            n += 1
        end
    end
    return SampleStat{0,T,Tuple{}}(n, ())
end

@inline function _reduce(::Type{SampleMean{T}}, iter) where {T}
    s = zero(T)
    if Base.IteratorSize(iter) isa Union{Base.HasLength, Base.HasShape}
        n = Int(length(iter)::Integer)::Int
        @inbounds @simd for x in iter
            s += oftype(s, x)
        end
    else
        n = 0
        @inbounds for x in iter
            n += 1
            s += oftype(s, x)
        end
    end
    μ = (n < 1 ? s : s/n)::T
    return SampleStat{1,T,Tuple{T}}(n, (μ,))
end

@generated function _reduce(::Type{SampleStat{M,T}}, iter) where {M,T}
    # For k ≥ 2, powers are computed as: u_k = u_1*u_{k-1}
    init    = [:($(Symbol(:s_,k)) = zero(T)^$k) for k in 2:M]
    update  = [:($(Symbol(:u_,k)) = u_1*$(Symbol(:u_,k-1));
                 $(Symbol(:s_,k)) += $(Symbol(:u_,k))) for k in 2:M]
    moments_n = Expr(:tuple, :(μ), [:($(Symbol(:s_,k))/n) for k in 2:M]...)
    moments_0 = ntuple(k -> zero(T)^k, Val(M)) # moments when n=0
    V = typeof(moments_0)
    quote
        $(Expr(:meta, :inline))
        s = zero(T)
        n = 0
        if Base.IteratorSize(iter) isa Union{Base.HasLength, Base.HasShape}
            n = Int(length(iter)::Integer)::Int
            @inbounds @simd for x in iter
                s += oftype(s, x)
            end
        else
            @inbounds for x in iter
                n += 1
                s += oftype(s, x)
            end
        end
        n ≥ 1 || return SampleStat{M,T,$V}(n, $(moments_0))
        μ = (s/n)::T
        $(init...)
        if Base.IteratorSize(iter) isa Union{Base.HasLength, Base.HasShape}
            @inbounds @simd for x in iter
                u_1 = convert(T, x - μ)
                $(update...)
            end
        else
            @inbounds for x in iter
                u_1 = convert(T, x - μ)
                $(update...)
            end
        end
        return SampleStat{M,T,$V}(n, $(moments_n))
    end
end

# Conversion constructors.
SampleStat(A::SampleStat) = A

SampleStat{M}(A::SampleStat{M}) where {M} = A
SampleStat{M}(A::SampleStat{<:Any,T}) where {M,T} =
    SampleStat{M,T}(count(A), moments(A)[1:M])

SampleStat{M,T}(A::SampleStat{M,T}) where {M,T} = A
SampleStat{M,T}(A::SampleStat) where {M,T} =
    SampleStat{M,T}(count(A), moments(A)[1:M])::SampleStat{M,T}

SampleStat{M,T,V}(A::SampleStat{M,T,V}) where {M,T,V} = A
SampleStat{M,T,V}(A::SampleStat) where {M,T,V} =
    SampleStat{M,T}(count(A), moments(A)[1:M])::SampleStat{M,T,V}

Base.convert(::Type{T}, x::T) where {T<:SampleStat} = x
Base.convert(::Type{T}, x) where {T<:SampleStat} = T(x)::T

# Extend `TypeUtils`.
TypeUtils.get_precision(::Type{T}) where {T<:SampleStat} = get_precision(obstype(T))
function TypeUtils.adapt_precision(::Type{T}, A::SampleStat) where {T<:Precision}
    return SampleStat{order(A),adapt_precision(T, obstype(A))}(A)
end

# Extend `StatsBase`.
StatsBase.nobs(A::SampleStat) = count(A)
StatsBase.moment(A::SampleStat, k::Integer) = A[k]

# Extend `Statistics`.
function Statistics.mean(A::SampleStat{M}) where {M}
    M ≥ 1 || error("sample statistics does not include variance")
    return @inbounds A[1]
end

Statistics.std(A::SampleStat; kwds...) = sqrt(var(A; kwds...))

function Statistics.var(A::SampleStat{M}; corrected::Bool=true) where {M}
    M ≥ 2 || error("sample statistics does not include variance")
    v = @inbounds A[2]
    if corrected
        n = count(A)
        return n*v/(n - 1) # this expression shall preserve type
    else
        return v
    end
end

# NOTE `merge(A::SampleStat, B)` returns an object of same type as `A`.
#
"""
    merge(A::SampleStat, B) -> C::typeof(A)

Merge sample statistics `A` with observation(s) or sample statistics `B`. The result `C` has
the same type as `A` and represents the sample statistics for the observations in `A` plus
those in `B` assuming all these observations are independent.

If `B` is a sample statistics of the same order as `A` or a single observation (i.e. a
number), `A` and `B` are merged using single-pass *pairwise* or *incremental* update
formulae (Welford, 1962; Bennett et al., 2009). This is useful for parallel computations or
when new observations arrive continuously.

Otherwise, `B` is assumed to be an iterator (e.g. an array) of observations whose sample
statistics are computed and then merged with `A` as described above.

"""
function Base.merge(A::SampleStat, B)
    if B isa Number
        # Ensure compatibility of single observation `B` by converting it to the type of
        # observations in `A`.
        return merge(A, convert(obstype(A), B)::obstype(A))
    elseif B isa SampleStat
        # Ensure compatibility of observations by converting `B` to same type as `A`.
        order(B) ≥ order(A) || throw(ArgumentError(
            "other sample statistics must of order ≥ $(order(A)), got $(order(B))-order sample statistics"))
        return merge(A, convert(typeof(A), B)::typeof(A))
    else
        # Fallback method, assume `B` is an iterator whose elements are observations.
        return merge(A, reduce(typeof(A), B)::typeof(A))
    end
end

@generated function Base.merge(A::S, x::X) where {M,X,S<:SampleStat{M,X}}
    #
    # Adapted to our definitions of the moments, the single-pass incremental update formula
    # (III.4) in Bennett et al. (2016) writes:
    #
    #     μₚ(C) = α*[μₚ(A) + (-β*δ)^p + sum_{k=1}^{p-2} binomial(p,k)*(-β*δ)^k*μₚ₋ₖ(A)]
    #           + β*[μₚ(B) + (α*δ)^p]
    #
    # where last term simplifies to β*(α*δ)^p for p > 1 and with:
    #
    #     B = {x}
    #     n(B) = 1
    #     μ₁(B) = x
    #     μₖ(B) = 0    (for all k > 1)
    #
    # and thus:
    #
    #     C = A ∪ {x}
    #     n(C) = n(A) + 1
    #     δ = x - μ₁(A)
    #     β = 1/(n(A) + 1)
    #     α = n(A)*β
    #
    code = Expr[]
    # Pre-compute powers of (α*δ)^k
    αδ = [Symbol("αδ_",k) for k in 1:M] # names of symbols for (α*δ)^k
    push!(code, :($(αδ[1]) = α*δ))
    for k in 2:M
        push!(code, :($(αδ[k]) = $(αδ[k÷2])*$(αδ[k-k÷2])))
    end
    # Pre-compute powers of (-β*δ)^k
    βδ = [Symbol("βδ_",k) for k in 1:M]  # names of symbols for (-β*δ)^k
    push!(code, :($(βδ[1]) = -β*δ))
    for k in 2:M
        push!(code, :($(βδ[k]) = $(βδ[k÷2])*$(βδ[k-k÷2])))
    end
    # Push merge expressions for μ_2, ..., μ_M
    for p in 2:M
        # Build μₚ(A) + (-β*δ)^p + sum_{k=1}^{p-2} binomial(p,k)*(-β*δ)^k*μₚ₋ₖ(A)
        ex = :(A[$p] + $(βδ[p]))
        if p > 2
            # Build sum_{k=1}^{p-2} binomial(p,k)*(-β*δ)^k*μₚ₋ₖ(A)
            sum = Expr(:call, :+)
            for k in 1:p-2
                push!(sum.args, :($(binomial(p,k))*$(βδ[k])*A[$(p-k)]))
            end
            push!(ex.args, sum)
        end
        push!(code, :($(Symbol("μ_",p)) = α*$(ex) + β*$(αδ[p])))
    end
    moments = Expr(:tuple, ntuple(k -> Symbol("μ_",k), Val(M))...)
    quote
        $(Expr(:meta, :inline))
        T = get_precision(X)
        nA = count(A)
        n = nA + 1
        β = (one(T)/n)::T
        α = (nA*β)::T
        μ_1 = (α*A[1] + β*x)::T
        δ = (x - A[1])::T
        $(code...)
        return S(n, $moments)
    end
end

@inline function Base.merge(A::S, x::T) where {T,S<:SampleCount{T}}
    return S(count(A) + 1, ()) # directly call inner constructor
end

@inline function Base.merge(A::S, x::T) where {T,S<:SampleMean{T}}
    n, μ = count(A), A[1]
    u = (x - μ)/(n + 1)
    return S(n + 1, (μ + u,)) # directly call inner constructor
end

@inline function Base.merge(A::S, x::T) where {T,S<:SampleVariance{T}}
    # Update sample mean and variance using recurrence rules similar to those given by
    # Welford (1962).
    n, μ, v = count(A), A[1], A[2]
    u = (x - μ)/(n + 1)
    return S(n + 1, (μ + u, n*(v/(n + 1) + u*u))) # directly call inner constructor
end

@inline function Base.merge(A::S, B::S) where {S<:SampleCount}
    return S(count(A) + count(B), ()) # directly call inner constructor
end

@inline function Base.merge(A::S, B::S) where {S<:SampleMean}
    T = get_precision(S)
    nA, μA = count(A), A[1]
    nB, μB = count(B), B[1]
    #
    # Merging of the mean:
    #
    #    n = nA + nB                   (1 op.)
    #    μ = (nA/n)*μA + (nB/n)*μB     (+ 5 ops. but "symmetric")
    #      = μA + (nB/n)*(μB - μA)     (+ 4 ops.)
    #      = μB + (nA/n)*(μA - μB)     (+ 4 ops.)
    #
    n = nA + nB
    α = (T(nA)/n)::T
    β = (T(nB)/n)::T
    μ = α*μA + β*μB
    return S(n, (μ,)) # directly call inner constructor
end

@inline function Base.merge(A::S, B::S) where {S<:SampleVariance}
    T = get_precision(S)
    nA, μA, vA = count(A), A[1], A[2]
    nB, μB, vB = count(B), B[1], B[2]
    n = nA + nB
    α = (T(nA)/n)::T
    β = (T(nB)/n)::T
    μ = α*μA + β*μB
    #
    # A simple expression for the merged variance which is nonnegative (in 9 ops.):
    #
    #     v = α*(vA + (μA - μ)^2) + β*(vB + (μB - μ)^2)
    #
    # Another version proposed by Pébay et al. (2016) is also nonnegative (in 8 ops.):
    #
    #     v = α*vA + β*vB + α*β*(μA - μB)^2
    #
    # factorization saves one operation and adds a small correction
    # `β*(μA - μB)² ≈ β*σ²*(1/nA + 1/nB) ≈ σ²/n` `vA ≈ σ²` to `vA ≈ σ²`:
    #
    v = α*(vA + β*(μB - μA)^2) + β*vB
    return S(n, (μ, v)) # directly call inner constructor
end

@generated function Base.merge(A::S, B::S) where {M,S<:SampleStat{M}}
    #
    # Adapted to our definitions of the moments, the single-pass pairwise update formula
    # (III.1) in Bennett et al. (2016) writes:
    #
    #     μₚ(C) = α*[μₚ(A) + (-β*δ)^p] + β*[μₚ(B) + (α*δ)^p]
    #             + sum_{k=1}^{p-2} binomial(p,k)*[(-β*δ)^k*α*μₚ₋ₖ(A) + (α*δ)^k*β*μₚ₋ₖ(B)]
    #
    # in 7p - 3 operations (not counting pre-computed powers and binomial coefficients)
    # which can be put in an "interpolation form":
    #
    #     μₚ(C) = α*[μₚ(A) + (-β*δ)^p + sum_{k=1}^{p-2} binomial(p,k)*(-β*δ)^k*μₚ₋ₖ(A)]
    #           + β*[μₚ(B) + ( α*δ)^p + sum_{k=1}^{p-2} binomial(p,k)*( α*δ)^k*μₚ₋ₖ(B)]
    #
    # in 6p - 7 operations and with:
    #
    #     C = A ∪ B)
    #     n(C) = n(A) + n(B)
    #     δ = μ₁(B) - μ₁(A)
    #     α = n(A)/n(A ∪ B)
    #     β = n(B)/n(A ∪ B)
    #
    # TODO The sums should be computed in such an order to minimize the propagation of
    #      rounding errors.
    #
    code = Expr[]
    # Pre-compute powers of (α*δ)^k
    αδ = [Symbol("αδ_",k) for k in 1:M] # names of symbols for (α*δ)^k
    push!(code, :($(αδ[1]) = α*δ))
    for k in 2:M
        push!(code, :($(αδ[k]) = $(αδ[k÷2])*$(αδ[k-k÷2])))
    end
    # Pre-compute powers of (-β*δ)^k
    βδ = [Symbol("βδ_",k) for k in 1:M]  # names of symbols for (-β*δ)^k
    push!(code, :($(βδ[1]) = -β*δ))
    for k in 2:M
        push!(code, :($(βδ[k]) = $(βδ[k÷2])*$(βδ[k-k÷2])))
    end
    # Push merge expressions for μ_2, ..., μ_M
    for p in 2:M
        exA = :(A[$p] + $(βδ[p]))
        exB = :(B[$p] + $(αδ[p]))
        if p > 2
            # Build sum_{k=1}^{p-2} binomial(p,k)*(-β*δ)^k*μₚ₋ₖ(A)
            # and   sum_{k=1}^{p-2} binomial(p,k)*(α*δ)^k*μₚ₋ₖ(B)
            push!(exA.args, Expr(:call, :+))
            push!(exB.args, Expr(:call, :+))
            for k in 1:p-2
                w = binomial(p,k)
                push!(exA.args[end].args, :($w*$(βδ[k])*A[$(p-k)]))
                push!(exB.args[end].args, :($w*$(αδ[k])*B[$(p-k)]))
            end
        end
        push!(code, :($(Symbol("μ_",p)) = α*$(exA) + β*$(exB)))
    end
    moments = Expr(:tuple, ntuple(k -> Symbol("μ_",k), Val(M))...)
    quote
        $(Expr(:meta, :inline))
        T = get_precision(S)
        nA = count(A)
        nB = count(B)
        # NOTE We could implement the following shortcut but this would imply branch and
        #      thus prevent vectorization:
        #          nA > 0 || return B
        #          nB > 0 || return A
        n = nA + nB
        α = (T(nA)/n)::T
        β = (T(nB)/n)::T
        μ_1 = (α*A[1] + β*B[1])::T
        δ = (B[1] - A[1])::T
        $(code...)
        return S(n, $moments)
    end
end

# Extend `Base.reduce`, the many methods are needed to avoid ambiguities.
Base.reduce(::Type{S}, x::Number) where {S<:SampleStat} = S(x)
for T in (:Any, :AbstractArray, :(SharedArrays.SharedArray))
    @eval Base.reduce(::Type{S}, iter::$T) where {S<:SampleStat} = S(iter)
end

for f in (:isequal, :(==))
    @eval begin
        function Base.$f(x::SampleStat{M}, y::SampleStat{M}) where {M}
            return (count(x) == count(y)) && $f(moments(x), moments(y))
        end
    end
end

# FIXME counts must be the same
function Base.isapprox(x::SampleCount, y::SampleCount; kwds...)
    return count(x) == count(y)
end
function Base.isapprox(x::SampleMean, y::SampleMean; kwds...)
    return count(x) == count(y) && isapprox(moments(x)[1], moments(y)[1]; kwds...)
end
function Base.isapprox(x::SampleStat{M,Tx}, y::SampleStat{M,Ty};
                       atol::Number=zero(Tx)+zero(Ty), kwds...) where {M,Tx,Ty}
    # NOTE `atol` keyword refers to the mean, its power is taken for other modes.
    count(x) == count(y) || return false
    atol1 = convert(typeof(zero(Tx)+zero(Ty)), atol)
    for k in 1:M
        isapprox(moments(x)[k], moments(y)[k]; atol=atol1^k, kwds...) || return false
    end
    return true
end

default_precision(x) = default_precision(get_precision(x))
default_precision(::Type{T}) where {T<:AbstractFloat} = isconcretetype(T) ? T : Float64

end # module
