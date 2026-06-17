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

Base.isempty(x::SampleStat) = iszero(count(x))

"""
    stat = SampleStat{M,T}()
    stat = SampleStat{M}(T) -> SampleStat{M,float(T)}()

Return an empty sample statistics object for `M` moments of i.i.d. variables of type `T`. If
specified as a type parameter, `T` must be be floating-point; if specified as an argument,
it is automatically converted to floating-point. The returned object has no observations and
all moments equal to zero (with proper units).

"""
SampleStat{M}(::Type{T}) where {M,T<:Number} = empty(SampleStat{M,float(T)})
SampleStat{M,T}() where {M,T<:Number} = empty(SampleStat{M,T})

"""
    empty(stat::SampleStat)
    empty(typeof(stat::SampleStat))

Return a sample statistics object of same type as `stat` but with no observations.

"""
Base.empty(x::SampleStat) = empty(typeof(x))

Base.empty(::Type{SampleStat{M,T,V}}) where {M,T,V} = empty(SampleStat{M,T})::SampleStat{M,T,V}

@generated function Base.empty(::Type{SampleStat{M,T}}) where {M,T}
    check_order(M) || return :(throw_bad_order(M))
    check_obstype(T) || return :(throw_bad_obstype(T))
    moments = powers(zero(T), Val(M))
    quote
        $(Expr(:meta, :inline))
        return SampleStat{M,T,$(typeof(moments))}(0, $moments)
    end
end

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
SampleStat(n::Integer, moments::NTuple{M,Number}) where {M} = SampleStat{M}(n, moments)

# Infer observation type `T` if not specified.
@inline function SampleStat{M}(n::Integer, moments::NTuple{M,Number}) where {M}
    M > 0 || error("type parameter `T` must be provided for `SampleCount` statistics")
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

Return a sample statistics object for `M` moments of i.i.d. observations of the same type as
`x` and initialized with the observation `x`. Hence, the returned object has a single
observation, a mean equal to `x`, and all other moments equal to zero (with proper units).

"""
SampleStat{M}(x::Number) where {M} = SampleStat{M,float(typeof(x))}(x)

@generated function SampleStat{M,T}(x::Number) where {M,T<:Number}
    check_order(M) || return :(throw_bad_order(M))
    check_obstype(T) || return :(throw_bad_obstype(T))
    z = zero(T)
    moments = ntuple(k -> k == 1 ? :(convert($T, x)::$T) : z^k, Val(M))
    quote
        $(Expr(:meta, :inline))
        return SampleStat{M,T,$(typeof(moments))}(1, $moments)
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

function Statistics.var(A::SampleStat{M}; corrected::Bool=false) where {M}
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
@inline function Base.merge(A::SampleStat{M,T}, x::Number) where {M,T<:Number}
    return merge(A, convert(T, x)::T)
end

@inline function Base.merge(A::SampleStat{0,T,V}, x::T) where {T<:Number,V}
    return SampleStat{0,T,V}(count(A) + 1, ())
end

@inline function Base.merge(A::SampleStat{1,T,V}, x::T) where {T<:Number,V}
    n, μ = count(A), A[1]
    u = (x - μ)/(n + 1)
    return SampleStat{1,T,V}(n + 1, (μ + u,))
end

@inline function Base.merge(A::SampleStat{2,T,V}, x::T) where {T<:Number,V}
    # Update sample mean and variance using recurrence rules similar to those given by
    # Welford (1962).
    n, μ, v = count(A), A[1], A[2]
    u = (x - μ)/(n + 1)
    return SampleStat{2,T,V}(n + 1, (μ + u, n*(v/(n + 1) + u*u)))
end

function Base.merge(A::SampleStat{M,T,V}, xs #= iterator =#) where {M,T<:Number,V}
    @inbounds for x in xs
        A = merge(A, x)
    end
    return A
end

@inline function Base.merge(A::SampleCount, B::SampleCount)
    return typeof(A)(count(A) + count(B), ())
end

@inline function Base.merge(A::SampleMean, B::SampleMean)
    T = get_precision(A)
    nA, μA = count(A), adapt_precision(T, A[1])
    nB, μB = count(B), adapt_precision(T, B[1])
    n = nA + nB
    α = (T(nA)/n)::T
    β = (T(nB)/n)::T
    μ = α*μA + β*μB
    return typeof(A)(n, (μ,))
end

@inline function Base.merge(A::SampleVariance, B::SampleVariance)
    T = get_precision(A)
    nA, μA, vA = count(A), adapt_precision(T, A[1]), adapt_precision(T, A[2])
    nB, μB, vB = count(B), adapt_precision(T, B[1]), adapt_precision(T, B[2])

    n = nA + nB
    α = (T(nA)/n)::T
    β = (T(nB)/n)::T
    μ = α*μA + β*μB
    # A simple expression for the merged variance which is nonnegative:
    #
    #     v = α*(vA + (μA - μ)^2) + β*(vB + (μB - μ)^2)
    #
    # Another version proposed by Pébay et al. (2016) is also nonnegative and involves fewer
    # operations:
    #
    #     v = (nA/n)*vA + (nB/n)*vB + (nA*nB/n)*(μB - μA)^2
    #
    v = α*(vA + nB*(μB - μA)^2) + β*vB
    return typeof(A)(n, (μ, v))
end

# Extend `Base.reduce`, the 2 methods are needed to avoid ambiguities.
Base.reduce(::Type{S}, x::Number) where {S<:SampleStat} = S(x)
Base.reduce(::Type{S}, xs) where {S<:SampleStat} = _reduce(S, xs)
Base.reduce(::Type{S}, xs::AbstractArray) where {S<:SampleStat} = _reduce(S, xs)

function _reduce(::Type{SampleStat{M}}, xs) where {M}
    return merge(SampleStat{M}(eltype(xs)), xs)
end

function _reduce(::Type{SampleStat{M,T}}, xs) where {M,T}
    return merge(SampleStat{M,T}(), xs)
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
function Base.isapprox(x::SampleStat{M,Tx}, y::SampleStat{M,Ty},
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
