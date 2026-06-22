# SampleStats

[![Build Status](https://github.com/emmt/SampleStats.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/emmt/SampleStats.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/emmt/SampleStats.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/emmt/SampleStats.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

`SampleStats` provides immutable objects to store sample statistics of independent and
identically distributed (i.i.d.) observations. Sample statistics can be efficiently computed
from a given set of observations, but also incremented, or merged to account for new
observation(s).

## Usage

If observations are provided by an iterator `iter` (e.g. an array) the `M`-order sample
statistics can be computed by:

``` julia
stat = SampleStat{M}(iter)
```

If a single observation `x` is available:

``` julia
stat = SampleStat{M}(x)
```

yields the `M`-order sample statistics for this observation. Finally, if no observations are
available yet, an empty sample statistics can be created by:

``` julia
stat = SampleStat{M}(T)
```

with `T` the type of a single observation.

If new or other observations are available, they can be merged in `stat` by:

``` julia
stat = merge(stat, other)
```

where `other` can be a single observation, a collection of observations (an iterator),
another sample statistics object. The result of `merge(stat, other)` is type-stable and is
always an object of the exact same type as `stat`. Although this would not be as efficient,
the first above example could be implemented by:

``` julia
stat = SampleStat{M}(eltype(iter)) # start with an "empty" sample statistics
for x in iter                      # for each observation...
    stat = merge(stat, x)          # ... merge the observation in the sample statistics
end
```

A sample statistics object can be indexed: `stat[k]` yields `μₖ`, the `k`-th moment stored
in `stat`. A `M`-order sample statistics object `stat` stores all moments `μₖ` for `k=1` to
`M`. For `n` observations, `x₁`, `x₂`, ..., `xₙ`, these moments are computed as:

``` julia
μ₁ = (1/n) Σᵢ xᵢ           # 1st moment is the sample mean
μₖ = (1/n) Σᵢ (xᵢ - μ₁)^k  # if k > 1
```

Applicable methods are:

``` julia
count(stat) # number of observations
nobs(stat) # number of observations
moments(stat) # all the moments
moment(stat, k) # k-th moment
stat[k     ]    # idem.
mean(stat) # sample mean
stat[1]    # idem.
var(stat; corrected=true)  # unbiased sample variance
var(stat; corrected=false) # biased sample variance
var(stat)                  # idem.
stat[2]                    # idem.
std(stat; corrected=...)   # sample standard deviation
isempty(stat) # whether no observations have been taken into account yet
```

By default, keyword `corrected` is `false` in the `var` and `std` method.


A sample statistics object `stat` has two *traits*:

``` julia
order(stat)           # sample statistics order
order(typeof(stat))   # idem.
obstype(stat)         # type of an observation
obstype(typeof(stat)) # idem.
```

The sample statistics order is the parameter `M` in the above examples. The observation type
is the type `T` in the above examples converted to floating-point because statistical
moments can only be memorized as floating-point numbers. As can be noted, a *trait* does
only depend on the type of the object.

There are a few exported aliases: `SampleCount` is the same as `SampleStat{0}`, `SampleMean`
is the same as `SampleStat{1}`, and `SampleVariance` is the same as `SampleStat{2}`.


## References

Incremental and pairwise update formulae for statistics of different orders have been
derived by:

- B. P. Welford, 1962, *"Note on a Method for Calculating Corrected Sums of Squares and
  Products"*, Technometrics **4**, 419-420.
  [DOI](https://doi.org/10.1080/00401706.1962.10490022)

- J. Bennett, R. Grout, P. Pebay, D. Roe, and D. Thompson, 2009, *"Numerically stable,
  single-pass, parallel statistics algorithms"*, IEEE International Conference on Cluster
  Computing and Workshops. [DOI](https://doi.org/10.1109/CLUSTR.2009.5289161)

- P. Pébay, T.B. Terriberry, H. Kolla, and J. Bennett, 2016, *"Numerically stable, scalable
  formulas for parallel and online computation of higher-order multivariate central moments
  with arbitrary weights"*, Comput. Stat. **31**, 1305–1325.
  [DOI](https://doi.org/10.1007/s00180-015-0637-z)

## Similar packages

The objective of `SampleStats` is not to replace existing packages (see below) but rather to
provide building blocks to implement well tested and efficient two-pass to compute sample
statistics and also single-pass incremental or pairwise update formulae.

Comparison with [`OnlineStats`](https://github.com/joshday/OnlineStats.jl):

- `OnlineStats` objects are mutable, those of `SampleStat` are immutable and, thus, more
  suitable to [avoid allocations](https://github.com/JuliaLang/AllocCheck.jl) and real-time
  computations.

- Computing sample statistics with `SampleStats` requires no allocations and is as fast or
  faster than with `OnlineStats`. However, since, one-pass recursive formulae are used,
  both are slower than calling `mean`, `var`, etc. from `Statistics`.

- Computing moments of order ≥ 2 with `SampleStats` is done with centered observations which
  avoids overflows and insures that moments of even order are non-negative.

- The moments computed with `SampleStats` are not bias corrected (i.e., they are biased).

- `SampleCount{T} = SampleStat{0,T}` is the counterpart of `OnlineStats.Count`,
  `SampleMean{T} = SampleStat{1,T}` is the counterpart of `OnlineStats.mean{T}`, and
  `SampleVariance{T} = SampleStat{2,T}` is the counterpart of `OnlineStats.Variance{T}`.

- `OnlineStats` offers many other possibilities. For now, `SampleStats` has a simple and
  limited objective: compute or update the statistical moments of observations.

Comparison with [`MultiVariateOnlineStatistics`](https://github.com/emmt/MultiVariateOnlineStatistics.jl):

- `MultiVariateOnlineStatistics` computes the sample statistics of multi-variate observations, an observation
  being specified by an array. In `SampleStat`, an observation is a number.

- `MultiVariateOnlineStatistics` is limited to 2nd order (variance) statistics. `SampleStat`
  has no such limitation.

- `MultiVariateOnlineStatistics` could use `SampleStat` to compute the moments.

Comparison with [`OnlineSampleStatistics`](https://github.com/FerreolS/OnlineSampleStatistics.jl):

- `OnlineSampleStatistics` is for multi-variate observations.
