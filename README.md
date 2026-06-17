# SampleStats

[![Build Status](https://github.com/emmt/SampleStats.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/emmt/SampleStats.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/emmt/SampleStats.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/emmt/SampleStats.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

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
