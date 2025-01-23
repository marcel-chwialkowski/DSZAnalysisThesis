using NeuralVerification, LazySets, LazySets.Approximations
import NeuralVerification: Network, Layer, ReLU, Id, compute_output, ActivationFunction, get_bounds, forward_act
using LinearAlgebra
import LazySets.Approximations.interval_hull

using ProbabilityBoundsAnalysis, IntervalArithmetic # PyPlot
import IntervalArithmetic: Interval
using Distributions, Random
using DataFrames
using SplitApplyCombine
using PyPlot

a = uniform(0, 1)
ProbabilityBoundsAnalysis.plot(a)
PyPlot.savefig("kachow.png")
