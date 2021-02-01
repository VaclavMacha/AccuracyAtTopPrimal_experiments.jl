module Experiments

using Reexport
using RecipesBase

@reexport using AccuracyAtTopPrimal
@reexport using BSON
@reexport using CSV
@reexport using DatasetProvider
@reexport using DataFrames
@reexport using Flux
@reexport using Plots

using BenchmarkTools
using DrWatson
using EvalMetrics
using HypothesisTests
using Measurements
using PaperUtils
using Query

using ProgressMeter
using Random
using Statistics

using AccuracyAtTopPrimal: AbstractThreshold
using Base.Iterators: partition
using BSON: load
using DrWatson: datadir
using Flux: gpu, cpu, params
using StatsBase: sample

export Data, Train, Model
export trainmodel, davemodel, loadmodel, loaddata, run_simulations, collect_metrics, selectmetric, selectbest, betterzero, crit_diag, run_benchmarks, summary, restore_path, loadrow, plotroc, wilcoxontest, corelationmatrix, mergecsv

modeldir(args...) = datadir("models", args...)

include("corrmatrix.jl")
include("data_train_model.jl")
include("train.jl")
include("evaluation.jl")

end # module
