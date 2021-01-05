using Flux
using MLDataPattern
using Random
using CSV
using DataFrames
using CodecZlib
using Mmap
using BSON
using Distributions: Uniform

using Flux: flatten
import MLDatasets

# datasetdir(args...) = projectdir("datasets", args...)
datasetdir(args...) = joinpath("/disk/macha/data_oms/datasets", args...)

# -------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------
abstract type Dataset; end
name(T::Type{<:Dataset}, args...) = join([name(T), args...])

function load(D::Type{<:Dataset}, T = Float32; kwargs...)
    return reshape_dataset.(load_raw(D, T); kwargs...)
end

function load_raw(D::Type{<:Dataset}, T)
    prepare(D)
    d = BSON.load(datasetdir(name(D, ".bson")))
    train = (Array{T}(d[:train][:x]'), d[:train][:y])
    valid = (Array{T}(d[:valid][:x]'), d[:valid][:y])
    test = (Array{T}(d[:test][:x]'), d[:test][:y])

    return train, valid, test
end

function reshape_dataset((x,y)::Tuple; labelmap = identity)
    return (reshape_samples(x), reshape_labels(labelmap.(y)))
end

reshape_labels(y) = y
reshape_labels(y::AbstractVector) = Array(reshape(y, 1, :))
reshape_samples(x) = x

function reshape_samples(x::AbstractArray{T, 3}) where T
    return Array(reshape(x, size(x, 1), size(x, 2), 1, size(x, 3)))
end

function split_data(x, y; seed = 1234, kwargs...)
    Random.seed!(seed)
    return stratifiedobs((x, y); kwargs...)
end

function create_dict(train, valid, test; T = Float32)
    return Dict(
        :train => Dict(:x => Array{T}(train[1]), :y => Vector(train[2])),
        :valid => Dict(:x => Array{T}(valid[1]), :y => Vector(valid[2])),
        :test => Dict(:x => Array{T}(test[1]), :y => Vector(test[2])),
    )
end

function build_network(D::Type{<:Dataset}; seed = 1234, T = Float32)
    Random.seed!(seed)
    return Dense(nfeatures(D), 1)
end

# -------------------------------------------------------------------------------
# Ionosphere
# -------------------------------------------------------------------------------
abstract type Ionosphere <: Dataset end
name(::Type{Ionosphere}) = "ionosphere"
url(::Type{Ionosphere}) = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
nfeatures(::Type{Ionosphere}) = 34

function prepare(T::Type{Ionosphere})
    bson_file = datasetdir(name(T, ".bson"))
    isfile(bson_file) && return

    csv_file = datasetdir(name(T, ".csv"))
    isfile(csv_file) || download(url(T), csv_file)

    d = CSV.read(csv_file; header = false)
    x = Array(d[:, 1:(end-1)])
    y = Vector(d[:, end] .== "g")

    train, valid, test = split_data(x, y;  p = (0.5, 0.25), obsdim = 1)
    BSON.bson(bson_file, create_dict(train, valid, test))
    return
end

# -------------------------------------------------------------------------------
# Spambase
# -------------------------------------------------------------------------------
abstract type Spambase <: Dataset end
name(::Type{Spambase}) = "spambase"
url(::Type{Spambase}) = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
nfeatures(::Type{Spambase}) = 57

function prepare(T::Type{Spambase})
    bson_file = datasetdir(name(T, ".bson"))
    isfile(bson_file) && return

    csv_file = datasetdir(name(T, ".csv"))
    isfile(csv_file) || download(url(T), csv_file)

    d = CSV.read(csv_file; header = false)
    x = Array(d[:, 1:(end-1)])
    y = Vector(d[:, end] .== 1)

    train, valid, test = split_data(x, y;  p = (0.5, 0.25), obsdim = 1)
    BSON.bson(bson_file, create_dict(train, valid, test))
    return
end

# -------------------------------------------------------------------------------
# Hepmass
# -------------------------------------------------------------------------------
abstract type Hepmass <: Dataset end
name(::Type{Hepmass}) = "hepmass"
function url(::Type{Hepmass})
    return [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00347/all_train.csv.gz",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00347/all_test.csv.gz"
    ]

end
nfeatures(::Type{Hepmass}) = 28

function prepare(T::Type{Hepmass})
    bson_file = datasetdir(name(T, ".bson"))
    isfile(bson_file) && return

    csv_file_train = datasetdir(name(T, "_train.csv.gz"))
    csv_file_test = datasetdir(name(T, "_test.csv.gz"))
    isfile(csv_file_train) || download(url(T)[1], csv_file_train)
    isfile(csv_file_test) || download(url(T)[2], csv_file_test)

    d_train = CSV.File(transcode(GzipDecompressor, Mmap.mmap(csv_file_train))) |> DataFrame
    x_train = Array(d_train[:, 2:end])
    y_train = Vector(d_train[:, 1] .== 1)

    d_test = CSV.File(transcode(GzipDecompressor, Mmap.mmap(csv_file_test))) |> DataFrame
    x_test = Array(d_test[:, 2:end])
    y_test = Vector(d_test[:, 1] .== 1)

    train, valid = split_data(x_train, y_train;  p = 0.75, obsdim = 1)
    test = (x_test, y_test)
    BSON.bson(bson_file, create_dict(train, valid, test))
    return
end

# -------------------------------------------------------------------------------
# Gisette
# -------------------------------------------------------------------------------
abstract type Gisette <: Dataset end
name(::Type{Gisette}) = "gisette"
function url(::Type{Gisette})
    return [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.labels",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_valid.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/gisette_valid.labels"
    ]

end
nfeatures(::Type{Gisette}) = 5000

function prepare(T::Type{Gisette})
    bson_file = datasetdir(name(T, ".bson"))
    isfile(bson_file) && return

    csv_file_train_data = datasetdir(name(T, "_train_data.csv"))
    csv_file_train_labels = datasetdir(name(T, "_train_labels.csv"))
    csv_file_valid_data = datasetdir(name(T, "_valid_data.csv"))
    csv_file_valid_labels = datasetdir(name(T, "_valid_labels.csv"))
    isfile(csv_file_train_data) || download(url(T)[1], csv_file_train_data)
    isfile(csv_file_train_labels) || download(url(T)[2], csv_file_train_labels)
    isfile(csv_file_valid_data) || download(url(T)[3], csv_file_valid_data)
    isfile(csv_file_valid_labels) || download(url(T)[4], csv_file_valid_labels)

    d_train = CSV.read(csv_file_train_data; header = false)
    l_train = CSV.read(csv_file_train_labels; header = false)
    x_train = Array{Float64}(d_train[:, 1:end-1])
    y_train = Vector(l_train[:, 1] .== 1)

    d_valid = CSV.read(csv_file_valid_data; header = false)
    l_valid = CSV.read(csv_file_valid_labels; header = false)
    x_valid = Array{Float64}(d_valid[:, 1:end-1])
    y_valid = Vector(l_valid[:, 1] .== 1)

    nrm = sqrt.(sum(abs2, x_train; dims = 1))
    nrm[nrm .== 0] .= 1
    x_train ./= nrm
    x_valid ./= nrm

    train = (x_train, y_train)
    valid, test = split_data(x_valid, y_valid;  p = 0.5, obsdim = 1)
    BSON.bson(bson_file, create_dict(train, valid, test))
    return
end


# -------------------------------------------------------------------------------
# Gisette
# -------------------------------------------------------------------------------
abstract type ToyExample <: Dataset end
nfeatures(::Type{ToyExample}) = 2

function load(::Type{ToyExample}; T = Float32, k = 1, kwargs...)
    n = 1000
    x_neg = hcat(rand(Uniform(-1,0), n-k), rand(Uniform(-2,2), n-k))
    if k == 1
        x_neg = vcat(x_neg, [2 0])
    else
        x_neg = vcat(x_neg, [[2 i] for i in range(-1; stop = 1, length = k)]...)
    end
    x_pos = hcat(rand(Uniform(0,1), n), rand(Uniform(-2,2), n))

    x = Array{T}(vcat(x_neg, x_pos))
    y = Vector(1:2n .> n)

    return reshape_dataset((x', y); kwargs...)
end
