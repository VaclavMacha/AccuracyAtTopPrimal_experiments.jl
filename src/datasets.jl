using Flux
using Random
using Distributions: Uniform

using DatasetProvider
using DatasetProvider: Ionosphere, Spambase, Gisette, MNIST, FashionMNIST, CIFAR10, CIFAR20, CIFAR100, HEPMASS
using DatasetProvider: GrayImages, ColorImages

using Flux: flatten


# -------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------
function reshape_dataset(data::Tuple)
    x, y = data
    return (reshape_samples(x), reshape_labels(y))
end

reshape_labels(y) = y
reshape_labels(y::AbstractVector) = Array(reshape(y, 1, :))
reshape_samples(x) = x

reshape_samples(x::AbstractMatrix) = Matrix(x')
reshape_samples(x::AbstractArray) = Array(reshape(x, :, size(x)[end]))

function loaddataset(N::Type{<:Name}, poslabels)
    data = DatasetProvider.Dataset(N; asmatrix = true, shuffle = true, poslabels, seed = 1234, binarize = true)

    return  reshape_dataset.(DatasetProvider.load(TrainValidTest(), data))
end

function build_network(N::Type{<:Name}; seed = 1234, T = Float32)
    Random.seed!(seed)
    input = prod(DatasetProvider.nattributes(N))
    return Dense(input, 1)
end

# -------------------------------------------------------------------------------
# Toy example
# -------------------------------------------------------------------------------
function load_toyexample(; T = Float32, k = 1, kwargs...)
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
