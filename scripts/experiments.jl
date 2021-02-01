using DrWatson
quickactivate(@__DIR__, "AccuracyAtTopPrimal_oms")

using Experiments
using Experiments: AccuracyAtTopPrimal, DatasetProvider

using DatasetProvider: Ionosphere, Spambase, Gisette, MNIST, FashionMNIST, CIFAR10, CIFAR20, CIFAR100, SVHN2, HEPMASS

# ------------------------------------------------------------------------------------------
# Model settings
# ------------------------------------------------------------------------------------------
τ = [0.01, 0.05]
K = [1, 3, 5, 10, 15, 20]
β = [10, 1, 0.1, 0.01, 0.001, 0.0001]
λ = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
surrogate = hinge

models = vcat(
    Model(PatMat; τ, β, surrogate),
    Model(PatMatNP; τ, β, surrogate),
    Model(TopPush; λ),
    Model(TopPushK; K),
    Model(τFPL; τ, λ, surrogate),
    Model(TopMean; τ, λ, surrogate),
    Model(Grill; τ, λ, surrogate),
    Model(GrillNP; τ, λ, surrogate),
)

# ------------------------------------------------------------------------------------------
# Datasets with 10 seeds
# ------------------------------------------------------------------------------------------
datasets = [
    Data(Ionosphere),
    Data(Spambase),
    Data(Gisette; batchsize = 512),
    Data(MNIST; poslabels = 1, batchsize = 512),
    Data(FashionMNIST; poslabels = 1, batchsize = 512),
    Data(CIFAR10; poslabels = 1, batchsize = 512),
    Data(CIFAR20; poslabels = 1, batchsize = 512),
    Data(CIFAR100; poslabels = 1, batchsize = 512),
    Data(SVHN2; poslabels = 1, batchsize = 512),
]

train = [Train(;
    seed = seed,
    iters = 10000,
    saveat = 500,
    optimiser = ADAM,
    step = 0.01,
) for seed in 1:10]

run_simulations(datasets, train, models)

# HEPMASS
train_hep = [Train(;
    seed = seed,
    iters = 10000,
    saveat = 10000,
    optimiser = ADAM,
    step = 0.01,
) for seed in 1:10]

run_simulations(Data(HEPMASS; batchsize = 512), train_hep, models)


# ------------------------------------------------------------------------------------------
# Datasets with 1 seeds
# ------------------------------------------------------------------------------------------
datasets_1 = reduce(vcat, [[
    Data(MNIST; poslabels = l, batchsize = 512),
    Data(FashionMNIST; poslabels = l, batchsize = 512),
    Data(CIFAR10; poslabels = l, batchsize = 512),
    Data(CIFAR20; poslabels = l, batchsize = 512),
    Data(CIFAR100; poslabels = l, batchsize = 512),
    Data(SVHN2; poslabels = l + 1, batchsize = 512),
] for l in 0:9])

train_1 = Train(;
    seed = 1,
    iters = 10000,
    saveat = 500,
    optimiser = ADAM,
    step = 0.01,
)

run_simulations(datasets_1, train_1, models)


# ------------------------------------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------------------------------------
datasets_bench = [
    Data(Ionosphere),
    Data(Spambase),
    Data(Gisette; batchsize = 512),
    Data(MNIST; poslabels = 1, batchsize = 512),
    Data(FashionMNIST; poslabels = 1, batchsize = 512),
    Data(CIFAR10; poslabels = 1, batchsize = 512),
    Data(SVHN2; poslabels = 1, batchsize = 512),
    Data(HEPMASS; batchsize = 512),
]

table = run_benchmarks(datasets_bench; force = true)
