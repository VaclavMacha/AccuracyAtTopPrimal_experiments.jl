using Experiments
using Experiments: AccuracyAtTopPrimal, DatasetProvider, EvalMetrics

using DatasetProvider: MNIST, FashionMNIST, CIFAR10, CIFAR20, CIFAR100, SVHN2

# ------------------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------------------
# dataset settings
datasets = reduce(vcat, [[
    Data(MNIST; poslabels = l, batchsize = 512),
    Data(FashionMNIST; poslabels = l, batchsize = 512),
    Data(CIFAR10; poslabels = l, batchsize = 512),
    Data(CIFAR20; poslabels = l, batchsize = 512),
    Data(CIFAR100; poslabels = l, batchsize = 512),
    Data(SVHN2; poslabels = l + 1, batchsize = 512),
] for l in 0:9])

# train settings
train = Train(;
    seed = 1,
    iters = 10000,
    saveat = 500,
    optimiser = ADAM,
    step = 0.01,
)

# model settings
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
# Train and collect
# ------------------------------------------------------------------------------------------
run_simulations(datasets, train, models)
