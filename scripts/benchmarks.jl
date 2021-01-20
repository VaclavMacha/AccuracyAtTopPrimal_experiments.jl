using Experiments
using Experiments: AccuracyAtTopPrimal, DatasetProvider, Flux

using DatasetProvider: Ionosphere, Spambase, Gisette, MNIST, FashionMNIST, CIFAR10, CIFAR20, CIFAR100, SVHN2, HEPMASS

# ------------------------------------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------------------------------------
datasets = [
    Data(Ionosphere),
    Data(Spambase),
    Data(Gisette; batchsize = 512),
    Data(MNIST; poslabels = 1, batchsize = 512),
    Data(FashionMNIST; poslabels = 1, batchsize = 512),
    Data(CIFAR10; poslabels = 1, batchsize = 512),
    Data(SVHN2; poslabels = 1, batchsize = 512),
    Data(HEPMASS; batchsize = 512),
]

table = run_benchmarks(datasets; force = true)
