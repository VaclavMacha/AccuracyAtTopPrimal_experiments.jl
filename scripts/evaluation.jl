using Experiments
using PaperUtils

using Experiments: AccuracyAtTopPrimal, DatasetProvider, EvalMetrics, summary

using DatasetProvider: Ionosphere, Spambase, Gisette, MNIST, FashionMNIST, CIFAR10, CIFAR20, CIFAR100, SVHN2, HEPMASS

# ------------------------------------------------------------------------------------------
# Load results
# ------------------------------------------------------------------------------------------
EvalMetrics.showwarnings(false)

iter = 10000
force = false
df_train = collect_metrics(; iter, subset = :train, force)
df_valid = collect_metrics(; iter, subset = :valid, force)
df_test = collect_metrics(; iter, subset = :test, force)


# ------------------------------------------------------------------------------------------
# Datasets summary
# ------------------------------------------------------------------------------------------
datasets = [
    Data(Ionosphere),
    Data(Spambase),
    Data(Gisette; batchsize = 512),
    Data(HEPMASS; batchsize = 512),
    Data(MNIST; poslabels = 1, batchsize = 512),
    Data(FashionMNIST; poslabels = 1, batchsize = 512),
    Data(CIFAR10; poslabels = 1, batchsize = 512),
    Data(CIFAR20; poslabels = 1, batchsize = 512),
    Data(CIFAR100; poslabels = 1, batchsize = 512),
    Data(SVHN2; poslabels = 1, batchsize = 512),
]

table = summary(datasets; force = true)


# ------------------------------------------------------------------------------------------
# Critical diagrams
# ------------------------------------------------------------------------------------------
for m in [:fpr_1, :fpr_5, :quant_1, :quant_5, :top, :auroc_1, :auroc_5]
    crit_diag(df_valid, df_test, m; α = 0.01, addparams = true, addseed = true)
end


# ------------------------------------------------------------------------------------------
# Loss <= Loss_zero
# ------------------------------------------------------------------------------------------
table = betterzero(df_train)


# ------------------------------------------------------------------------------------------
# Wilcoxon test
# ------------------------------------------------------------------------------------------
for m in [:fpr_1, :fpr_5, :quant_1, :quant_5, :top, :auroc_1, :auroc_5]
    wilcoxontest(df_valid, df_test, m; addparams = true, addseed = true)
end

# ------------------------------------------------------------------------------------------
# ROC plots
# ------------------------------------------------------------------------------------------
for m in [:fpr_1, :fpr_5, :quant_1, :quant_5, :top, :auroc_1, :auroc_5]
    plotroc(df_valid, m; iter, subset = :test)
end

mergecsv()
