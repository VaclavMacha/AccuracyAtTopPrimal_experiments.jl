using Experiments
using Experiments: AccuracyAtTopPrimal, DatasetProvider

using DatasetProvider: HEPMASS

# ------------------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------------------
# dataset settings
datasets = Data(HEPMASS; batchsize = 512)

# train settings
train = Train(;
    seed = seed,
    iters = 10000,
    saveat = 5000,
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

iter = 10000
force = true
df_train = collect_metrics(; iter, subset = :train, force)
df_valid = collect_metrics(; iter, subset = :valid, force)
df_test = collect_metrics(; iter, subset = :test, force)
@info "finished"
