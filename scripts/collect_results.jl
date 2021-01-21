using Experiments
using Experiments: EvalMetrics

EvalMetrics.showwarnings(false)

iter = 10000
force = true
df_train = collect_metrics(; iter, subset = :train, force)
df_valid = collect_metrics(; iter, subset = :valid, force)
df_test = collect_metrics(; iter, subset = :test, force)

@info "finished"
