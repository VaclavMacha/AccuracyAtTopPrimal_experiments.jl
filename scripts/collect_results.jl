using DrWatson
@quickactivate "AccuracyAtTopPrimal_oms"

include(srcdir("utilities.jl"))

# ------------------------------------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------------------------------------
df = collect_benchmark()

mkpath(datadir("results"))
CSV.write(datadir("results", "benchmark.csv"), df)


# ------------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------------
df = collect_metrics(; key = :train)

cls = [
    :model,
    :dataset,
    :surrogate,
    :τ,
    :β,
    :K,
    :λ,
    :loss_mean,
    :loss_zeros_mean,
]

select!(df, cls)

table = map(groupby(df, [:model, :dataset, :τ])) do grp
    select_better_loss(grp)
end |> rows -> reduce(vcat, rows)

for (i, (model, τ)) in enumerate(zip(table.model, table.τ))
    ismissing(τ) && continue
    table.model[i] = string(model, "(", τ, ")")
end
select!(table, [:model, :dataset, :pars])

wide = unstack(table, :dataset, :pars)
wide = wide[:, [:model, :Ionosphere, :Spambase, :Gisette, :Hepmass]]

mkpath(datadir("results"))
CSV.write(datadir("results", "parameters.csv"), wide)

# ------------------------------------------------------------------------------------------
# Ranks
# ------------------------------------------------------------------------------------------
df_valid = collect_metrics(; key = :valid)
df_test = collect_metrics(; key = :test)

models = [
    (Grill, 0.01),
    (Grill, 0.05),
    (GrillNP, 0.01),
    (GrillNP, 0.05),
    (PatMat, 0.01),
    (PatMat, 0.05),
    (PatMatNP, 0.01),
    (PatMatNP, 0.05),
    (TopMean, 0.01),
    (TopMean, 0.05),
    (TopPush, NaN),
    (TopPushK, NaN),
    (τFPL, 0.01),
    (τFPL, 0.05),
]

metrics = [
    :tpr_at_top_mean,
    :tpr_at_fpr_1_mean,
    :tpr_at_fpr_5_mean,
    :tpr_at_quant_1_mean,
    :tpr_at_quant_5_mean,
]

ranks = map(metrics) do metric
    df = find_best(df_valid, df_test, models, metric)
    table = @linq df |>
        by([:type], rank = mean(:tiedrank))
    rename!(table, :rank => Symbol(replace("$metric", "_mean" => "")))
    return table
end |> tbls -> innerjoin(tbls..., on = :type)

mkpath(datadir("results"))
CSV.write(datadir("results", "ranks.csv"), ranks)

ranks_model = map(metrics) do metric
    df = find_best(df_valid, df_test, models, metric; bymodelmetric = true)
    table = @linq df |>
        by([:type], rank = mean(:tiedrank))
    rename!(table, :rank => Symbol(replace("$metric", "_mean" => "")))
    return table
end |> tbls -> innerjoin(tbls..., on = :type)

mkpath(datadir("results"))
CSV.write(datadir("results", "ranks_model.csv"), ranks_model)
