logrange(x1, x2; kwargs...) = exp10.(range(log10(x1), log10(x2); kwargs...))


# Eval functions
function tpr_at_fpr(targets, scores, rate)
    t = threshold_at_fpr(targets, scores, rate)
    return true_positive_rate(targets, scores, t)
end

function tpr_at_quantile(targets, scores, rate)
    t = quantile(scores, 1 - rate)
    return true_positive_rate(targets, scores, t)
end

function tpr_at_top(targets, scores)
    t = maximum(scores[targets .== 0])
    return true_positive_rate(targets, scores, t)
end

function partial_auroc(targets, scores, fprmax)
    rates = logrange(1e-5, fprmax; length=1000)
    ts = unique(threshold_at_fpr(targets, scores, rates))

    fprs = false_positive_rate(targets, scores, ts)
    inds = unique(i -> fprs[i], 1:length(fprs))
    fprs = fprs[inds]
    tprs = true_positive_rate(targets, scores, ts[inds])

    auc_max = abs(maximum(fprs) - minimum(fprs))

    return 100*EvalMetrics.auc_trapezoidal(fprs, tprs)/auc_max
end

#
function mergesettings(dict)
    d = deepcopy(dict)
    data = d[:dataset_settings]
    train = d[:train_settings]
    model = d[:model_settings]
    pars = pop!(model, :pars)

    return merge(data, train, model, pars)
end

extract(d, iter, subset, field) = d[:iterations][iter][subset][field]

function computemetrics(d, iter, subset)
    targets = extract(d, iter, subset, :targets)
    scores = extract(d, iter, subset, :scores)

    dict = mergesettings(d)
    dict[:subset] = subset
    dict[:iter] = iter
    dict[:loss] = extract(d, iter, subset, :loss)
    dict[:loss_zero] = extract(d, iter, subset, :loss_zero)
    dict[:top] = tpr_at_top(targets, scores)
    dict[:fpr_1] = tpr_at_fpr(targets, scores, 0.01)
    dict[:fpr_5] = tpr_at_fpr(targets, scores, 0.05)
    dict[:quant_1] = tpr_at_quantile(targets, scores, 0.01)
    dict[:quant_5] = tpr_at_quantile(targets, scores, 0.05)
    dict[:auroc_1] = partial_auroc(targets, scores, 0.01)
    dict[:auroc_5] = partial_auroc(targets, scores, 0.05)
    get!(dict, :τ, missing)
    get!(dict, :K, missing)
    get!(dict, :β, missing)
    delete!(dict, :surrogate)

    return DataFrame(dict)
end

function collect_metrics(
    path = modeldir();
    iter = 1000,
    subset::Symbol = :test,
    force = false
)

    file = datadir("results", "table_$(subset)_$(iter).csv")
    isfile(file) && !force && return CSV.read(file, DataFrame; header = true)

    rows = []
    for (root, dirs, files) in walkdir(path)
        for file in files
            dict = BSON.load(joinpath(root, file))
            push!(rows, computemetrics(dict, iter, subset))
        end
    end
    table = reduce(vcat, rows)

    mkpath(dirname(file))
    CSV.write(file, table)
    return table
end

#
function selectmetric(df, metric::Symbol)
    table = select(df, vcat([:dataset, :model, :K, :β, :λ, :τ], metric))

    table = @from i in table begin
        @group i by {i.dataset, i.model, i.K, i.β, i.λ, i.τ} into gr
        @select {
            dataset = Query.key(gr).dataset,
            model = Query.key(gr).model,
            K = Query.key(gr).K,
            β = Query.key(gr).β,
            λ = Query.key(gr).λ,
            τ = Query.key(gr).τ,
            metric = mean(getproperty(gr, metric)),
        }
        @collect DataFrame
    end
    rename!(table, :metric => metric)
    return table
end

function selectmetric(df, metrics)
    dfs = selectmetric.(Ref(df), metrics)
    return innerjoin(
        dfs...;
        on = [:dataset, :model, :K, :β, :λ, :τ],
        matchmissing = :equal
    )
end

function selectbest(df_valid, df_test, metric::Symbol; wide = true)
    table_valid = selectmetric(df_valid, metric)
    table_test = selectmetric(df_test, metric)

    key_val = Symbol(metric, "_valid")
    rename!(table_valid, metric => key_val)

    table = innerjoin(
        table_valid,
        table_test;
        on = [:dataset, :model, :K, :β, :λ, :τ,],
        matchmissing = :equal
    )

    table_best = @from i in table begin
        @group i by {i.dataset, i.model} into gr
        @select {
            dataset = Query.key(gr).dataset,
            model = Query.key(gr).model,
            metric = getproperty(gr, metric)[argmax(getproperty(gr, key_val))]
        }
        @collect DataFrame
    end
    rename!(table_best, :metric => metric)

    if wide
        return unstack(table_best, :model, metric)
    else
        return table_best
    end
end

function crit_diag(df_valid, df_test, metric; α = 0.05)
    table = selectbest(df_valid, df_test, metric; wide = true)
    rank_df = PaperUtils.rankdf(table)

    R = Vector(rank_df[end, 2:end])
    n, k = size(rank_df) .- 1
    algnames = names(rank_df)[2:end]
    replace!(algnames, "τFPL" => "\$\\tau\$-FPL")
    ncd = PaperUtils.nemenyi_cd(k, n, α)

    file = datadir("results", "crit_diag_$(metric).tex")
    mkpath(dirname(file))
    PaperUtils.string2file(file, PaperUtils.ranks2tikzcd(R, algnames, ncd))
    return
end
