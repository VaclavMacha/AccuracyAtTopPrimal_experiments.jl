# ------------------------------------------------------------------------------------------
# Eval metrics
# ------------------------------------------------------------------------------------------
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

logrange(x1, x2; kwargs...) = exp10.(range(log10(x1), log10(x2); kwargs...))

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


# ------------------------------------------------------------------------------------------
# Collect results
# ------------------------------------------------------------------------------------------
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
    dict[:loss_best] = minimum(d[:loss])
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
    prog = ProgressUnknown("Collecting results ($(subset) set at iteration = $(iter)) ")
    for (root, dirs, files) in walkdir(path)
        for file in files
            try
                dict = BSON.load(joinpath(root, file))
                push!(rows, computemetrics(dict, iter, subset))
            catch
                @warn "File corrupted: $(joinpath(root, file))"
            end
            ProgressMeter.next!(prog)
        end
    end
    ProgressMeter.finish!(prog)
    table = reduce(vcat, rows)

    mkpath(dirname(file))
    CSV.write(file, table)
    return table
end


# ------------------------------------------------------------------------------------------
# Metric selection
# ------------------------------------------------------------------------------------------
function datasetname(name, poslabel, seed; addposlabels = true, addseed = true)
    pars = []
    addposlabels && push!(pars, "poslabel = $poslabel")
    addseed && push!(pars, "seed = $seed")
    if isempty(pars)
        return string(name)
    else
        return string(name, "(", join(pars, ","), ")")
    end
end

function modelname(model, τ; addparams = true)
    addparams || return model
    if model in ["Grill", "GrillNP", "τFPL", "TopMean", "PatMat", "PatMatNP"]
        return string.(model, "(", τ, ")")
    end
    return model
end

function selectmetric(df, metric::Symbol; reducefunc = mean, addparams = true, kwargs...)
    tmp = select(df, vcat([:dataset, :poslabels, :seed, :model,  :K, :β, :λ, :τ], metric))
    tmp.dataset .= datasetname.(tmp.dataset, tmp.poslabels, tmp.seed; kwargs...)
    tmp.model .= modelname.(tmp.model, tmp.τ; addparams)

    table = @from i in tmp begin
        @group i by {i.dataset, i.model, i.K, i.β, i.λ, i.τ} into gr
        @select {
            dataset = key(gr).dataset,
            model = key(gr).model,
            K = key(gr).K,
            β = key(gr).β,
            λ = key(gr).λ,
            τ = key(gr).τ,
            metric = reducefunc(getproperty(gr, metric)),
        }
        @collect DataFrame
    end
    rename!(table, :metric => metric)
    return table
end

function selectmetric(df, metrics; kwargs...)
    dfs = selectmetric.(Ref(df), metrics; kwargs...)
    return innerjoin(
        dfs...;
        on = [:dataset, :model, :K, :β, :λ, :τ],
        matchmissing = :equal
    )
end

function selectbest(
    df_valid,
    df_test,
    metric::Symbol;
    selector = argmax,
    wide = true,
    kwargs...
)

    table_valid = selectmetric(df_valid, metric; kwargs...)
    table_test = selectmetric(df_test, metric; kwargs...)

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
            metric = getproperty(gr, metric)[selector(getproperty(gr, key_val))]
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

function betterparams(model, gr)
    n = length(gr.loss)
    inds = findall(gr.loss .<= gr.loss_zero)

    if length(inds) == n
        return "all"
    elseif isempty(inds)
        return "none"
    else
        if get(model) == "TopPushK"
            pars = gr.K
        elseif any(contains.(get(model), ["Grill", "τFPL", "TopMean", "TopPush"]))
            pars = gr.λ
        else
            pars = gr.β
        end
        return join(sort(get.(pars)[inds]), ", ")
    end
end

function betterzero(df; wide = true)
    table = selectmetric(
        df,
        [:loss, :loss_zero];
        reducefunc = minimum,
        addposlabels = false,
        addseed = false
    )

    table_pars = @from i in table begin
        @group i by {i.dataset, i.model} into gr
        @select {
            dataset = Query.key(gr).dataset,
            model = Query.key(gr).model,
            params = betterparams(Query.key(gr).model, gr)
        }
        @collect DataFrame
    end

    if wide
        return unstack(table_pars, :dataset, :params)
    else
        return table_pars
    end
end

# ------------------------------------------------------------------------------------------
# Critical diagrams
# ------------------------------------------------------------------------------------------
renamealgs(alg) = replace(alg, "τ" => "\$\\tau\$-")

function crit_diag(df_valid, df_test, metric; α = 0.05, kwargs...)
    table = selectbest(df_valid, df_test, metric; wide = true, kwargs...)
    rank_df = PaperUtils.rankdf(table)

    R = Vector(rank_df[end, 2:end])
    n, k = size(rank_df) .- 1
    algnames = names(rank_df)[2:end]
    algnames = renamealgs.(algnames)
    ncd = PaperUtils.nemenyi_cd(k, n, α)

    file = datadir("results", "crit_diag_$(metric).tex")
    mkpath(dirname(file))
    PaperUtils.string2file(file, PaperUtils.ranks2tikzcd(R, algnames, ncd))
    return
end

# ------------------------------------------------------------------------------------------
# Dataset summary
# ------------------------------------------------------------------------------------------
samplesize(N) = join(DatasetProvider.nattributes(N), "x")
ratio(y) = round(100*sum(y)/length(y); digits = 1)

function summary(dataset::Data{N}) where N
    train, valid, test = loaddata(dataset)
    return DataFrame([
        :dataset => N,
        :sample => samplesize(N),
        :batchsize => dataset.batchsize == 0 ? missing : dataset.batchsize,
        :train => length(train[2]),
        :train_ratio => ratio(train[2]),
        :valid => length(valid[2]),
        :valid_ratio => ratio(valid[2]),
        :test => length(test[2]),
        :test_ratio => ratio(test[2]),
    ])
end

function summary(datasets; force = false)
    file = datadir("results", "summary_datasets.csv")
    isfile(file) && !force && return CSV.read(file, DataFrame; header = true)

    table = reduce(vcat, summary.(datasets))

    mkpath(dirname(file))
    CSV.write(file, table)
    return table
end
