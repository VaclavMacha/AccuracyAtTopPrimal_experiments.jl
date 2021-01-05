using CSV
using DataFrames
using DataFramesMeta
using EvalMetrics
using Plots
using Statistics

pyplot()

using ProgressMeter: Progress, next!

# -------------------------------------------------------------------------------
# Eval functions
# -------------------------------------------------------------------------------
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

# -------------------------------------------------------------------------------
# Compute evaluation and add it to existing model files
# -------------------------------------------------------------------------------
function add_targets_scores(file::String, train::Tuple, valid::Tuple, test::Tuple; runon = cpu)
    if !endswith(file, ".bson")
        return false
    end

    d = BSON.load(file)
    overwrite = any([
        add_targets_scores!(d, train..., :train, runon),
        add_targets_scores!(d, valid..., :valid, runon),
        add_targets_scores!(d, test..., :test, runon),
        add_loss!(d, :minibatch),
        add_metrics!(d, :train),
        add_metrics!(d, :valid),
        add_metrics!(d, :test)
    ])
    overwrite && bson(file, d)
    return overwrite
end

function add_targets_scores!(d::Dict, x, y, key::Symbol, runon)
    overwrite = false
    if haskey(d, key)
        if !haskey(d[key], :targets)
            d[key][:targets] = cpu(vec(y))
            overwrite = true
        end
        if !haskey(d[key], :scores)
            model = d[:model] |> runon
            d[key][:scores] = cpu(compute_scores(model, x))
            overwrite = true
        end
    else
        model = d[:model] |> runon
        d[key] = Dict(
            :targets => cpu(vec(y)),
            :scores => cpu(compute_scores(model, x)),
        )
        overwrite = true
    end
    overwrite = add_loss!(d, key) || overwrite
    return overwrite
end

function add_loss!(d, key::Symbol)
    overwrite = false
    type = extract_model_type(d)
    if !haskey(d, key)
        return overwrite
    end
    if !haskey(d[key], :loss)
        pars = params(d[:model])
        targets = d[key][:targets]
        scores = d[key][:scores]

        @unpack type = d[:model_settings]
        loss = build_loss(type, d[:model_settings])

        d[key] = convert(Dict{Symbol, Any}, d[key])
        d[key][:loss] = loss(targets, scores, pars)
        overwrite = true
    end
    if !haskey(d[key], :loss_zeros)
        pars = params(d[:model])
        for p in pars
            p .= 0
        end
        targets = d[key][:targets]
        scores = d[key][:scores]

        @unpack type = d[:model_settings]
        loss = build_loss(type, d[:model_settings])

        d[key] = convert(Dict{Symbol, Any}, d[key])
        d[key][:loss_zeros] = loss(targets, zero(scores), pars)
        overwrite = true
    end
    return overwrite
end

function add_metrics!(d, key::Symbol)
    overwrite = false
    targets = d[key][:targets]
    scores = d[key][:scores]

    if !(typeof(d[key]) <: Dict{Symbol, Any})
        d[key] = convert(Dict{Symbol, Any}, d[key])
        overwrite = true
    end
    if !haskey(d[key], :tpr_at_top)
        d[key][:tpr_at_top] = tpr_at_top(targets, scores)
        overwrite = true
    end
    if !haskey(d[key], :tpr_at_fpr_1)
        d[key][:tpr_at_fpr_1] = tpr_at_fpr(targets, scores, 0.01)
        overwrite = true
    end
    if !haskey(d[key], :tpr_at_fpr_5)
        d[key][:tpr_at_fpr_5] = tpr_at_fpr(targets, scores, 0.05)
        overwrite = true
    end
    if !haskey(d[key], :tpr_at_quant_1)
        d[key][:tpr_at_quant_1] = tpr_at_quantile(targets, scores, 0.01)
        overwrite = true
    end
    if !haskey(d[key], :tpr_at_quant_5)
        d[key][:tpr_at_quant_5] = tpr_at_quantile(targets, scores, 0.05)
        overwrite = true
    end
    if !haskey(d[key], :auroc_1)
        d[key][:auroc_1] = partial_auroc(targets, scores, 0.01)
        overwrite = true
    end
    if !haskey(d[key], :auroc_5)
        d[key][:auroc_5] = partial_auroc(targets, scores, 0.05)
        overwrite = true
    end
    return overwrite
end

function run_evaluation(Dataset_Settings; runon = cpu)
    for dataset_settings in dict_list_simple(Dataset_Settings)
        @unpack dataset, posclass = dataset_settings
        @info "Dataset: $(dataset), positive class label: $(posclass)"

        labelmap = (y) -> y == posclass
        train, valid, test = load(dataset; labelmap = labelmap) |> runon

        dataset_dir = datadir("models", dataset_savename(dataset_settings))
        all_files = String[]
        for (root, dirs, files) in walkdir(dataset_dir)
            append!(all_files, joinpath.(root, files))
        end

        skipped = 0
        overwritten = 0
        p = Progress(length(all_files))
        for file in all_files
            overwrite = false
            try
                overwrite = add_targets_scores(file, train, valid, test; runon = runon)
            catch
                @warn "Problem with: $file"
            end
            if overwrite
                overwritten += 1
            else
                skipped += 1
            end
            next!(p; showvalues = [(:Skipped, skipped), (:Overwritten, overwritten)])
        end
    end
    return
end

# ------------------------------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------------------------------
function plot_convergence(model_settings; k = 1, iters = 1000)
    x, y = load(ToyExample; k = k)

    # training
    type = model_settings[:type]
    model = build_network(ToyExample)
    objective = build_loss(type, model_settings)
    pars = params(model)
    delete!(pars, model.b)

    loss(x, y) = objective(x, y, model, pars)

    # callback
    dict = Dict()
    function cb(φ)
        if haskey(dict, φ)
            dict[φ] = vcat(dict[φ], deepcopy(pars[1]))
        else
            dict[φ] = deepcopy(pars[1])
        end
        return
    end

    # training
    @showprogress map(range(0; stop = 2*π, length = 13)) do φ
        pars[1] .= [cos(φ) sin(φ)]
        opt = Flux.Optimiser(ExpDecay(0.1, 0.1, 300), Descent())
        for iter in 1:iters
            Flux.train!(loss, pars, [(x, y)], opt; cb = () -> cb(φ))
        end
    end

    # plots
    plt1 = plot(title = "$(model_name(type, model_settings)) with $(k) outliers");
    scatter!(plt1, x[1, vec(y) .== 0], x[2, vec(y) .== 0]; label = "negative sample", markerstrokecolor = false, markersize = 8);
    scatter!(plt1, x[1, vec(y) .== 1], x[2, vec(y) .== 1]; label = "positive samples", markerstrokecolor = false, markersize = 8);

    plt2 = plot(title = "trajectories", xlims = (-1.2, 1.2), ylims = (-1.2, 1.2));
    vline!(plt2, [0]; color = :red, linestyle = :dash, label = "");
    hline!(plt2, [0]; color = :red, linestyle = :dash, label = "");
    φ = range(0; stop = 2*π, length = 1000);
    plot!(plt2, cos.(φ), sin.(φ); color = :black, linestyle = :dot, label = "");
    for (key, val) in dict
        plot!(plt2, val[:,1], val[:,2]; label = "", linewidth = 2)
        scatter!(plt2, [val[end,1]], [val[end,2]]; label = "", primary = false, markersize = 14, markerstrokecolor = false, markershape = :hexagon)
    end

    plt = plot(plt1, plt2, layout = (1,2), size = (1200, 600))
    mkpath(plotsdir("convergence"))
    savefig(plt, plotsdir("convergence", "$(model_name(type, model_settings))_noutliers=$(k).png"))
    display(plt)
    return plt
end

function create_plots(
    path = datadir("models");
    save = true,
    )

    for (root, dirs, files) in walkdir(path)
        for file in files
            fl = joinpath(root, file)
            d = BSON.load(joinpath(root, file))

            plt = plot(
                pr_plot(d; key = :train, title = "train"),
                pr_plot(d; key = :test, title = "test"),
                ptau_plot(d; key = :train, title = ""),
                ptau_plot(d; key = :test, title = ""),
                roc_plot(d; key = :train, title = "", xlims = (1e-5, 1), xscale = :log10),
                roc_plot(d; key = :test, title = "", xlims = (1e-5, 1), xscale = :log10),
                layout = (3, 2),
                size = (800, 1200),
            )

            savename = replace(relpath(fl), "data/models/" => "")
            savename = replace(savename, ".bson" => ".png")

            mkpath(dirname(plotsdir("curves", savename)))
            savefig(plt, plotsdir("curves", savename))
        end
    end
    return
end

function pr_plot(d::Dict; key::Symbol = :test, kwargs...)
    targets = d[key][:targets]
    scores = d[key][:scores]
    return prplot(targets, scores; kwargs...)
end

function ptau_plot(d::Dict; key::Symbol = :test, kwargs...)
    targets = d[key][:targets]
    scores = d[key][:scores]

    τs = 0:0.005:1
    ts = quantile(scores, 1 .- τs)

    return plot(collect(τs), precision(targets, scores, ts);
        xlabel = "τ",
        ylabel = "precision",
        fillrange = 0,
        fillalpha = 0.15,
        kwargs...)
end

function roc_plot(d::Dict; key::Symbol = :test, kwargs...)
    targets = d[key][:targets]
    scores = d[key][:scores]
    return rocplot(targets, scores; kwargs...)
end

# ------------------------------------------------------------------------------------------
# Collecting results...
# ------------------------------------------------------------------------------------------
function add_missing!(d)
    get!(d, :dataset, missing)
    get!(d, :posclass, missing)
    get!(d, :batchsize, missing)
    get!(d, :iters, missing)
    get!(d, :optimiser, missing)
    get!(d, :steplength, missing)
    get!(d, :type, missing)
    get!(d, :τ, missing)
    get!(d, :K, missing)
    get!(d, :β, missing)
    get!(d, :surrogate, missing)
    get!(d, :λ, missing)
end

function collect_benchmark(path = datadir("benchmarks"); save = true)
    dfs = []
    for (root, dirs, files) in walkdir(path)
        for file in files
            fl = joinpath(root, file)
            dir_rel = relpath(fl, path)
            ~, bench = parse_savename(replace(dir_rel, "/" => "_"))
            bench = Dict(Symbol(key) => val for (key, val) in bench)

            d = BSON.load(fl)
            add_missing!(bench)
            bench[:time_per_iter] = mean(d[:times])/d[:iters]
            push!(dfs, DataFrame(bench))
        end
    end
    df = reduce(vcat, dfs)
    select!(df, [:type, :dataset, :time_per_iter])
    wide = unstack(df, :dataset, :time_per_iter)
    wide = wide[:, [:type, :Ionosphere, :Spambase, :Gisette, :Hepmass]]
    rename!(wide, :type => :model)
    return wide
end

err(x) = std(x; corrected = false)

function collect_metrics(
    path = datadir("models");
    key::Symbol = :test,
    iters = 1000,
    skipifcontains = "",
    save = true
    )

    dfs = []
    for (root, dirs, files) in walkdir(path)
        for file in files
            contains(file, string(1000)) || continue
            fl = joinpath(root, file)
            dir_rel = relpath(root, path)
            ~, dict = parse_savename(replace(dir_rel, "/" => "_"))
            dict = Dict(Symbol(key) => val for (key, val) in dict)
            add_missing!(dict)

            contains(fl, skipifcontains) && !isempty(skipifcontains)  && continue

            d = BSON.load(fl)
            dict[:tpr_at_top] = d[key][:tpr_at_top]
            dict[:tpr_at_fpr_1] = d[key][:tpr_at_fpr_1]
            dict[:tpr_at_fpr_5] = d[key][:tpr_at_fpr_5]
            dict[:tpr_at_quant_1] = d[key][:tpr_at_quant_1]
            dict[:tpr_at_quant_5] = d[key][:tpr_at_quant_5]
            dict[:auroc_1] = d[key][:auroc_1]
            dict[:auroc_5] = d[key][:auroc_5]
            dict[:loss] = d[key][:loss]
            dict[:loss_zeros] = d[key][:loss_zeros]
            push!(dfs, DataFrame(dict))
        end
    end
    df = reduce(vcat, dfs)
    select!(df, Not(["seed", "iters", "optimiser", "posclass", "steplength"]))
    rename!(df, "type" => "model")

    gdf = groupby(df, ["τ", "β", "K", "λ", "batchsize", "dataset", "model", "surrogate"])
    combs = [
        "auroc_1" => mean,
        "auroc_1" => err,
        "auroc_5" => mean,
        "auroc_5" => err,
        "tpr_at_fpr_1" => mean,
        "tpr_at_fpr_1" => err,
        "tpr_at_fpr_5" => mean,
        "tpr_at_fpr_5" => err,
        "tpr_at_quant_1" => mean,
        "tpr_at_quant_1" => err,
        "tpr_at_quant_5" => mean,
        "tpr_at_quant_5" => err,
        "tpr_at_top" => mean,
        "tpr_at_top" => err,
        "loss" => mean,
        "loss_zeros" => mean,
    ]
    df2 = combine(gdf, combs)

    if save
        mkpath(datadir("results"))
        CSV.write(datadir("results", "metrics.csv"), df2)
    end
    return df2
end

function collect_curves(
    Dataset_Settings,
    Train_Settings,
    Model_Settings;
    key = :test,
    iters = 1000,
    xlims = (1e-4, 1),
    npoints = 300,
)
    dataset_settings = deepcopy(Dataset_Settings)
    train_settings = deepcopy(Train_Settings)

    rates = logrange(xlims...; length = npoints - 1)
    rates = sort(vcat(rates, 0.01, 0.05))
    df = DataFrame()

    for model_settings in Model_Settings
        files = String[]
        for seed in 1:10
            train_settings[:seed] = seed
            dir = modeldir(dataset_settings, train_settings, model_settings)
            file = joinpath(dir, simulation_name(iters))
            isfile(file) || continue
            push!(files, file)
        end

        fprs = zeros(length(rates))
        tprs = zeros(length(rates))

        for file in files
            d = BSON.load(file)
            y = d[key][:targets]
            s = d[key][:scores]
            ts = threshold_at_fpr(y, s, rates)
            fp, tp = roccurve(y, s, ts)

            @info fp

            fprs .+= fp
            tprs .+= tp
        end
        fprs ./= length(files)
        tprs ./= length(files)

        @unpack type = model_settings
        mdl = modelname(type, model_settings)
        df["$(mdl)_fprates"] = fprs
        df["$(mdl)_tprates"] = tprs
    end
    delete!(dataset_settings, :posclass)
    delete!(train_settings, :seed)
    delete!(train_settings, :optimiser)
    delete!(train_settings, :iters)
    delete!(train_settings, :steplength)

    sett = merge(dataset_settings, train_settings)
    filename = savename(sett; allowedtypes = allowedtypes(), digits = 6)
    CSV.write(datadir("results", string(filename, "_$(key).csv")), df)
    return
end

# ------------------------------------------------------------------------------------------
# Ranks
# ------------------------------------------------------------------------------------------
const Model = Union{PatMat, PatMatNP, TopPush, TopPushK, Grill, GrillNP, τFPL, TopMean}

function mode_metric(::Type{<:Union{PatMatNP, GrillNP, τFPL}}, val)
    return Symbol("tpr_at_fpr_$(Int(100*val))_mean")
end

function mode_metric(::Type{<:Union{PatMat, Grill, TopMean}}, val)
    return Symbol("tpr_at_quant_$(Int(100*val))_mean")
end

function mode_metric(::Type{<:Union{TopPush, TopPushK}}, val)
    return Symbol("tpr_at_top_mean")
end

modeltype(name::String) = getfield(Main, Symbol(name))

function modelname(T::Type{<:Union{PatMat, PatMatNP, Grill, GrillNP, τFPL, TopMean}}, df)
    return "$(T)($(df[:τ][1]))"
end

modelname(T::Type{<:Union{TopPush, TopPushK}}, df) = "$T"

function select(
    df,
    dataset,
    T::Type{<:Union{PatMat, PatMatNP, Grill, GrillNP, τFPL, TopMean}},
    val::Real
)
    return @linq df |>
        where(:model .== string(T)) |>
        where(:dataset .== string(dataset)) |>
        where(:τ .== val)
end

function select(df, dataset, T::Type{<:Union{TopPush, TopPushK}}, val::Real)
    return @linq df |>
        where(:model .== string(T)) |>
        where(:dataset .== string(dataset))
end

function select(df, d::Dict)
    @unpack model, dataset = d
    table = @linq df |>
        where(:model .== model) |>
        where(:dataset .== dataset)
    ind = findall(vec(all(hcat([table[:, key] .== val for (key, val) in d]...); dims = 2)))
    return table[ind, :]
end

function find_best_params(df, dataset, T::Type{<:Model}, val, metric; best = findmax)
    table = select(df, dataset, T, val)
    ~, ind = best(table[:, metric])

    return Dict(key => table[ind, key] for key in vcat(:model, :dataset, validkeys(T)))
end

function find_best_params(df, dataset, T::Type{<:Model}, val; best = findmax)
    table = select(df, dataset, T, val)
    metric = mode_metric(T, val)
    ~, ind = best(table[:, metric])

    return Dict(key => table[ind, key] for key in vcat(:model, :dataset, validkeys(T)))
end

cols(args...) = vcat([:model, :dataset, :surrogate, :λ, :τ, :K, :β], args...)

function find_best(
    df_valid,
    df_test,
    dataset,
    models,
    metric;
    best = findmax,
    rev = true,
    bymodelmetric::Bool = false
)

    if bymodelmetric
        dicts = [find_best_params(df_valid, dataset, model, val; best = best) for (model, val) in models]
    else
        dicts = [find_best_params(df_valid, dataset, model, val, metric; best = best) for (model, val) in models]
    end

    rows = map(dicts) do dict
        row = select(df_test, dict)
        select!(row, cols(metric))
        row[:, :type] .= modelname(modeltype(row.model[1]), row)
        return row
    end
    table = reduce(vcat, rows)
    table[:, :tiedrank] = tiedrank(table[:, metric]; rev = rev)
    return table
end

function find_best(df_valid, df_test, models, metric; kwargs...)
    tbls = map([:Ionosphere, :Spambase, :Gisette, :Hepmass]) do dataset
        return find_best(df_valid, df_test, dataset, models, metric; kwargs...)
    end
    return reduce(vcat, tbls)
end

# ------------------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------------------
function select_better_loss(df)
    inds = findall(df.loss_mean .<= df.loss_zeros_mean)
    if length(inds) == length(df.loss_mean)
        pars = "all"
    elseif isempty(inds)
        pars = "none"
    else
        pars = select_parameters(modeltype(df.model[1]), df, inds)
    end
    return DataFrame(
        model = df.model[1],
        dataset = df.dataset[1],
        pars = pars
    )
end

function select_parameters(::Type{<:Union{PatMat, PatMatNP}}, df, inds)
    return string("β in [", join(sort(df.β[inds]), ", "), "]")
end


function select_parameters(::Type{TopPushK}, df, inds)
    return string("K in [", join(sort(df.K[inds]), ", "), "]")
end

function select_parameters(::Type{<:Union{Grill, GrillNP, τFPL, TopMean, TopPush}}, df, inds)
    return string("λ in [", join(sort(df.λ[inds]), ", "), "]")
end
