getdim(A::AbstractArray, d::Integer, i) = getindex(A, Base._setindex(i, d, axes(A)...)...)

function compute_scores(model, x; chunksize = 10000)
    x_obs = ndims(x)
    n = size(x, x_obs)
    scores = zeros(eltype(x), n)

    for inds in partition(1:n, chunksize)
        scores[inds] .= model(getdim(x, x_obs, inds))[:]
    end
    return scores
end

# batch provider
function batch_provider(d::Data, t::Train, x, y)
    Random.seed!(t.seed)
    if length(y) == d.batchsize || d.batchsize == 0
        return ((x, y) for iter in 1:t.iters)
    else

        neg = findall(vec(y) .== 0)
        pos = findall(vec(y) .== 1)

        n_neg = d.batchsize ÷ 2
        n_pos = d.batchsize - n_neg

        x_obs = ndims(x)
        y_obs = ndims(y)

        function make_batch()
            inds = vcat(
                sample(neg, n_neg; replace = length(neg) < n_neg),
                sample(pos, n_pos; replace = length(pos) < n_pos),
            )
            shuffle!(inds)
            return (getdim(x, x_obs, inds), getdim(y, y_obs, inds))
        end
        return (make_batch() for iter in 1:t.iters)
    end
end

# train!
function custom_train!(loss, ps, data, opt; cb = (args...) -> ())
    ps = Flux.Zygote.Params(ps)
    cb = Flux.Optimise.runall(cb)
    local loss_val

    for d in data
        try
            gs = Flux.Zygote.gradient(ps) do
                loss_val = loss(Flux.Optimise.batchmemaybe(d)...)
                return loss_val
            end
            Flux.Optimise.update!(opt, ps, gs)
            cb(loss_val, Flux.Optimise.batchmemaybe(d)...)
        catch ex
            if ex isa Flux.Optimise.StopException
                break
            else
                rethrow(ex)
            end
        end
    end
end

# callback
Base.@kwdef mutable struct CallBack
    iters::Int
    title::String = "Training:"
    bar::Progress = Progress(iters, 5, title)
    saveat::Int = 1000
    savefunc::Function = (args...) -> nothing
    counter::Int = 0
    loss = Float32[]
end

CallBack(iters; kwargs...) = CallBack(; iters, kwargs...)

function (c::CallBack)(loss_val, x, y)
    c.counter += 1
    push!(c.loss, loss_val)
    mod(c.counter, c.saveat) == 0 && c.savefunc(c, x, y)

    tm = round((c.bar.tlast - c.bar.tfirst)/c.bar.counter; sigdigits = 2)

    next!(c.bar; showvalues = vcat(
        ("Iteration", string(c.counter, "/", c.iters)),
        ("Time/iteration", string(tm, "s")),
        ("Loss0", c.loss[1]),
        ("Loss", c.loss[end]),
    ))
    return
end

# savemodel
function savemodel(
    c::CallBack,
    d::Data,
    t::Train,
    m::Model,
    model,
    x,
    y,
)
    file = modeldir(string(d), string(t), string(m, ".bson"))
    dict = isfile(file) ? BSON.load(file) : Dict{Symbol, Any}()

    s, y = compute_scores(model, x), vec(y) |> cpu

    get!(dict, :dataset_settings, todict(d))
    get!(dict, :train_settings, todict(t))
    get!(dict, :model_settings, todict(m))
    get!(dict, :iterations, Dict{Int, Any}())

    # upadte loss
    get!(dict, :loss, c.loss)
    dict[:loss] = c.loss

    # update time
    tm = (c.bar.tlast - c.bar.tfirst)
    get!(dict, :time_per_iter, 1.)
    get!(dict, :time_total, 1.)
    dict[:time_per_iter] = tm/c.bar.counter
    dict[:time_total] = tm

    # add new iter
    dict[:iterations][c.counter] = Dict(
        :minibatch => Dict(:targets => y, :scores => s),
        :model => deepcopy(cpu(model)),
    )

    mkpath(dirname(file))
    BSON.bson(file, dict)
    return
end

function loadmodel(d::Data, t::Train, m::Model)
    file = modeldir(string(d), string(t), string(m, ".bson"))
    if isfile(file)
        return BSON.load(file)
    else
        error("model not trained")
        return
    end
end

# trainmodel
function trainmodel(
    d::Data,
    t::Train,
    m::Model{T},
    x,
    y;
    force = false,
    runon = cpu,
) where {T}

    file = modeldir(string(d), string(t), string(m, ".bson"))
    isfile(file) && !force && return
    isfile(file) && rm(file)

    # create model
    model, pars = build_network(d, t) |> runon
    objective = build_loss(m)
    loss(x, y) = objective(x, y, model, pars)

    # create callback
    savefunc(c, x, y) = savemodel(c, d, t, m, model, x, y)

    cb = CallBack(;
        title = string(nameof(T), ": "),
        iters = t.iters,
        saveat = t.saveat,
        savefunc = savefunc,
    )

    # training
    Random.seed!(t.seed)
    batches = batch_provider(d, t, x, y)
    opt = t.optimiser(t.step)

    custom_train!(loss, pars, batches, opt; cb = cb)
    return
end

# run simulations
function run_simulations(
    Datasets,
    Trains,
    Models;
    runon = cpu,
    force = false,
)
    Datasets = isa(Datasets, Data) ? [Datasets] : Datasets
    Trains = isa(Trains, Train) ? [Trains] : Trains
    Models = isa(Models, Model) ? [Models] : Models

    for dataset in Datasets
        @info dataset
        train, valid, test = loaddata(dataset) |> runon

        for t in Trains, m in Models
            trainmodel(dataset, t, m, train...; force, runon)

            dict = loadmodel(dataset, t, m)
            update_results!(dataset, t, m, dict, train..., :train; runon)
            update_results!(dataset, t, m, dict, valid..., :valid; runon)
            update_results!(dataset, t, m, dict, test..., :test; runon)

            file = modeldir(string(dataset), string(t), string(m, ".bson"))
            BSON.bson(file, dict)
        end
    end
    return
end

function update_results!(
    d::Data,
    t::Train,
    m::Model,
    dict::Dict,
    x,
    y,
    key::Symbol;
    runon = cpu
)

    model0, pars0 = build_network(d, t; zero = true) |> runon
    for iter in keys(dict[:iterations])
        model = deepcopy(dict[:iterations][iter][:model]) |> runon
        pars = params(model)
        delete!(pars, model.b)
        loss = build_loss(m)

        get!(dict[:iterations][iter], key, Dict{Symbol, Any}())
        get!(dict[:iterations][iter][key], :targets, cpu(vec(y)))
        get!(dict[:iterations][iter][key], :scores, cpu(compute_scores(model, x)))
        get!(dict[:iterations][iter][key], :loss, cpu(loss(x, y, model, pars)))
        get!(dict[:iterations][iter][key], :loss_zero, cpu(loss(x, y, model0, pars0)))
    end
end
