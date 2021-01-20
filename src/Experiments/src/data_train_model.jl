# ------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------
struct Data{N<:Name}
    seed::Int
    shuffle::Bool
    poslabels::Int
    batchsize::Int

    function Data(
        N;
        seed = 1234,
        shuffle = true,
        poslabels = 0,
        batchsize = 0,
    )
        return new{N}(seed, shuffle, poslabels, batchsize)
    end
end

function Base.show(io::IO, d::Data{N}) where {N}
    pars = [:seed, :shuffle,:poslabels]
    d.batchsize != 0 && push!(pars, :batchsize)
    print(io, nameof(N), NamedTuple{(pars...,)}(getfield.(Ref(d), pars)))
    return
end

function todict(d::Data{N}) where {N}
    return Dict(
        :dataset => nameof(N),
        :seed => d.seed,
        :shuffle => d.shuffle,
        :poslabels => d.poslabels,
        :batchsize => d.batchsize,
    )
end

function loaddata(d::Data{N}) where {N}
    data = DatasetProvider.Dataset(
        N;
        seed = d.seed,
        shuffle = d.shuffle,
        asmatrix = true,
        poslabels = d.poslabels,
        binarize = true,
    )

    return  reshape_dataset.(DatasetProvider.load(TrainValidTest(), data))
end

# Reshape to Flux style
reshape_dataset(data::Tuple) = (reshape_samples(data[1]), reshape_labels(data[2]))

reshape_labels(y) = y
reshape_labels(y::AbstractVector) = Array(reshape(y, 1, :))

reshape_samples(x; T = Float32) = T.(x)
reshape_samples(x::AbstractMatrix; T = Float32) = Matrix{T}(x')
reshape_samples(x::AbstractArray; T = Float32) = Array{T}(reshape(x, :, size(x)[end]))

# ------------------------------------------------------------------------------------------
# Train
# ------------------------------------------------------------------------------------------
Base.@kwdef struct Train
    seed::Int = 1
    iters::Int = 1000
    saveat::Int = 200
    optimiser = ADAM
    step::Float64 = 0.01
end

function Base.show(io::IO, m::Train)
    @unpack seed, iters, saveat, optimiser, step = m
    print(io, "Train", (; seed, iters, saveat, optimiser, step))
    return
end

function todict(t::Train)
    return Dict(
        :seed => t.seed,
        :iters => t.iters,
        :saveat => t.saveat,
        :optimiser => nameof(t.optimiser),
        :step => t.step,
    )
end

function build_network(::Data{N}, t::Train; zero::Bool = false) where {N}
    Random.seed!(t.seed)
    model = Dense(prod(DatasetProvider.nattributes(N)), 1)
    pars = params(model)
    delete!(pars, model.b)
    if zero
        for p in pars
            p .= 0
        end
    end
    return model, pars
end

# ------------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------------
struct Model{T<:AbstractThreshold}
    pars
    surr
    λ

    Model(T, pars, surr, λ) = new{T}(pars, surr, λ)
end

function Model(T::Type{<:AbstractThreshold}; λ = 0.001, surrogate = hinge, kwargs...)
    d = Dict(:pars => parameters(T; surrogate, kwargs...), :λ => λ, :surr => surrogate)

    return map(dict_list(d)) do dict
        @unpack pars, surr, λ = dict
        return Model(T, pars, surr, λ)
    end
end

function Base.show(io::IO, m::Model{T}) where {T}
    return print(io, nameof(T), (; m.pars..., surrogate = nameof(m.surr), λ = m.λ))
end

function todict(m::Model{T}) where {T}
    return Dict(
        :model => nameof(T),
        :pars => ntuple2dict(m.pars),
        :surr => nameof(m.surr),
        :λ => m.λ,
    )
end

# Thresholds
build(T::Union{Type{PatMat}, Type{PatMatNP}}, τ, surr, β) = T(τ, x -> surr(x, β))
validkeys(::Union{Type{PatMat}, Type{PatMatNP}}) = (:τ, :surrogate, :β)

build(T::Union{Type{Grill}, Type{GrillNP}, Type{τFPL}, Type{TopMean}}, τ) = T(τ)
validkeys(::Union{Type{Grill}, Type{GrillNP}, Type{τFPL}, Type{TopMean}}) = (:τ, )

build(T::Type{TopPush}) = T()
validkeys(::Type{TopPush}) = tuple()

build(T::Type{TopPushK}, K) = T(K)
validkeys(::Type{TopPushK}) = (:K, )


function parameters(T::Type{<:AbstractThreshold}; kwargs...)
    keys = validkeys(T)
    vals = getindex.(Ref(kwargs), keys)

    return map(dict_list(Dict(tuple.(keys, vals)))) do dict
        return NamedTuple{keys}(getindex.(Ref(dict), keys))
    end
end

# Loss functions
sqsum(x) = sum(abs2, x)

function build_loss(m::Model{T}) where {T}
    @unpack surr, λ = m
    thres = build(T, m.pars...)

    loss(x, y, model, pars) = loss(y, model(x), pars)

    function loss(y, s, pars)
        t = threshold(thres, y, s)
        return fnr(y, s, t, surr) + eltype(s)(λ) * sum(sqsum, pars)
    end
    return loss
end

function build_loss(m::Model{T}) where {T<:Union{Type{Grill}, Type{GrillNP}}}
    @unpack surr, λ = m
    thres = build(T, m.pars...)

    loss(x, y, model, pars) = loss(y, model(x), pars)

    function loss(y, s, pars)
        t = threshold(thres, y, s)
        return fnr(y, s, t, surr) + fpr(y, s, t, surr) + eltype(s)(λ) * sum(sqsum, pars)
    end
    return loss
end
