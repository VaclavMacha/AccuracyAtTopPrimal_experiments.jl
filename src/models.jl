using Flux
using AccuracyAtTopPrimal

# -------------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------------
sqsum(x) = sum(abs2, x)

validkeys(::Union{Type{PatMat}, Type{PatMatNP}}) = [:τ, :surrogate, :β, :λ]

function build_threshold(T::Union{Type{PatMat}, Type{PatMatNP}}, d::Dict)
    return T(d[:τ], x -> d[:surrogate](x, d[:β]))
end

model_name(T::Union{Type{PatMat}, Type{PatMatNP}}, d::Dict) = "$(T)($(d[:τ]),$(d[:β]))"

function validkeys(::Union{Type{Grill}, Type{GrillNP}, Type{τFPL}, Type{TopMean}})
    return [:τ, :surrogate, :λ]
end

function build_threshold(
    T::Union{Type{Grill}, Type{GrillNP}, Type{τFPL}, Type{TopMean}},
    d::Dict
)

    return T(d[:τ])
end

function model_name(
    T::Union{Type{Grill}, Type{GrillNP}, Type{τFPL}, Type{TopMean}},
    d::Dict
)
    return "$(T)($(d[:τ]))"
end

validkeys(::Type{TopPush}) = [:surrogate, :λ]
build_threshold(::Type{TopPush}, ::Dict) = TopPush()
model_name(T::Type{TopPush}, ::Dict) = "$(T)"
validkeys(::Type{TopPushK}) = [:K, :surrogate, :λ]
build_threshold(::Type{TopPushK}, d::Dict) = TopPushK(d[:K])
model_name(T::Type{TopPushK}, d::Dict) = "$(T)($(d[:K]))"

function build_loss(type, d::Dict)
    @unpack surrogate, λ = d
    thres = build_threshold(type, d)

    function loss(x, y, model, pars)
        return loss(y, model(x), pars)
    end

    function loss(y, s, pars)
        t = threshold(thres, y, s)
        return fnr(y, s, t, surrogate) + eltype(s)(λ) * sum(sqsum, pars)
    end
    return loss
end

function build_loss(type::Union{Type{Grill}, Type{GrillNP}}, d::Dict)
    @unpack surrogate, λ = d
    thres = build_threshold(type, d)

    function loss(x, y, model, pars)
        return loss(y, model(x), pars)
    end

    function loss(y, s, pars)
        t = threshold(thres, y, s)
        return fnr(y, s, t, surrogate) + fpr(y, s, t, surrogate) + eltype(s)(λ) * sum(sqsum, pars)
    end
    return loss
end

# -------------------------------------------------------------------------------
# Build from dictionary
# -------------------------------------------------------------------------------
isvalid(T, d::Dict) = all(haskey(d, key) for key in validkeys(T))

function dict_list_valid(dict::Dict)
    d = deepcopy(dict)
    delete!(d, :type)
    model_pars = Dict[]

    for T in dict[:type]
        isvalid(T, d) || continue
        dd = Dict(key => d[key] for key in validkeys(T))
        dd[:type] = T
        if T <: Union{TopPushK, PatMat, PatMatNP}
            dd[:λ] = 0.001
        end
        append!(model_pars, dict_list(dd))
    end
    return model_pars
end
