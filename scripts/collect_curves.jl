using DrWatson
@quickactivate "AccuracyAtTopPrimal_oms"

include(srcdir("utilities.jl"))


# ------------------------------------------------------------------------------------------
# best parameters
# ------------------------------------------------------------------------------------------
df_valid = collect_metrics(; key = :valid)
df_test = collect_metrics(; key = :test)
df = find_best(df_valid, df_test, models, :tpr_at_top_mean; bymodelmetric = true)

# ------------------------------------------------------------------------------------------
# Not Hepmass
# ------------------------------------------------------------------------------------------
train_settings = Dict(
    :batchsize => 0,
    :iters => 1000,
    :optimiser => ADAM,
    :steplength => 0.01,
    :seed => 1,
)

map(string.([Ionosphere, Spambase, Gisette])) do d
    dataset_settings = Dict(
        :dataset => d,
        :posclass => 1,
    )

    table = df[df.dataset .== string(d), :]
    Model_Settings = map(eachrow(table)) do row
        model = modeltype(row.model)
        return Dict(:type => model, [key => row[key] for key in validkeys(model)]...)
    end
    collect_curves(dataset_settings, train_settings, Model_Settings)
end



# ------------------------------------------------------------------------------------------
# Hepmass
# ------------------------------------------------------------------------------------------
train_settings = Dict(
    :batchsize => 131250,
    :iters => 1000,
    :optimiser => ADAM,
    :steplength => 0.01,
    :seed => 1,
)

map(string.([Hepmass])) do d
    dataset_settings = Dict(
        :dataset => d,
        :posclass => 1,
    )

    table = df[df.dataset .== string(d), :]
    Model_Settings = map(eachrow(table)) do row
        model = modeltype(row.model)
        return Dict(:type => model, [key => row[key] for key in validkeys(model)]...)
    end
    collect_curves(dataset_settings, train_settings, Model_Settings)
end
