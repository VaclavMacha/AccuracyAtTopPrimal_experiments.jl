using DrWatson
@quickactivate "AccuracyAtTopPrimal_oms"

include(srcdir("utilities.jl"))

# ------------------------------------------------------------------------------------------
# Not Hepmass
# ------------------------------------------------------------------------------------------
Dataset_Settings = Dict(
    :dataset => [Ionosphere, Spambase, Gisette],
    :posclass => 1,
)

Train_Settings = Dict(
    :batchsize => 0,
    :iters => 100,
    :optimiser => Descent,
    :steplength => 0.01,
    :seed => 1,
)

Model_Settings = Dict(
    :type => [PatMat, PatMatNP, TopPush, TopPushK, τFPL, TopMean, Grill, GrillNP],
    :surrogate => hinge,
    :τ => 0.01,
    :K => 20,
    :β => 1,
    :λ => 1e-4,
)

run_benchmark(Dataset_Settings, Train_Settings, Model_Settings; runon = cpu)

# ------------------------------------------------------------------------------------------
# Hepmass
# ------------------------------------------------------------------------------------------
Dataset_Settings = Dict(
    :dataset => Hepmass,
    :posclass => 1,
)

Train_Settings = Dict(
    :batchsize => 131250,
    :iters => 100,
    :optimiser => Descent,
    :steplength => 0.01,
    :seed => 1,
)

run_benchmark(Dataset_Settings, Train_Settings, Model_Settings; runon = cpu)
