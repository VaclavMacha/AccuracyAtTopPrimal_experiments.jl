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
    :iters => 1000,
    :optimiser => ADAM,
    :steplength => 0.01,
    :seed => 1,
)

Model_Settings = Dict(
    :type => [PatMat, PatMatNP, TopPush, TopPushK, τFPL, TopMean, Grill, GrillNP],
    :surrogate => hinge,
    :τ => [0.01, 0.05],
    :K => [1, 3, 5, 10, 15, 20],
    :β => [10, 1, 0.1, 0.01, 0.001, 0.0001],
    :λ => [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
)

run_simulations(Dataset_Settings, Train_Settings, Model_Settings; runon = cpu)
run_evaluation(Dataset_Settings)

# ------------------------------------------------------------------------------------------
# Hepmass
# ------------------------------------------------------------------------------------------
Dataset_Settings = Dict(
    :dataset => Hepmass,
    :posclass => 1,
)

Train_Settings = Dict(
    :batchsize => 131250,
    :iters => 1000,
    :optimiser => ADAM,
    :steplength => 0.01,
    :seed => 1,
)

run_simulations(Dataset_Settings, Train_Settings, Model_Settings; runon = cpu)
run_evaluation(Dataset_Settings)
