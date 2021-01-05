using DrWatson
@quickactivate "AccuracyAtTopPrimal_oms"

include(srcdir("utilities.jl"))

# ------------------------------------------------------------------------------------------
# TopPush convergence
# ------------------------------------------------------------------------------------------
model_settings = Dict(:type => PatMatNP, :surrogate => hinge, :λ => 0, :K => 6, :τ => 0.01, :β => 1)
k = 6

plot_convergence(model_settings; k = k)
