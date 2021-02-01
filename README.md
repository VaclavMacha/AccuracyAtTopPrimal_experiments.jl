# AccuracyAtTop_experiments.jl

This repository is a complementary material to our paper *General Framework for Binary Classification on Top Samples*. This paper was submitted to the **Optimization Methods and Software**.

# Running the codes

All required packages are listed in the `Project.toml` file. Before running any of provided scripts, go to the project directory and from the Pkg REPL run the following commands

```julia
(@v1.5) pkg> activate .
(AccuracyAtTop_experiments.jl) pkg> instantiate
```

For more information see the [manual.](https://julialang.github.io/Pkg.jl/v1/environments/#Using-someone-else's-project-1) Certain problems can occur because we use unregistered packages. In case of problems, do not hesitate to contact us.

The repository consists of two main parts. The module `Experiments` in the `src` folder contains all necessary codes for running and evaluating experiments. The `scripts` folder contains scripts for running and evaluating experiments with concrete settings.

The core of the codes is in the separate repository [AccuracyAtTopPrimal.jl](https://github.com/VaclavMacha/AccuracyAtTopPrimal.jl).
