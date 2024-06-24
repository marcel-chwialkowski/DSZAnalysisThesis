# DSZAnalysis
Probability Bounds Analysis of Neural Networks in Julia.

Code (see notebooks in the examples directory) to reproduce the examples of:
[A Zonotopic Dempster-Shafer Approach to the Quantitative Verification of Neural Networks, Eric Goubault, Sylvie Putot, 2024](https://hal.science/hal-04546350).


## Dependencies and installation
Builds in particular on:
- the [LazySets](https://juliareach.github.io/LazySets.jl/) package for calculus with convex sets 
- the [ProbabilityBoundsAnalysis](https://github.com/AnderGray/ProbabilityBoundsAnalysis.jl) package for p-boxes analysis relying on Dempster-Shafer interval structures  
- the [NeuralVerification](https://sisl.github.io/NeuralVerification.jl/latest/functions/) package

### Using docker

Retrieve the DSZAnalysis directory and run in this directory the command:

```docker build . -f Dockerfile```

An image ```sha256:...``` is built, which you can execute by:

```docker run -it --name FM sha256:...```

This should open Julia through docker, in which you can run the examples by:
- ```include("ToyExample.jl")```
- ```include("ACASXu.jl")```
- ```include("RocketLander.jl")```

### Using your own Julia installation

These files been executed with Julia 1.8.1 and the following Julia packages and versions:

```Using Pkg
Pkg.add(name="ProbabilityBoundsAnalysis", version="0.2.11"); 
Pkg.add(url="https://github.com/sisl/NeuralVerification.jl"); 
Pkg.add(name="LazySets", version="2.11.3"); 
Pkg.add(name="IntervalArithmetic",version="0.20.9"); 
Pkg.add(name="Distributions", version = "0.25.107"); 
Pkg.add(name="DataFrames", version="1.6.1"); 
Pkg.add(name="SplitApplyCombine",version="1.2.3"); 
Pkg.add(name="PyPlot", version = "2.11.2"); 
Pkg.add(name="Plots", version="1.39.0 »)
```
