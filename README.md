# DSZAnalysis
Probability Bounds Analysis of Neural Networks in Julia.

Code (see notebooks in the examples directory) to reproduce the examples of:
[A Zonotopic Dempster-Shafer Approach to the Quantitative Verification of Neural Networks, Eric Goubault, Sylvie Putot, 2024](https://hal.science/hal-04546350).


## Dependencies and installation
Builds on:
- the [LazySets](https://juliareach.github.io/LazySets.jl/) package for calculus with convex sets 
- the [ProbabilityBoundsAnalysis](https://github.com/AnderGray/ProbabilityBoundsAnalysis.j) package for p-boxes analysis relying on Dempster-Shafer interval structures  
- the [NeuralVerification](https://sisl.github.io/NeuralVerification.jl/latest/functions/) package

### Using docker

Retrieve the DSZAnalysis directory and run in this directory the command:

```docker build . -f Dockerfile```

An image ```sha...``` is built, which you can execute by:

```docker run -it --name FM sha...```

This should open Julia in which you can run the examples by:
- ```include("ToyExample.jl")```
- ```include("ACASXu.jl")```
- ```include("RocketLander.jl")```
