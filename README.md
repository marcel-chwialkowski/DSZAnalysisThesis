# DSZAnalysis
Probability Bounds Analysis of Neural Networks in Julia.

Code (see notebooks in the examples directory) to reproduce the examples of:
[A Zonotopic Dempster-Shafer Approach to the Quantitative Verification of Neural Networks, Eric Goubault, Sylvie Putot, 2024](https://hal.science/hal-04546350).


## Dependencies
Builds on:
- the [LazySets](https://juliareach.github.io/LazySets.jl/) package for calculus with convex sets 
- the [ProbabilityBoundsAnalysis](https://github.com/AnderGray/ProbabilityBoundsAnalysis.j) package for p-boxes analysis relying on Dempster-Shafer interval structures  
- the [NeuralVerification](https://sisl.github.io/NeuralVerification.jl/latest/functions/) package 
