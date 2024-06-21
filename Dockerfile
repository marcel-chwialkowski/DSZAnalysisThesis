FROM julia:1.8.1

MAINTAINER Eric Goubault and Sylvie Putot

WORKDIR /home/DSZAnalysis/examples
COPY ../examples /home/DSZAnalysis/examples
COPY ../src /home/DSZAnalysis/src

RUN julia -e 'using Pkg; Pkg.add(name="ProbabilityBoundsAnalysis", version="0.2.11"); Pkg.add(url="https://github.com/sisl/NeuralVerification.jl"); Pkg.add(name="LazySets", version="2.11.3"); Pkg.add(url="https://github.com/JuliaInterop/NBInclude.jl.git"); Pkg.add(name="IntervalArithmetic",version="0.20.9"); Pkg.add(name="Distributions", version = "0.25.107"); Pkg.add(name="DataFrames", version="1.6.1"); Pkg.add(name="SplitApplyCombine",version="1.2.3")'

RUN julia -e 'using NBInclude; nbexport("/home/DSZAnalysis/examples/ToyExample.jl", "/home/DSZAnalysis/examples/ToyExample.ipynb"); nbexport("/home/DSZAnalysis/examples/RocketLander.jl", "/home/DSZAnalysis/examples/RocketLander.ipynb"); nbexport("/home/DSZAnalysis/examples/ACASXu.jl", "/home/DSZAnalysis/examples/ACASXu.ipynb")' 









