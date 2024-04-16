using NeuralVerification, LazySets, LazySets.Approximations
import NeuralVerification: Network, Layer, ReLU, Id, compute_output, ActivationFunction, get_bounds, forward_act
using Plots
using LinearAlgebra
import LazySets.Approximations.interval_hull

import Pkg; Pkg.add("ProbabilityBoundsAnalysis")
Pkg.add("PyPlot")
using ProbabilityBoundsAnalysis, PyPlot, IntervalArithmetic
import IntervalArithmetic: Interval
Pkg.add("Distributions")
using Distributions, Random
Pkg.add("DataFrames")
using DataFrames
Pkg.add("SplitApplyCombine")
using SplitApplyCombine



# initialize all pboxes with same number of focal elements and Normal law, possibly truncated 
function init_pbox_Normal(init_lb::Vector{Float64},init_ub::Vector{Float64},nb_focal_elem::Int64,truncate_focals::Bool)
    steps = parametersPBA.steps # saving the number of focal elements
    mean = (init_lb + init_ub) / 2.0
    a = 3.0
    dev = (init_ub - mean) / a
    ProbabilityBoundsAnalysis.setSteps(nb_focal_elem)
    pbox = normal.(mean,dev)
    ProbabilityBoundsAnalysis.setSteps(steps) # resetting the number of focal elements
    if (truncate_focals) # truncating to restrict the range to [init_lb_prop_1_2,init_ub_prop_1_2]
        for i in 1:length(pbox)
            pbox[i].u = max.(pbox[i].u,init_lb[i]) # left cut of the focal elements
            pbox[i].d = max.(pbox[i].d,init_lb[i]) # left cut of the focal elements
            pbox[i].d = min.(pbox[i].d,init_ub[i]) # right cut of the focal elements
            pbox[i].u = min.(pbox[i].u,init_ub[i]) # right cut of the focal elements
            pbox[i].bounded[1] = true # due to the cut, pbox is now bounded
            pbox[i].bounded[2] = true # due to the cut, pbox is now bounded
        end
    end
    return pbox
end


# initialize a vector of pboxes with different number of focal elements for each input component
function init_pbox_Normal(init_lb::Vector{Float64},init_ub::Vector{Float64},nb_focal_elem::Vector{Int64},truncate_focals::Bool)
    steps = parametersPBA.steps # saving the number of focal elements
    mean = (init_lb + init_ub) / 2.0
    a = 3.0
    dev = (init_ub - mean) / a
    pb = Vector{pbox}(undef, length(init_lb))
    for i in 1:length(pb)
        ProbabilityBoundsAnalysis.setSteps(nb_focal_elem[i])
        pb[i] = normal(mean[i],dev[i])
    end
    ProbabilityBoundsAnalysis.setSteps(steps) # resetting the default number of focal elements
    if (truncate_focals) # truncating to restrict the range to [init_lb_prop_1_2,init_ub_prop_1_2]
        for i in 1:length(pb)
            pb[i].u = max.(pb[i].u,init_lb[i]) # left cut of the focal elements
            pb[i].d = max.(pb[i].d,init_lb[i]) # left cut of the focal elements
            pb[i].d = min.(pb[i].d,init_ub[i]) # right cut of the focal elements
            pb[i].u = min.(pb[i].u,init_ub[i]) # right cut of the focal elements
            pb[i].bounded[1] = true # due to the cut, pbox is now bounded
            pb[i].bounded[2] = true # due to the cut, pbox is now bounded
        end
    end
    return pb
end



# Utilities to interpret a neural network in Pbox

function Relu(x::pbox)
    #print(x.u,"\n")
    #print(x.d,"\n")
    u = max.(0, x.u)
    d = max.(0, x.d)
    y = pbox(u,d)
    return y   
end

function pbox_affine_map_Invdep(M::Matrix{Float64},input::Vector{pbox},b::Vector{Float64})
    output = Vector{pbox}(undef, length(b))
        for i in 1:length(b)
            output[i] = b[i] 
            for j in 1:length(input)
                output[i] = convIndep(output[i], M[i,j]*input[j], op = +) 
                #print("output[i]=",output[i],"\n")
            end
        end
    return output
end

function pbox_affine_map(no_layer::Int64, layer::Layer, input::Vector{pbox}, input_is_indep::Bool)
    if (no_layer > 1 || !input_is_indep)
        output = layer.weights * input + layer.bias; 
    else
        output = pbox_affine_map_Invdep(layer.weights,input,layer.bias)
    end
    return output
end

function pbox_act_map(act::ActivationFunction, input::Vector{pbox})
    if (act == Id())
        return input
    elseif (act == ReLU())
        return Relu.(input)
    else 
        print("warning, ",act," not yet implemented")
        return input
    end
end

function pbox_approximate_nnet(nnet::Network, input::Vector{pbox}, input_is_indep::Bool)
    bounds = Vector{Vector{pbox}}(undef, length(nnet.layers) + 1)
    bounds[1] = input
    for i in 1:length(nnet.layers)
        temp = pbox_affine_map(i,nnet.layers[i],bounds[i], input_is_indep)
        bounds[i+1] = pbox_act_map(nnet.layers[i].activation,temp)
    end

    return bounds[length(nnet.layers)+1]
end





######################################################## Sampling facilities ######################################################## 

# a random sample is a random cut
rand(F :: pbox, N :: Int) = cut.([F], Base.rand(N))
#rand(F :: vector{pbox}, N :: Int) = cut.(F, Base.rand(N))

function compute_samples(nnet::Network, input::Vector{pbox},nb_samples::Int)
    
    sampled_inputs = fill(zeros(nb_samples), length(input))

    sampled_interval_inputs = rand.(input,nb_samples)
    for i in 1:length(input)
        sampled_inputs[i] = Base.rand.(sampled_interval_inputs[i])
    end
    sampled_inputs = invert(sampled_inputs)   # invert from package SplitApplyCombine https://github.com/JuliaData/SplitApplyCombine.jl
    #print("sampled_inputs=",sampled_inputs,"\n")

    # sampled_outputs = [] 
    # for k in 1:nb_samples
    #     push!(sampled_outputs, compute_output(nnet,sampled_inputs[k]))
    # end

    sampled_outputs = Array{Float64}(undef, nb_samples, length(nnet.layers[length(nnet.layers)].bias))
    for k in 1:nb_samples
        sampled_outputs[k,:] = compute_output(nnet,sampled_inputs[k])
    end
    #print("sampled_outputs=",sampled_outputs,"\n")  

    return sampled_outputs
end
