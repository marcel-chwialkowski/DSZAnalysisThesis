using Random
using ProbabilityBoundsAnalysis
using Distributions
using IntervalArithmetic
using PyPlot
using Base.Threads
include("../src/sampling_functions.jl")

function gaussian_samples(input_boxes, num_samples, epsilon)
    #input box is described by a tuple ([lower_mean, upper_mean], std)
    #each covering should have a precision of at least epsilon
    psamples = [[] for i in 1:length(input_boxes)]
    
    #generate samples for each input box
    for i in 1:length(input_boxes)
        input_box = input_boxes[i]
        lower_mean = input_box[1][1]
        upper_mean = input_box[1][2]
        std = input_box[2]
        num_divisions = ceil((upper_mean - lower_mean) /(2 * epsilon))
        means = [lower_mean + i * (upper_mean - lower_mean) / num_divisions for i in 0:num_divisions-1]
        push!(means, upper_mean)

        for mean in means
            dist = Normal(mean, std)
            samples = Distributions.rand(dist, num_samples)
            push!(psamples[i], samples)
        end
    end
    return psamples
end

#slow 
function propagate_gaussians(psamples, network, num_samples, x, network_name="network")
    combinations = collect(Iterators.product((1:length(lst) for lst in psamples)...))
    for combo in combinations
        n = length(combo)
        outp = []
        
        #for each of the samples
        for i in 1:num_samples
            inp = []
            #in each of the dimensions
            for j in 1:n
                push!(inp, psamples[j][combo[j]][i])
            end
            push!(outp, compute_output(network, inp))
        end
        push!(outps, outp)
    end
    return outps
end

#mutlithreaded - faster
function propagate_gaussians_multhread(psamples, network, num_samples)
    combinations = collect(Iterators.product((1:length(lst) for lst in psamples)...))
    num_combinations = length(combinations)
    outps = Vector{Vector{Vector{Float64}}}(undef, num_combinations)
    
    
    start_time = time()
    println("Total combinations to process: ", num_combinations)
    
    Threads.@threads for idx in 1:num_combinations
        combo = combinations[idx]
        n = length(combo)
        outp = Vector{Vector{Float64}}(undef, num_samples)
        
        for i in 1:num_samples
            inp = Vector{Float64}(undef, n)
            for j in 1:n
                inp[j] = psamples[j][combo[j]][i]
            end
            outp[i] = compute_output(network, inp)
        end
        outps[idx] = outp
        
    end
    println("Time: ", (time() - start_time)/60, " minutes")    
    return outps
end

#computes approximates for property satisfaction GPB
function check_property(outps, W, y)
    #W is a matrix, y is a vector
    #Wx < y
    init = [1.0,0.0]
    for outp in outps
        right = 0
        wrong = 0
        for sample in outp
            if all(W * sample .< y)
                right += 1
            else
                wrong += 1
            end
        end
        proba = right / (wrong + right)
        init[1] = min(init[1], proba)
        init[2] = max(init[2], proba)
    end
    return init
end
