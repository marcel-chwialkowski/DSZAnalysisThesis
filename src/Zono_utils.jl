# This file contains utilities for set-based (zonotopic) analysis, relying on functions from the LaySets and NeuralVerification packages

#  applying the activation layer on zonotopes
function zono_act_map(act::ActivationFunction, input::Zonotope)
    
    if (act == Id())
        return input
    elseif (act == ReLU())
        return overapproximate(Rectification(input), Zonotope)
    else 
        print("warning, ",act," not yet implemented")
        return input
    end

end



# propagating zonotope through network (starting from a hypperrectangle)
function zono_approximate_nnet(nnet::Network, input::Hyperrectangle)
    #bounds = Vector{PAF}(undef, length(nnet.layers) + 1)
    res_zono = input
    
    for i in 1:length(nnet.layers)
        res_zono = affine_map(nnet.layers[i].weights,res_zono,nnet.layers[i].bias)
        res_zono = zono_act_map(nnet.layers[i].activation,res_zono)
    end

    return res_zono  
end


# set-based (non probabilistic) reachability analysis of a neural network (for comparisons - not actually used in the DSZ Analysis)
function Classical_Analysis(nnet::Network, input::Vector{Interval{Float64}}, plot::Bool, exact::Bool)
    X = Hyperrectangle(low=inf.(input),high=sup.(input))

    # result with Hyperrectangles
    result_bounds = get_bounds(nnet,X)
    int_hull = result_bounds[length(result_bounds)]
    print("Interval result =",low(int_hull),high(int_hull),"\n")

    # result with zonotopes (AI2 analysis)
    problem = Problem(nnet, X, X)
    result_zono = NeuralVerification.solve(Ai2z(),problem)
    #print(result_zono.reachable)
    int_hull = box_approximation.(result_zono.reachable)
    print("AI2z result =",low.(int_hull),high.(int_hull),"\n")

    # exact reachability
    if (exact)
        result_exact = NeuralVerification.solve(ExactReach(),problem)
        exact_int_hull = box_approximation.(result_exact.reachable)
        #print(exact_int_hull)
    end

    if (plot)
        Plots.plot(result_bounds[length(result_bounds)], label = "Box result")
        #plot!(zono_result[length(result_bounds)], label = "Zono result")
        if (exact)
            plot!(exact_int_hull, label = "exact_int_hull")
        end
        #plot!(result_exact.reachable, label = "exact result")
        plot!(result_zono.reachable, label = "Zono result")
    end
end