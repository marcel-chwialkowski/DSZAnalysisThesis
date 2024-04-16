

#########################################################  Probabilistic zonotopes: use zonotopic analysis and intrepret at the end as a probabilistic zonotopes where we assign to new symbols any cdf defined on [-1,1]

# rescale between [-1,1] to build affine forms / zonotopes 
function rescale(input::pbox)
    input_bounds = range(input)
    if (diam(input_bounds) > 0)
       return 2*(input - mid(input_bounds))/diam(input_bounds)
    else
       return input
    end
 end
 

 function affpbox_approximate_nnet(nnet::Network, input::Vector{pbox}, input_is_indep::Bool)
 
    # classical zonotopic analysis from the range of the support of input distributions
    input_bounds = range.(input)
    zz = zono_approximate_nnet(nnet, Hyperrectangle(low=inf.(input_bounds),high=sup.(input_bounds))) 

    c = LazySets.center(zz)
    A = LazySets.genmat(zz)
 
    # interpreting output as a vector of pboxes
    Aeps = A[:,1:length(input)]
    Aeta = A[:,length(input)+1:length(A[1,:])]   
 
    v1 = rescale.(input) # the input pboxes must be rescaled between [-1,1]
    v2 = makepbox(-1..1) * ones(length(A[1,:])-length(input)) # the new noise symbols (etas) have unknown distribution between -1,1
    
    if (input_is_indep)
        pzaff = pbox_affine_map_Invdep(Aeps,v1,c)
    else
        pzaff = Aeps*v1 + c
    end
    
    pzaffres = pzaff + Aeta*v2
    
    return pzaffres
 
 end


# instead of just producing the resulting pbox, evaluates the condition on the zonotopic form before evaluating it in pbox
 function affpbox_approximate_nnet_and_condition(nnet::Network, input::Vector{pbox}, input_is_indep::Bool, mat_spec::Matrix{Float64})

    # classical zonotopic analysis from the range of the support of input distributions
    input_bounds = range.(input)
    zz = zono_approximate_nnet(nnet, Hyperrectangle(low=inf.(input_bounds),high=sup.(input_bounds))) 

    # applying the matrix specification
    zz = concretize(mat_spec * zz) 
 
    c = LazySets.center(zz)
    A = LazySets.genmat(zz)
 
    # interpreting output as a vector of pboxes
    Aeps = A[:,1:length(input)]
    Aeta = A[:,length(input)+1:length(A[1,:])]   
 
    # vector of p-boxes
    v1 = rescale.(input) # the input pboxes must be rescaled between [-1,1]
    v2 = makepbox(-1..1) * ones(length(A[1,:])-length(input)) # the new noise symbols (etas) have unknown distribution between -1,1
    
    if (input_is_indep)
        pzaff = pbox_affine_map_Invdep(Aeps,v1,c)
    else
        pzaff = Aeps*v1 + c
    end
    pzaffres = pzaff + Aeta*v2
    
    print("probability of unsafety : Ay < 0 is ","\n")
    print("pzaffres[1] <= 0: ", pzaffres[1] <= 0, "\n")
    print("pzaffres[2] <= 0: ", pzaffres[2] <= 0, "\n")
    print("pzaffres[3] <= 0: ", pzaffres[3] <= 0, "\n")
    print("pzaffres[4] <= 0: ", pzaffres[4] <= 0, "\n")
 end




