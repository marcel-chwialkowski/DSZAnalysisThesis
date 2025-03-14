# This file implements the neural network analysis with Dempster Shafer Zonotope Structures (DSZ), as described in Section 5 of 
# the FM 2024 paper "A Zonotopic Dempster-Shafer Approach to the Quantitative Verification of Neural Networks"
 

mutable struct DSZ
    flow :: Vector{Zonotope}            # focal elements (length nb_steps)
    n :: Int64                          # input dimension (number of noise symbols)
    p :: Int64                          # output dim (nb of neurons)
    nb_steps :: Int64                   # nb of focal elements
end 

# vp: input pbox
# x: output of the function: assignment as a cartesian product of the focal elements of the n inputs 
# n: input dimension (number of noise symbols)
# level: tuple (l1,...,ln) giving which focal element of which input is assigned
# index: global index in x
function assign_inputs!(vp :: Vector{pbox}, x :: Array{Interval}, n::Int64 , level::Vector{Int64}, index::Int64)
    #print("level=",level,"\n","length(vp[n].u=",length(vp[n].u),"\n")
    index = index + 1 #index = global_id(level)
    #print("index = ",index,"\n")   
    for i in 1:n
        x[index,i] = interval(vp[i].u[level[i]],vp[i].d[level[i]])
    end
    if (level[n] != length(vp[n].u))
        level[n] = level[n] + 1
        assign_inputs!(vp, x , n, level, index)
    else
        level[n] = 1
        k = n-1
        while (k >= 1 && level[k] == length(vp[k].u))
            level[k] = 1
            k = k-1
        end
        if (k > 0)
            level[k] = level[k] + 1
            assign_inputs!(vp, x , n, level, index)
        end
    end
end


# building a DSZ from the vector of pbox/DSI for an input vector with independent components 
function generate_DSZ(vp :: Vector{pbox})
    n = length(vp) # input dimension (number of noise symbols)
    p = n  # we build the DSZ for (x_1, ... x_n)

    # number of zonotopic focal elements (assuming the input components may not have the same nb of focal elements)
    nb_steps = length(vp[1].u)
    for i in 2:n
        nb_steps = nb_steps * length(vp[i].u)
    end
    # print("Number of zonotopic focal elements=",nb_steps,"\n")

    
    temp =  collect(Base.product(interval.(vp[1].u,vp[1].d),interval.(vp[2].u,vp[2].d)))
    for i in 3:n
        temp = collect(Base.product(temp,interval.(vp[i].u,vp[i].d)))
        temp = collect.(Iterators.flatten.(vec(temp)))
    end
    if n>=3
        c = temp #collect.(Iterators.flatten.(vec(temp)))
    else
        c = collect.(Iterators.flatten.(vec(temp)))
    end
    

    flow2= Vector{Hyperrectangle}(undef,nb_steps)
    i = 1
    for k in c
        flow2[i] = Hyperrectangle(low=inf.(k),high=sup.(k))
        i = i+1
    end

    return DSZ(flow2, n, p, nb_steps)
end



# Building pbox from vector of focal elements
function makepbox2(focal_el)
     x_lo = getfield.(focal_el, :lo);
     x_hi = getfield.(focal_el, :hi);
     u = sort(x_lo) 
     d = sort(x_hi)
     return pbox(u, d)
 end


# converting DSZ to vector of pbox
function DSZ_to_pbox(dsz :: DSZ, is_bounded::Bool)
    vp = Vector{pbox}(undef,dsz.p)

    # number of focal elements
    #save_steps = parametersPBA.steps
    ProbabilityBoundsAnalysis.setSteps(dsz.nb_steps)

    c = LazySets.center.(dsz.flow)
    A = LazySets.genmat.(dsz.flow)
   
    aff = Array{Interval}(undef,dsz.nb_steps,dsz.p)

    # concretization of each zonotope as a vector of intervals
    for i in 1:dsz.nb_steps 
        B = A[i]
        v = interval(-1,1) * ones(length(B[1,:])) 
        aff[i,:] = A[i]*v + c[i]
    end

    temp= Array{Interval}(undef,dsz.nb_steps,1)

    for k in 1:dsz.p
        temp = aff[:,k] 
        vp[k] = makepbox2(temp)
        if (is_bounded)
            vp[k].bounded[1] = true
            vp[k].bounded[2] = true
        end
    end
    
    return vp
end



#Define the cdf function on a vector of Pboxes (and interpret element-wise)
function vect_cdf(s :: Vector{pbox}, vx::Vector{Float64})
        res = Vector{Interval}(undef,length(s))
        for i in 1:length(s)
            d = s[i].d; u = s[i].u; n = s[i].n;
            bounded = s[i].bounded;
            x = vx[i]
    
            if x < u[1];
                res[i] =  interval(0,1/n) * (1-bounded[1]);
            elseif x >= d[end];
                if bounded[2]; 
                    res[i] = 1; 
                else
                    res[i] =  interval((n-1)/n, 1);
                end
            else
                indUb = sum(u .<= x)/n;
                indLb = sum(d .<= x)/n;
    
                res[i] = interval(indLb, indUb)
            end
        end
        return res
end

# Define cdf function for vector vx
function dsz_cdf(dsz :: DSZ, vx::Vector{Float64})

    c = LazySets.center.(dsz.flow)
    A = LazySets.genmat.(dsz.flow)
   
    aff = Array{Interval}(undef,dsz.nb_steps,dsz.p)
    # remplacer ci-dessous simplement par le range des zonotopes ? 
    sumUb = 0.0
    sumLb = 0.0
    for i in 1:dsz.nb_steps 
        B = A[i]
        v = interval(-1,1) * ones(length(B[1,:])) 
        aff[i,:] = A[i]*v + c[i]
        
        valUb = true
        valLb = true
        for k in 1:dsz.p
            if (left(aff[i,k]) > vx[k])
                valUb = false
            end
            if (right(aff[i,k]) > vx[k])
                valLb = false
            end
        end
        if (valUb)
            sumUb = sumUb + 1
        end
        if (valLb)
            sumLb = sumLb + 1
        end 
    end
    #println("aff=",aff)

    # the focal elements are not sorted so the treatment of the unbounded focal element case has to be slightly different
    # TODO. probably to be fixed - the number of unbounded focal elements on the PAF should not be 1 but actually depends on the discretization
    # if ((sumLb ==paf.nb_steps) && is_bounded)
    #     sumLb = sumLb-1
    # end
    # if ((sumUb ==0) && is_bounded)
    #     sumUb = sumUb+1
    # end
    indUb = sumUb/dsz.nb_steps;
    indLb = sumLb/dsz.nb_steps;

    return interval(indLb, indUb)

end

# Define contribution to cdf function for vector vx for one DSZ focal element
function dsz_focal_cdf(input :: Zonotope, vx::Vector{Float64}) 
    valUb = true
    valLb = true
    sumUb = 0.0
    sumLb = 0.0
    for k in 1:length(vx)
        if (LazySets.low(input, k) > vx[k])
            valUb = false
        end
        if (LazySets.high(input, k) > vx[k])
            valLb = false
        end
    end
    if (valUb)
        sumUb = 1.0
    end
    if (valLb)
        sumLb = 1.0
    end 
    return interval(sumLb, sumUb)
end





# From an input vector of pbox, builds the DSZ and propagates it through the neural network; finally, evaluate safety condition on the result
function dsz_approximate_nnet_and_condition_nostorage(nnet::Network, input::Vector{pbox}, mat_spec::Matrix{Float64},rhs_spec::Vector{Float64}, print_pbox=false)
    # dsz_in = generate_DSZ(input)
    vp = input
    n = length(vp) # input dimension (number of noise symbols)
    p = n  # we build the DSZ for (x_1, ... x_n)


    nb_steps = length(vp[1].u)
    for i in 2:n
        nb_steps = nb_steps * length(vp[i].u)
    end
    
    # print("Number of zonotopic focal elements=",nb_steps,"\n")    
    
    iter = Vector{UnitRange{Int64}}(undef,n)
    for i in 1:n
        iter[i] = UnitRange{Int}(1,length(vp[i].u))
    end
    c1 = vec([collect(x) for x in Iterators.product(iter...)])
    
    tab_low = Vector{Float64}(undef,n)
    tab_high = Vector{Float64}(undef,n)

    proba = interval(0.0, 0.0)
    vec_proba = Vector{Interval}(undef,length(rhs_spec)+1)
    for i in 1:length(rhs_spec)+1
        vec_proba[i] = interval(0.0, 0.0)
    end
    for index in c1
        for i in 1:n
            tab_low[i] = vp[i].u[index[i]]
            tab_high[i] = vp[i].d[index[i]]
        end
        input_zono = Hyperrectangle(low=tab_low,high=tab_high)
        res_zono = zono_approximate_nnet(nnet, input_zono)
        spec_zono = concretize(mat_spec*res_zono)  # remplacer par linear_map
        # proba of the conjunction
        proba = proba + dsz_focal_cdf(spec_zono,rhs_spec) 
        # proba of each condition of the conjunction separately
        for i in 1:length(rhs_spec)
            spec_zono = concretize(mat_spec[i,:]*res_zono)
            vec_proba[i] = vec_proba[i] + dsz_focal_cdf(spec_zono,Vector{Float64}([rhs_spec[i]]))
        end
    end

    proba = proba / nb_steps;
    vec_proba = vec_proba / nb_steps;
    vec_proba[length(rhs_spec)+1] = proba

    #println("Probability : ",proba)
    return vec_proba;
end





###### Heuristic to define automatically the number of focal elements for input approximation

function dsz_focal_refinement(nnet::Network, init_lb::Vector{Float64},init_ub::Vector{Float64}, mat_spec::Matrix{Float64},rhs_spec::Vector{Float64},is_bounded::Bool, eps::Float64)
    vect_nb_focal_elem = Vector{Int64}(undef, length(init_lb)) 
    nb_elem = 1
    for i in 1:length(init_lb)
	    vect_nb_focal_elem[i] = 5 
	    nb_elem = nb_elem * vect_nb_focal_elem[i]
    end
    acas_inputpbox = init_pbox_Normal(init_lb,init_ub,vect_nb_focal_elem,is_bounded)
    println("vect_nb_focal_elem=",vect_nb_focal_elem,":")
    @time proba = dsz_approximate_nnet_and_condition_nostorage(nnet, acas_inputpbox, mat_spec,rhs_spec)
    w = sup(proba)-inf(proba)
    w_test = Vector{Float64}(undef, length(init_lb))
    is_improved = false
    increment = 1
    while (! is_improved)
    	for i in 1:length(init_lb)
	        temp = max(increment,round.(Int32,0.1*vect_nb_focal_elem[i]))
	        vect_nb_focal_elem[i] = vect_nb_focal_elem[i] + temp
            acas_inputpbox = init_pbox_Normal(init_lb,init_ub,vect_nb_focal_elem,is_bounded)
	        @time proba = dsz_approximate_nnet_and_condition_nostorage(nnet, acas_inputpbox, mat_spec,rhs_spec)
	        w_test[i] = sup(proba)-inf(proba)
	        if (w > w_test[i])
		        is_improved = true
	        end
            if (w_test[i] < eps)
                println("Final vect_nb_focal_elem=",vect_nb_focal_elem,":")
                println("Final Proba = ",proba)
                return proba
            end
	        println("vect_nb_focal_elem=",vect_nb_focal_elem,":"," prob width=",w_test[i])
            vect_nb_focal_elem[i] = vect_nb_focal_elem[i] - temp
    	end
	    increment = increment * 2
    end

    incr_coeff = 50.0; # this increasing rate will be reduced with iterations
    improvement_vector = round.(Int32, (w./w_test.-1).*vect_nb_focal_elem.*incr_coeff)
    println("w_test=",w_test)
    println("w/w_test.-1", w./w_test.-1)
    println("improvement_vector", improvement_vector)
    # putting to 0 all but the greatest 2 components
    #sort = sortperm(improvement_vector)
    increment = 1
    sort = sortperm(w./w_test)
    for i in 1:length(sort)-2
	    improvement_vector[sort[i]] = 0
    end
    while (maximum(w./w_test.-1) >= 0.01 && nb_elem < 1000000)
    	vect_nb_focal_elem = vect_nb_focal_elem + improvement_vector
	    if (incr_coeff > 5.)
		    incr_coeff = incr_coeff/2
	    end
	    nb_elem = 1
    	for i in 1:length(init_lb)
            nb_elem = nb_elem * vect_nb_focal_elem[i]
    	end
        #vect_nb_focal_elem[argmin(w_test)] = vect_nb_focal_elem[argmin(w_test)] + 10 # seems better to augment only one compoennt
	    acas_inputpbox = init_pbox_Normal(init_lb,init_ub,vect_nb_focal_elem,is_bounded)
    	println("vect_nb_focal_elem=",vect_nb_focal_elem,":")
    	@time proba = dsz_approximate_nnet_and_condition_nostorage(nnet, acas_inputpbox, mat_spec,rhs_spec)
    	w = sup(proba)-inf(proba)  
        if (w < eps)
            println("Final vect_nb_focal_elem=",vect_nb_focal_elem,":")
            println("Final Proba = ",proba)
            return proba
        end
	    if (nb_elem > 1000000)
	        break
	    end
	    w_test = Vector{Float64}(undef, length(init_lb))
    	for i in 1:length(init_lb)
	        temp = max(increment,round.(Int32,0.1*vect_nb_focal_elem[i]))
            vect_nb_focal_elem[i] = vect_nb_focal_elem[i] + temp
            acas_inputpbox = init_pbox_Normal(init_lb,init_ub,vect_nb_focal_elem,is_bounded)
            @time proba = dsz_approximate_nnet_and_condition_nostorage(nnet, acas_inputpbox, mat_spec,rhs_spec)
            w_test[i] = sup(proba)-inf(proba)
	        println("vect_nb_focal_elem=",vect_nb_focal_elem,":"," prob width=",w_test[i])
            if (w_test[i] < eps)
                println("Final vect_nb_focal_elem=",vect_nb_focal_elem,":")
                println("Final Proba = ",proba)
                return proba
            end
            vect_nb_focal_elem[i] = vect_nb_focal_elem[i] - temp
    	end
	    improvement_vector = round.(Int32, (w./w_test.-1).*vect_nb_focal_elem.*incr_coeff)
	    # putting to 0 all but the greatest 2 components
    	#sort = sortperm(improvement_vector)
    	sort = sortperm(w./w_test)
    	for i in 1:length(sort)-2
            improvement_vector[sort[i]] = 0
    	end
	    println("w_test=",w_test)
    	println("w/w_test.-1", w./w_test.-1)
    	println("improvement_vector", improvement_vector)
    end

    println("Final vect_nb_focal_elem=",vect_nb_focal_elem,":")
    println("Final Proba = ",proba)
    return proba

end

