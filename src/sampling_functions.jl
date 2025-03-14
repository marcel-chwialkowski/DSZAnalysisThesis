import Base: rand
using Distributions 
using PyPlot
import Base: Atomic

#sample a value from a p-box by:
#1. picking a value  u from [0,1] uniformly
#2. picking a value from [\overline{F}(u), \underline{F}(u)] uniformly
function sample_from_pbox(pbox, num = 1)
    res = Vector{Float64}()
    for i in 1:num
        u = Base.rand()
        interval = cut(pbox, u)
        lower = inf(interval)
        upper = sup(interval)
        sampled_value = Base.rand() * (upper - lower) + lower
        push!(res, sampled_value)
    end
    return res
end

#same, but samples multiple values from multiple p-boxes
function sample_from_pboxes(pboxes, num = 1)
    res = Vector{Vector{Float64}}()
    for i in 1:num
        ress = Vector{Float64}()
        for pbox in pboxes
            u = Base.rand()
            interval = cut(pbox, u)
            lower = inf(interval)
            upper = sup(interval)
            sampled_value = Base.rand() * (upper - lower) + lower
            push!(ress, sampled_value)
        end
        push!(res, ress)
    end
    return res
end

#same, but in step 2. uses beta(alpha, beta) distribution instead of the uniform distribution.
function sample_from_pboxes_beta_param(pboxes, num = 1, alpha = 1/2, beta = 1/2)
    res = Vector{Vector{Float64}}()
    for i in 1:num
        ress = Vector{Float64}()
        for pbox in pboxes
            u = Base.rand()
            interval = cut(pbox, u)
            lower = inf(interval)
            upper = sup(interval)
            beta_dist = Beta(alpha, beta)
            sample = Distributions.rand(beta_dist)
            sampled_value = sample * (upper - lower) + lower
            push!(ress, sampled_value)
        end
        push!(res, ress)
    end
    return res
end

#builds an empirical CDF from data. However, for repeating sample values this function returns multiple
#values for the same x. The function is still useful for plotting the empirical CDFs.
function empirical_cdf(data)
    n = length(data)
    sorted_data = sort(data)
    cdf = Vector{Float64}()
    for i in 1:n
        cdf_value = i / n
        push!(cdf, cdf_value)
    end
    return sorted_data, cdf
end

#same as before, but doesnt return multiple values for the same x
function empirical_cdf_diff(data)
    n = length(data)
    sorted_data = sort(data)
    unique_vals = unique(sorted_data)
    cdf = Dict()

    for (i, val) in enumerate(sorted_data)
        cdf[val] = i / n 
    end

    res = [(unique_vals[i], cdf[unique_vals[i]]) for i in 1:length(unique_vals)]
    return res
end

#given p-boxes described by [lowr1, upr1], [lowr2, upr2], returns a p-box that envelopes both
function envelope(upr1, upr2, lowr1, lowr2)
    n1 = length(upr1)
    n2 = length(upr2)
    uprres = [(-10000.0, 0.0)]
    iter1 = 1
    iter2 = 1
    while iter1 <= n1 && iter2 <= n2
        if iter1 >= n1
            nxt = (upr2[iter2][1], max(uprres[end][2], upr2[iter2][2]))
            iter2 += 1
        elseif iter2 >= n2
            nxt = (upr1[iter1][1], max(uprres[end][2], upr1[iter1][2]))
            iter1 += 1
        elseif upr1[iter1][1] < upr2[iter2][1]
            nxt = (upr1[iter1][1], max(uprres[end][2], upr1[iter1][2]))
            iter1 += 1
        elseif upr1[iter1][1] > upr2[iter2][1]
            nxt = (upr2[iter2][1], max(uprres[end][2], upr2[iter2][2]))
            iter2 += 1
        else
            nxt = (upr1[iter1][1], max(uprres[end][2], upr1[iter1][2], upr2[iter2][2]))
            iter1 += 1
            iter2 += 1
        end
        push!(uprres, nxt)
    end
    popfirst!(uprres)

    m1 = length(lowr1)
    m2 = length(lowr2)
    lowrres = [(1000.0, 1.0)]
    iter1 = m1
    iter2 = m2
    while iter1 >= 1 && iter2 >= 1
        if iter1 < 1
            nxt = (lowr2[iter2][1], min(lowrres[end][2], lowr2[iter2][2]))
            iter2 -= 1
        elseif iter2 < 1
            nxt = (lowr1[iter1][1], min(lowrres[end][2], lowr1[iter1][2]))
            iter1 -= 1
        elseif lowr1[iter1][1] > lowr2[iter2][1]
            nxt = (lowr1[iter1][1], min(lowrres[end][2], lowr1[iter1][2]))
            iter1 -= 1
        elseif lowr1[iter1][1] < lowr2[iter2][1]
            nxt = (lowr2[iter2][1], min(lowrres[end][2], lowr2[iter2][2]))
            iter2 -= 1
        else
            nxt = (lowr1[iter1][1], min(lowrres[end][2], lowr1[iter1][2], lowr2[iter2][2]))
            iter1 -= 1
            iter2 -= 1
        end
        push!(lowrres, nxt)
    end
    lowrres = reverse(lowrres)
    pop!(lowrres)
    return uprres, lowrres
end

#given a list of cdfs, envelopes them all
function envelope_cdfs(cdfs)
    n = length(cdfs)
    uprres, lowres = envelope(cdfs[1], cdfs[2], cdfs[1], cdfs[2])
    for i in 3:n
        println(i)
        uprres, lowres = envelope(uprres, cdfs[i], lowres, cdfs[i])
    end
    return uprres, lowres
end

#examples of usage of the envelope functions and empirical_cdf functions
function main()
    #=
    D1 =  [-1.2, -0.5, 0.0, 0.3, 1.1]
    C1 =  [0.2, 0.4, 0.6, 0.8, 1.0]
    D2 =  [0.2, 0.5, 1.0, 1.5, 2.5]
    C2 =  [0.2, 0.4, 0.6, 0.8, 1.0]
    D3 =  [-2.5, -1.5, 0.0, 1.5, 2.5]
    C3 =  [0.2, 0.4, 0.6, 0.8, 1.0]

    p1 = [(D1[i], C1[i]) for i in 1:length(D1)]
    p2 = [(D2[i], C2[i]) for i in 1:length(D2)]
    p3 = [(D3[i], C3[i]) for i in 1:length(D3)]

    upr, low = envelope_cdfs([p1, p2, p3])
    x_upr = [upr[i][1] for i in 1:length(upr)]
    val_upr = [upr[i][2] for i in 1:length(upr)]
    x_low = [low[i][1] for i in 1:length(low)]
    val_low = [low[i][2] for i in 1:length(low)]

    plot(x_upr, val_upr, marker="o", linestyle="-")
    plot(x_low, val_low, marker="o", linestyle="-")
    PyPlot.savefig("../pictures/toy_envelope.png")
    

    println(low)
    println(upr)
    return 0
    =#
    x = Normal(0, 1)
    data1 = Distributions.rand(x, 100)
    y = Normal(0, 4)
    data2 = Distributions.rand(y, 100)
    z = Beta(1/2,1/2)
    data3 = Distributions.rand(z, 100)

    cdf1 = empirical_cdf_diff(data1)
    cdf2 = empirical_cdf_diff(data2)
    cdf3 = empirical_cdf_diff(data3)
    x_1 = [cdf1[i][1] for i in 1:length(cdf1)]
    y_1 = [cdf1[i][2] for i in 1:length(cdf1)]
    x_2 = [cdf2[i][1] for i in 1:length(cdf2)]
    y_2 = [cdf2[i][2] for i in 1:length(cdf2)]
    x_3 = [cdf3[i][1] for i in 1:length(cdf3)]
    y_3 = [cdf3[i][2] for i in 1:length(cdf3)]
    plot(x_1,y_1, marker="o", linestyle="-")
    plot(x_2,y_2, marker="o", linestyle="-")
    plot(x_3,y_3, marker="o", linestyle="-")
    PyPlot.savefig("../pictures/test_envelope.png")
    PyPlot.close()

    upr, low = envelope_cdfs([cdf1, cdf2, cdf3])
    x_upr = [upr[i][1] for i in 1:length(upr)]
    val_upr = [upr[i][2] for i in 1:length(upr)]
    x_low = [low[i][1] for i in 1:length(low)]
    val_low = [low[i][2] for i in 1:length(low)]

    plot(x_upr, val_upr, marker="o", linestyle="-")
    plot(x_low, val_low, marker="o", linestyle="-")
    PyPlot.savefig("../pictures/envelope.png")
    
end

#main()