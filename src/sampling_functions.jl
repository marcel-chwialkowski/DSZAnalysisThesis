import Base: rand
using Distributions 

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

function sample_from_pboxes_beta(pboxes, num = 1)
    res = Vector{Vector{Float64}}()
    for i in 1:num
        ress = Vector{Float64}()
        for pbox in pboxes
            u = Base.rand()
            interval = cut(pbox, u)
            lower = inf(interval)
            upper = sup(interval)
            beta_dist = Beta(1/2, 1/2)
            sample = Distributions.rand(beta_dist)
            sampled_value = sample * (upper - lower) + lower
            push!(ress, sampled_value)
        end
        push!(res, ress)
    end
    return res
end

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

function pbox_from_data(data)
    sorted_data, cdf = empirical_cdf(data)
    n = length(sorted_data)
    alpha = 0.1  # significance level for 95% confidence
    K_n = sqrt(-0.5 * log(alpha / 2)) / sqrt(n)
    return sorted_data, cdf, max.(cdf .- K_n,[0 for i in 1:n]), min.(cdf .+ K_n,[1 for i in 1:n])
end