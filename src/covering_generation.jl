using Random
using ProbabilityBoundsAnalysis
using Distributions
using IntervalArithmetic
using PyPlot

#within a grid, width x height, generate an increasing path on the grid.
function generate_random_path(width, height, step_size)
    m, n = height, width  
    u, r = m / step_size, n / step_size  
    
    path = [(0.0, 0.0)]
    x, y = 0.0, 0.0 
    
    while u + r > 0
        if r == 0  
            y += step_size
            u -= 1
        elseif u == 0  
            x += step_size
            r -= 1
        else
            if rand() < r / (u + r)
                x += step_size
                r -= 1
            else
                y += step_size
                u -= 1
            end
        end
        push!(path, (x, y))
    end
    push!(path, (width, height))
    
    return path
end

#transform function  - takes a path on the grid and fits it to the pbox by changing its x values.
#in the end, we don't find utility for it.
function fit_to_box(box :: pbox, path::Vector{Tuple{Float64, Float64}})
    for i in eachindex(path)
        if path[i][2] < 0.0
            path[i] = (path[i][1], 0.0)
        end
        if path[i][2] > 1.0
            path[i] = (path[i][1], 1.0)
        end
        r = cut(box, path[i][2])
        path[i] = (inf(r)+ (sup(r) - inf(r)) * path[i][1], path[i][2])
    end
    return path
end

#auxilliary function - regularises the path for processing.
function add_duplicates(path::Vector{Tuple{Float64, Float64}})
    n = length(path)
    nsiz = path[n][1]
    pathnice = Vector{Tuple{Float64, Float64}}()
    current = 1
    while current < n
        push!(pathnice, path[current])
        if path[current][1] == path[current + 1][1]
            interm = (path[current + 1][1], path[current][2])
            push!(pathnice, interm)
            current += 1
        else
            push!(pathnice, path[current + 1])
            current += 2
        end
    end
    if length(pathnice) < 2 * nsiz
        push!(pathnice, path[n])
    end
    return pathnice
end

#auxilliary function - removes duplicates, after processing.
function remove_duplicates(path::Vector{Tuple{Float64, Float64}})
    unique_path = [path[1]]  # Start with the first element
    for i in 2:length(path)
        if path[i] != path[i-1]
            push!(unique_path, path[i])
        end
    end
    return unique_path
end

#takes a path on the grid and fits it to the pbox by changing its x values.
#in the end, we don't find utility for it.
function fit_to_box_robust(box :: pbox, path::Vector{Tuple{Float64, Float64}})
    fittedpath = Vector{Tuple{Float64, Float64}}()
    endofbox = path[length(path)][1]
    currx = 0.0
    n = Int(length(path) / 2)
    for i in 1:(n - 1)
        fst = path[2 * i - 1]
        fstmod = (currx, fst[2])
        push!(fittedpath, fstmod)
         
        snd = path[2 * i]
        thd = path[2 * i + 1]
        proportion = (snd[2] - fst[2]) / (endofbox - fst[2])
        boxval = cdf(box, thd[2])
        currx = inf(boxval) + (sup(boxval) - inf(boxval)) * proportion
        sndmod = (currx, snd[2])
        push!(fittedpath, sndmod)
    end
    #last level
    fst = path[2 * n - 1]
    snd = path[2 * n]
    fstmod = (currx, fst[2])
    push!(fittedpath, fstmod)
    proportion = (snd[2] - fst[2]) / (endofbox - fst[2])
    boxval = cdf(box, snd[2])
    currx = inf(boxval) + (sup(boxval) - inf(boxval)) * proportion
    sndmod = (currx, snd[2])
    push!(fittedpath, sndmod)

    return fittedpath
end

#checks if path is inside a p-box
function check_if_path_in_pbox(box :: pbox, path::Vector{Tuple{Float64, Float64}})
    for (x, y) in path
        boxval = cdf(box, x)
        if inf(boxval) < y || sup(boxval) > y
            return false
        end
    end
    return true
end

#gets tails of p-box
function get_left_right_tails(box :: pbox, val :: Float64)
    left_tail = inf(cut(box, val))
    right_tail = sup(cut(box, 1 - val))
    return left_tail, right_tail
end

#tries to do randomised sampling of a covering.
#not used
function sample_covering(box :: pbox, precision :: Float64, n :: Int)
    samples = []
    for i in 1:n
        path = generate_random_path(1.0, 1.0, precision)
        pathfitted = fit_to_box(box, path)
        push!(samples, pathfitted)
    end
    return samples
end

#plots a random path
function plot_path(height, width, step_size)
    path = generate_random_path(height, width, step_size)
    
    # Extract x and y coordinates
    x_coords = [p[1] for p in path]
    y_coords = [p[2] for p in path]
    
    # Plot the path
    figure(figsize=(6,6))
    plot(x_coords, y_coords, linestyle="-", color="r", markersize=1, linewidth=0.5, alpha =1.0)
    
    grid(true, linestyle="--", alpha=1.0)
    legend()
    PyPlot.savefig("random_path.png")

end

#plots multiple random paths
function plot_paths(height, width, step_size, num)
    figure(figsize=(6,6))
    for i in 1:num
        path = generate_random_path(height, width, step_size)
    
        # Extract x and y coordinates
        x_coords = [p[1] for p in path]
        y_coords = [p[2] for p in path]

        # Plot the path
        plot(x_coords, y_coords, linestyle="-", color="r", markersize=1, linewidth=0.5, alpha =1.0)
    end
    grid(true, linestyle="--", alpha=1.0)
    PyPlot.savefig("random_paths.png")
end

#plots a covering sampled for a pbox
function plot_paths_within_pbox(box :: pbox, precision :: Float64, num :: Int)
    samples = sample_covering(box, precision, num)
    plot(box)
    #figure(figsize=(6,6))
    for path in samples
        println("done!")
        x_coords = [p[1] for p in path]
        y_coords = [p[2] for p in path]
        plot(x_coords, y_coords, linestyle="-", color="r", markersize=1, linewidth=0.5, alpha =1.0)
    end
    grid(true, linestyle="--", alpha=1.0)
    PyPlot.savefig("random_paths_within_pbox.png")
end

# Example usage
#box = normal(interval(0,3),1)
#path = generate_random_path(1.0, 1.0, 0.05)
#pathdup = add_duplicates(path)
#pathfitted = fit_to_box_robust(box, pathdup)
#pathnorepeat = remove_duplicates(pathfitted)

#plot(box)
#figure(figsize=(6,6))
#x_coords = [p[1] for p in pathnorepeat]
#y_coords = [p[2] for p in pathnorepeat]
#plot(x_coords, y_coords, linestyle="-", color="r", markersize=1, linewidth=0.5, alpha =1.0)

#grid(true, linestyle="--", alpha=1.0)
#PyPlot.savefig("pathfitted.png")
#println(generate_random_path(10,10, 0.1))  # Grid 2x3 with steps of 0.5
#println(sample_covering(box, 0.1, 100))
#plot_paths(10,10,0.1,1000)
#plot_paths_within_pbox(box, 0.05, 10000)
#=
function generate_paths_in_pbox(box :: pbox, precisionv :: Float64, precisionh :: Float64)
    #find value in pbox such that cdf is at least precision.
    left = inf(cut(box, 0.001))
    
    #starting from left at height 0, check all paths
    paths = []
    pathstocheck = [[[left, 0.001]]]

    iter =5 

    while length(pathstocheck) > 0
        #println("all paths:")
        #println(pathstocheck)
        currpath = pop!(pathstocheck)
        n = length(currpath)
        currind = currpath[n]
        #println("newpath")
        #println(currpath)

        #ending criterion
        if currind[2] > 1 || (1 - currind[2] < 0.01)
            if length(paths) > 0
                break
            end
            if length(paths) % 100 == 0
                println(length(paths))
            end
            push!(paths, currpath)
            continue
        end

        #how much we can go up or right
        verpossible = sup(cdf(box, currind[1])) + 0.01
        horpossible = sup(cut(box, currind[2])) + 0.01

        #check if we can go up
        if currind[2] + precisionv <= verpossible
            println("go up")
            push!(currpath, [currind[1], currind[2] + precisionv])
            push!(pathstocheck, currpath)
            #println(currpath)
            #println(pathstocheck)
            pop!(currpath)
        end

        #check if we can go right
        if currind[1] + precisionh <= horpossible
            println("go right")
            push!(currpath, [currind[1] + precisionh, currind[2]])
            push!(pathstocheck, currpath)
            #println(currpath)
            #println(pathstocheck)
            pop!(currpath)
        end
        
        iter -= 1
        if iter == 0
            break
        end
    end
    return paths
end
=#


#function that generates paths in a pbox without a transform. Takes a grid of points within a p-box through which
#the staircase function can go, and generates all possible paths.
function generate_paths_in_pbox(box::pbox, precisionv::Float64, precisionh::Float64)
    left = inf(cut(box, precisionv))
    paths = []
    pathstocheck = [(left, 0.001, [])] 

    cached_heights = precompute_verpossible(box, precisionv, precisionh)
    cached_widths = precompute_horpossible(box, precisionv)

    println("caching done!")

    while !isempty(pathstocheck)
        (x, y, moves) = pop!(pathstocheck)

        # Ending criterion
        if y >= 1.0 - 0.01
            path = [(left, 0.001)]
            for move in moves
                if move == :up
                    path = vcat(path, [(path[end][1], path[end][2] + precisionv)])
                elseif move == :right
                    path = vcat(path, [(path[end][1] + precisionh, path[end][2])])
                end
            end
            push!(paths, path)
            if length(paths) == 100000
                return paths
            end
            continue
        end

        verpossible = cached_heights[x]
        horpossible = cached_widths[y]

        if y + precisionv <= verpossible
            push!(pathstocheck, (x, y + precisionv, vcat(moves, [:up])))
        end

        if x + precisionh <= horpossible
            push!(pathstocheck, (x + precisionh, y, vcat(moves, [:right])))
        end
    end
    return paths
end

#helper function to generate the possible heights for a given x
function precompute_verpossible(box :: pbox, precisionv :: Float64, precisionh :: Float64)
    left = inf(cut(box, precisionv))
    right = sup(cut(box, 1 - 0.001))
    verpossible = sup(cdf(box, left)) + 0.01
    cached_heights = Dict()
    while left <= right
        cached_heights[left] = verpossible
        left += precisionh
        verpossible = sup(cdf(box, left)) + 0.01
    end
    return cached_heights
end

#helper function to generate the possible widths for a given y
function precompute_horpossible(box :: pbox, precisionv :: Float64)
    left = 0.001
    cached_widths = Dict()
    while left < 1.00
        horpossible = sup(cut(box, left)) + 0.01
        cached_widths[left] = horpossible
        left += precisionv
    end
    return cached_widths
end


box = normal(interval(0,0.7),1)
precisionv = 0.066
precisionh = 0.125


paths = generate_paths_in_pbox(box, precisionv, precisionh)
plot(box)
#print(paths[1])

for path in paths
    x_coords = [p[1] for p in path]
    y_coords = [p[2] for p in path]
    plot(x_coords, y_coords, linestyle="-", color="r", markersize=1, linewidth=0.5, alpha =1.0)
end

#PyPlot.axis("equal") 
grid(true, linestyle="--", alpha=1.0)
PyPlot.savefig("../pictures/pathGeneratedToPbox.png")
PyPlot.close()
