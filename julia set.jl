using Pkg
Pkg.instantiate()
using StatsBase
using Unitful
include("./functions.jl")



function create_julia_set_data(c, R, iterations, x_pixels, y_pixels
    ; padding = 1, x_scale = 1, y_scale = 1)
    # create the iterator function 
    f(z) = z^2 + c

    x_mid = x_pixels ÷ 2 + x_pixels % 2
    y_mid = y_pixels ÷ 2 + y_pixels % 2


    data = zeros(y_pixels, x_pixels)
    convergence_iterations = zeros(y_pixels, x_pixels)
    # use 16 bit integers for speed 
    convergenceˣ = Vector{Int16}()
    convergenceʸ = Vector{Int16}()
    for x = 1:x_pixels
        for y = 1:y_pixels
            # create z 
            z = (R * (x - x_mid) / x_mid) / x_scale + (R * (y - y_mid) / (y_mid * y_scale))im
            n = 1
            while n < iterations && abs(z) <= R
                z = f(z)
                n += 1
            end
            convergence_iterations[y, x] = n
            if n == iterations && abs(z) <= R
                data[y, x] = 1
                append!(convergenceˣ, x)
                append!(convergenceʸ, y)
            end
        end
    end

    if padding > 1
        y_padding = (padding - 1) * y_pixels |>
                    round |>
                    Int

        x_padding = (padding - 1) * x_pixels |>
                    round |>
                    Int

        data = [
            zeros(y_padding, size(data)[2])
            data
            zeros(y_padding, size(data)[2])
        ]

        data = [zeros(size(data)[1], x_padding) data zeros(size(data)[1], x_padding)]
    end

    return data, [transpose(convergenceʸ); transpose(convergenceˣ)], convergence_iterations
end

# height and width of the poster/canvas
width = (60u"cm" |> u"inch") / u"inch"
height = (40u"cm" |> u"inch") / u"inch"
dpi = 300
# R² - R > |c|
y_pixels = (height * dpi + 1) |> Float64 |> floor |> Int
x_pixels = (width * dpi + 1) |> Float64 |> floor |> Int

c = -0.835 - 0.2321im
R = 2
iterations = 40

# c = -0.8im
# R = 2
# iterations = 30

if R^2 - R < abs(c)
    println("Inequality not satisfied")
    error()
end


data, convergence_points, convergence_iterations =
    create_julia_set_data(c
                          , R
                          , iterations
                          , x_pixels
                          , y_pixels
                          , y_scale = 1.5
                          , x_scale = 1.2)

# heatmap(1:size(data)[2], 1:size(data)[1], data)
# heatmap(1:size(convergence_iterations)[2], 1:size(convergence_iterations)[1], convergence_iterations)

# fav
# 1
data, convergence_points, _ = create_julia_set_data(-0.835 - 0.2321im
                                                    , 2
                                                    , 40
                                                    , x_pixels
                                                    , y_pixels
                                                    , y_scale = 1.8)
# 
# 2
# data, convergence_points, _ = create_julia_set_data(-0.005 - 0.8im
#                                                     , 2
#                                                     , 30
#                                                     , √2 * y_pixels |> round |> Int
#                                                     , y_pixels
#                                                     , y_scale = 1.5
#                                                     , x_scale = 1.2)


# save the data

# create the tree 
tree =
    float(convergence_points) |>
    @> KDTree(leafsize = 100)


# replace with distances 
pixel_data = data
# replace the zeros with distances 
# for xᵢ in 1:size(pixel_data)[2]
#     for yᵢ in 1:size(pixel_data)[1]
#         if pixel_data[yᵢ, xᵢ] == 0
#             pixel_data[yᵢ, xᵢ] = nn(tree, [yᵢ; xᵢ])[2]
# end end end 

indices = findall(pixel_data .== 0)
distances = nn(tree, getindex.(indices, [1 2]) |> transpose)[2]
pixel_data[indices] = distances

colour = cgrad([:white, :green, :black, :black], [0.000001, 0.45])

image = faded_plot(pixel_data
                   , size(pixel_data)[2]
                   , size(pixel_data)[1]
                   , colour = colour
                   , power = 0.15
                   , extra_black_levels = 0
                   , calculate_distances = false)

save_path = joinpath("./julia_sets", "C=$c", "R=$R")
mkpath(save_path)


for format in ["svg", "png", "ps"]
    savefig(image
            , joinpath(save_path, "x=$x_pixels,y=$y_pixels." * format))
end

##
# faded_plot(pixel_data
#            ,size(pixel_data)[2]
#            ,size(pixel_data)[1]
#            ,colour = cgrad([:white, :mediumpurple2, :black, :black], [0.000001, 0.75])
#            ,power = 0.15
#            ,extra_black_levels = 0
#            ,calculate_distances = false)


# faded_plot(pixel_data
#            ,size(pixel_data)[2]
#            ,size(pixel_data)[1]
#            ,colour = cgrad([:white, :indigo, :black, :black], [0.000001, 0.75])
#            ,power = 0.2
#            ,extra_black_levels = 0
#            ,calculate_distances = false)




# faded_plot(pixel_data
#            ,size(pixel_data)[2]
#            ,size(pixel_data)[1]
#            ,colour = cgrad([:white, :red2, :black, :black], [0.000001, 0.7])
#            ,power = 0.18
#            ,extra_black_levels = 0
#            ,calculate_distances = false)

