using Plots 
using Chain 
using NearestNeighbors
using Statistics
using ProgressMeter
using DynamicalSystems
using DataFrames
using DataFramesMeta
using DynamicPipe
using ProgressMeter
using ColorSchemes

# here we define soem useful functiuons
## 
"""
Function to create find create the distnces vector
""" 
function distances_matrix(sim_data
                          ,x_pixels
                          ,y_pixels
                          ;x_range_mulitplier = 1.05
                          ,y_range_mulitplier = 1.05
                          ,y_column = 2)
    sim_data = copy(sim_data)
    # conver to array if needed 
    if typeof(sim_data) == DataFrame
        sim_data = Array(sim_data)
    elseif typeof(sim_data) == Dataset{2, Float64}
        sim_data = [sim_data[i,j] for i in 1:size(sim_data)[1], j in 1:size(sim_data)[2]]
    end
    
    # convert to floats 
    sim_data = convert(Array{Float64}, sim_data)
    # center the image at 0, 0
    sim_data[:, 1] = sim_data[:, 1] .- mean(sim_data[:, 1])
    sim_data[:, 2] = sim_data[:, 2] .- mean(sim_data[:, 2])
    
    # standardise the axis 
    sim_data[:, 1] = sim_data[:, 1] ./ std(sim_data[:, 1]) 
    sim_data[:, 2] = sim_data[:, 2] ./ std(sim_data[:, 2])
    
    
    # if the y_range is not specified we set it 
    y_range  = @chain sim_data begin
        _[:,y_column]
        abs.([maximum(_), minimum(_)])
        maximum
        _ * y_range_mulitplier
    end
    
    
    x_range = sim_data |>
        @>  _[:,1] |>
            [maximum(_), minimum(_)] |>
            abs.(_) |>
            maximum |>
            _ * x_range_mulitplier


    # create the pixel data 
    pixel_data = zeros(x_pixels*y_pixels, 2)

    pixel_data[:,1] = [i for x in 1:y_pixels 
                       for i in range(-x_range, x_range, length = x_pixels)]

    pixel_data[:, 2] = [i for i in range(-y_range, y_range, length = y_pixels) 
                        for x in 1:x_pixels]


    # create the tree used to find the distances 
    tree = @chain sim_data begin
        _[:, [1,2]]
        transpose
        KDTree(_
               ,Euclidean()
               ;leafsize = size(_)[2]/1000 |> round |> Int)
    end
    
    sim_data = nothing

    # create the distnaces to the simulated data 
    distances = @chain pixel_data begin
        transpose
        nn(tree, _)
        # the second entry is the distnaces
        _[2]
    end

    distances = 
        @chain distances begin
            reshape((x_pixels, y_pixels))
            transpose
        end

    return distances
end

# -- testing --
# distances_vector(sim_data, x_pixels, y_pixels) 

##
"""
Create a glowing plot of from a distnaces matrix 
"""
function faded_plot(data
                    ,x_pixels 
                    ,y_pixels 
                    ;power = 0.05
                    ,colour = orange_fade
                    ,extra_black_levels = 10
                    ,x_range_mulitplier = 1.05
                    ,y_range_mulitplier = 1.05
                    ,y_column = 2
                    ,calculate_distances = true
                    ,rebase_points = false)

    if calculate_distances 
        distances_data = distances_matrix(data,
                                x_pixels, 
                                y_pixels,
                                x_range_mulitplier = x_range_mulitplier,
                                y_range_mulitplier = y_range_mulitplier,
                                y_column = y_column)
    else
        distances_data = data
    end

    # add the extra black levels
    # colour = @chain deepcopy(colour) begin
    #     push!(
    #         RGB{Float64}[
    #             RGB{Float64}(0, 0, 0)
    #             for i in 1:extra_black_levels
    #         ]
    #     )
    # end
    if extra_black_levels != 0 
        colour = [colour; 
                RGB{Float64}[
                    RGB{Float64}(0, 0, 0) for i in 1:extra_black_levels
                    ]
                ]
    end
    
    # transform the values to plot into a matrix 
    if size(distances_data) == (x_pixels*y_pixels,)
        distances_data = 
            @chain distances_data begin
                reshape((x_pixels, y_pixels))
                _ .^ power
                transpose
            end
    else
        distances_data = distances_data .^ power 
    end
    
    if rebase_points
        min_val = minimum(distances_data)
        
        for (x,y) in eachrow(data)
            distances_data[x,y] = min_val
            break
        end
    end

    # make the plot
    plot = heatmap(
        1:x_pixels
        ,1:y_pixels
        ,distances_data
        ,c = colour
        ,legend = false
        ,xaxis = false
        ,yaxis = false
        ,xticks = false
        ,yticks = false
        ,size = (x_pixels, y_pixels)
        ,background_color = :transparent 
        # ,foreground_color = :transparent
    ) 

    return plot
end

# -- testing --  

# @chain data begin 
#     distances_matrix(x_pixels, y_pixels)
#     faded_plot(x_pixels, y_pixels)
# end

##
"""
Create poincare plot form data 
"""
function faded_poincare_plot(data, ω, x_pixels, y_pixels; offset = 0, time_column = nothing,
                             x_column = 1, y_column = 2, power = 0.05, color = orange_fade
                             ,extra_black_levels = 10, x_range = nothing, y_range = nothing)

    # convert to array if data is a data frame 
    if typeof(data) == DataFrame
        data = data |> Array
    end

    # convert 1D array to 2D
    if data |> size |> length == 1 
        data = reshape(data, (data |> length, 1))
    end

    # defualt time column to last if not specified 
    if time_column |> isnothing
        time_column = 
            @chain data begin
                size
                _[end]
            end
    end

    t = data[:,time_column] .+ offset
    ωt = t .* ω

    # get the remainder of 2π division 
    remainder = ωt .% 2π

    # indicate if we have fully cycled around 2π
    indicator = 
        @chain remainder[2:end] .- remainder[1:end-1] begin
            _ .< 0
            push!(false)
        end

    data = data[indicator, :]

    plot =  @chain data begin
        faded_plot(x_pixels, y_pixels, power = power, color = color, 
                   extra_black_levels = extra_black_levels)
    end

    return plot
    
end

##

function make_poincare_data(system = modified_duffing_with_jacob;
                            u0 = [1, 0.000001], 
                            N = 1e6, 
                            ω = 1.2,
                            dt = 0.01)

    data = [u0 |> transpose;]
    
    @showprogress for i in 1:N
        sim_data = trajectory(system(data[end, :]),
                              (2*π)/ω,
                              dt = dt)
        final_point = sim_data[end, :]
        sim_data = nothing 
        data = [data;
                final_point |> transpose]
    end     
    
    return data[2:end, :]
end 

# -- Testing -- 
#make_poincare_data(modified_duffing_with_jacob) 

##

# -- testing --
# faded_poincare_plot(data, 0.1, x_pixels, y_pixels, offset = 0)

##

# We got this color using ColorScheme.jl 
# unofornately I have forgot its name 
color1 = @chain RGB{Float64}[
    RGB{Float64}(0.0,0.0,0.0), 
    RGB{Float64}(0.020202020202020204,0.0,0.0), 
    RGB{Float64}(0.04040404040404041,0.0,0.0), 
    RGB{Float64}(0.06060606060606061,0.0,0.0), 
    RGB{Float64}(0.08080808080808081,0.0,0.0), 
    RGB{Float64}(0.10101010101010101,0.0,0.0), 
    RGB{Float64}(0.12121212121212122,0.0,0.0), 
    RGB{Float64}(0.1414141414141414,0.0,0.0), 
    RGB{Float64}(0.16161616161616163,0.0,0.0),
    RGB{Float64}(0.18181818181818182,0.0,0.0), 
    RGB{Float64}(0.20202020202020202,0.0,0.0), 
    RGB{Float64}(0.2222222222222222,0.0,0.0), 
    RGB{Float64}(0.24242424242424243,0.0,0.0), 
    RGB{Float64}(0.26262626262626265,0.0,0.0), 
    RGB{Float64}(0.2828282828282828,0.0,0.0), 
    RGB{Float64}(0.30303030303030304,0.0,0.0), 
    RGB{Float64}(0.32323232323232326,0.0,0.0), 
    RGB{Float64}(0.3434343434343434,0.0,0.0), 
    RGB{Float64}(0.36363636363636365,0.0,0.0), 
    RGB{Float64}(0.3838383838383838,0.0,0.0), 
    RGB{Float64}(0.40404040404040403,0.0,0.0), 
    RGB{Float64}(0.42424242424242425,0.0,0.0), 
    RGB{Float64}(0.4444444444444444,0.0,0.0), 
    RGB{Float64}(0.46464646464646464,0.0,0.0), 
    RGB{Float64}(0.48484848484848486,0.0,0.0), 
    RGB{Float64}(0.5050505050505051,0.005050505050505083,0.0), 
    RGB{Float64}(0.5252525252525253,0.025252525252525304,0.0), 
    RGB{Float64}(0.5454545454545454,0.045454545454545414,0.0), 
    RGB{Float64}(0.5656565656565656,0.06565656565656564,0.0), 
    RGB{Float64}(0.5858585858585859,0.08585858585858586,0.0), 
    RGB{Float64}(0.6060606060606061,0.10606060606060608,0.0), 
    RGB{Float64}(0.6262626262626263,0.1262626262626263,0.0), 
    RGB{Float64}(0.6464646464646465,0.14646464646464652,0.0), 
    RGB{Float64}(0.6666666666666666,0.16666666666666663,0.0), 
    RGB{Float64}(0.6868686868686869,0.18686868686868685,0.0), 
    RGB{Float64}(0.7070707070707071,0.20707070707070707,0.0), 
    RGB{Float64}(0.7272727272727273,0.2272727272727273,0.0), 
    RGB{Float64}(0.7474747474747475,0.24747474747474751,0.0), 
    RGB{Float64}(0.7676767676767676,0.2676767676767676,0.0),
    RGB{Float64}(0.7878787878787878,0.28787878787878785,0.0),
    RGB{Float64}(0.8080808080808081,0.30808080808080807,0.0), 
    RGB{Float64}(0.8282828282828283,0.3282828282828283,0.0), 
    RGB{Float64}(0.8484848484848485,0.3484848484848485,0.0), 
    RGB{Float64}(0.8686868686868687,0.36868686868686873,0.0), 
    RGB{Float64}(0.8888888888888888,0.38888888888888884,0.0), 
    RGB{Float64}(0.9090909090909091,0.40909090909090906,0.0), 
    RGB{Float64}(0.9292929292929293,0.4292929292929293,0.0), 
    RGB{Float64}(0.9494949494949495,0.4494949494949495,0.0), 
    RGB{Float64}(0.9696969696969697,0.4696969696969697,0.0), 
    RGB{Float64}(0.98989898989899,0.48989898989898994,0.0), 
    RGB{Float64}(1.0,0.5101010101010102,0.010101010101010166), 
    RGB{Float64}(1.0,0.5303030303030303,0.030303030303030276), 
    RGB{Float64}(1.0,0.5505050505050506,0.05050505050505061), 
    RGB{Float64}(1.0,0.5707070707070707,0.07070707070707072), 
    RGB{Float64}(1.0,0.5909090909090908,0.09090909090909083), 
    RGB{Float64}(1.0,0.6111111111111112,0.11111111111111116), 
    RGB{Float64}(1.0,0.6313131313131313,0.13131313131313127), 
    RGB{Float64}(1.0,0.6515151515151516,0.1515151515151516), 
    RGB{Float64}(1.0,0.6717171717171717,0.1717171717171717), 
    RGB{Float64}(1.0,0.6919191919191918,0.19191919191919182),
    RGB{Float64}(1.0,0.7121212121212122,0.21212121212121215), 
    RGB{Float64}(1.0,0.7323232323232323,0.23232323232323226), 
    RGB{Float64}(1.0,0.7525252525252526,0.2525252525252526), 
    RGB{Float64}(1.0,0.7727272727272727,0.2727272727272727), 
    RGB{Float64}(1.0,0.792929292929293,0.29292929292929304), 
    RGB{Float64}(1.0,0.8131313131313131,0.31313131313131315), 
    RGB{Float64}(1.0,0.8333333333333333,0.33333333333333326),
    RGB{Float64}(1.0,0.8535353535353536,0.3535353535353536), 
    RGB{Float64}(1.0,0.8737373737373737,0.3737373737373737), 
    RGB{Float64}(1.0,0.893939393939394,0.39393939393939403), 
    RGB{Float64}(1.0,0.9141414141414141,0.41414141414141414), 
    RGB{Float64}(1.0,0.9343434343434343,0.43434343434343425), 
    RGB{Float64}(1.0,0.9545454545454546,0.4545454545454546), 
    RGB{Float64}(1.0,0.9747474747474747,0.4747474747474747), 
    RGB{Float64}(1.0,0.994949494949495,0.49494949494949503), 
    RGB{Float64}(1.0,1.0,0.5151515151515151), 
    RGB{Float64}(1.0,1.0,0.5353535353535352), 
    RGB{Float64}(1.0,1.0,0.5555555555555556), 
    RGB{Float64}(1.0,1.0,0.5757575757575757), 
    RGB{Float64}(1.0,1.0,0.595959595959596), 
    RGB{Float64}(1.0,1.0,0.6161616161616161), 
    RGB{Float64}(1.0,1.0,0.6363636363636365), 
    RGB{Float64}(1.0,1.0,0.6565656565656566), 
    RGB{Float64}(1.0,1.0,0.6767676767676767), 
    RGB{Float64}(1.0,1.0,0.696969696969697), 
    RGB{Float64}(1.0,1.0,0.7171717171717171), 
    RGB{Float64}(1.0,1.0,0.7373737373737375), 
    RGB{Float64}(1.0,1.0,0.7575757575757576), 
    RGB{Float64}(1.0,1.0,0.7777777777777777), 
    RGB{Float64}(1.0,1.0,0.797979797979798), 
    RGB{Float64}(1.0,1.0,0.8181818181818181), 
    RGB{Float64}(1.0,1.0,0.8383838383838385), 
    RGB{Float64}(1.0,1.0,0.8585858585858586), 
    RGB{Float64}(1.0,1.0,0.8787878787878789), 
    RGB{Float64}(1.0,1.0,0.898989898989899), 
    RGB{Float64}(1.0,1.0,0.9191919191919191), 
    RGB{Float64}(1.0,1.0,0.9393939393939394), 
    RGB{Float64}(1.0,1.0,0.9595959595959596), 
    RGB{Float64}(1.0,1.0,0.9797979797979799), 
    RGB{Float64}(1.0,1.0,1.0)] begin
        # add some more black color for the outside of the poster 
        append!(
            RGB{Float64}[
                RGB{Float64}(1.0,1.0,1.0)
                for i in 1:50
            ]
        )
        reverse
    end



orange_fade = @chain RGB{Float64}[
    RGB{Float64}(0.0,0.0,0.0), 
    RGB{Float64}(0.020202020202020204,0.0,0.0), 
    RGB{Float64}(0.04040404040404041,0.0,0.0), 
    RGB{Float64}(0.06060606060606061,0.0,0.0), 
    RGB{Float64}(0.08080808080808081,0.0,0.0), 
    RGB{Float64}(0.10101010101010101,0.0,0.0), 
    RGB{Float64}(0.12121212121212122,0.0,0.0), 
    RGB{Float64}(0.1414141414141414,0.0,0.0), 
    RGB{Float64}(0.16161616161616163,0.0,0.0),
    RGB{Float64}(0.18181818181818182,0.0,0.0), 
    RGB{Float64}(0.20202020202020202,0.0,0.0), 
    RGB{Float64}(0.2222222222222222,0.0,0.0), 
    RGB{Float64}(0.24242424242424243,0.0,0.0), 
    RGB{Float64}(0.26262626262626265,0.0,0.0), 
    RGB{Float64}(0.2828282828282828,0.0,0.0), 
    RGB{Float64}(0.30303030303030304,0.0,0.0), 
    RGB{Float64}(0.32323232323232326,0.0,0.0), 
    RGB{Float64}(0.3434343434343434,0.0,0.0), 
    RGB{Float64}(0.36363636363636365,0.0,0.0), 
    RGB{Float64}(0.3838383838383838,0.0,0.0), 
    RGB{Float64}(0.40404040404040403,0.0,0.0), 
    RGB{Float64}(0.42424242424242425,0.0,0.0), 
    RGB{Float64}(0.4444444444444444,0.0,0.0), 
    RGB{Float64}(0.46464646464646464,0.0,0.0), 
    RGB{Float64}(0.48484848484848486,0.0,0.0), 
    RGB{Float64}(0.5050505050505051,0.005050505050505083,0.0), 
    RGB{Float64}(0.5252525252525253,0.025252525252525304,0.0), 
    RGB{Float64}(0.5454545454545454,0.045454545454545414,0.0), 
    RGB{Float64}(0.5656565656565656,0.06565656565656564,0.0), 
    RGB{Float64}(0.5858585858585859,0.08585858585858586,0.0), 
    RGB{Float64}(0.6060606060606061,0.10606060606060608,0.0), 
    RGB{Float64}(0.6262626262626263,0.1262626262626263,0.0), 
    RGB{Float64}(0.6464646464646465,0.14646464646464652,0.0), 
    RGB{Float64}(0.6666666666666666,0.16666666666666663,0.0), 
    RGB{Float64}(0.6868686868686869,0.18686868686868685,0.0), 
    RGB{Float64}(0.7070707070707071,0.20707070707070707,0.0), 
    RGB{Float64}(0.7272727272727273,0.2272727272727273,0.0), 
    RGB{Float64}(0.7474747474747475,0.24747474747474751,0.0), 
    RGB{Float64}(0.7676767676767676,0.2676767676767676,0.0),
    RGB{Float64}(0.7878787878787878,0.28787878787878785,0.0),
    RGB{Float64}(0.8080808080808081,0.30808080808080807,0.0), 
    RGB{Float64}(0.8282828282828283,0.3282828282828283,0.0), 
    RGB{Float64}(0.8484848484848485,0.3484848484848485,0.0), 
    RGB{Float64}(0.8686868686868687,0.36868686868686873,0.0), 
    RGB{Float64}(0.8888888888888888,0.38888888888888884,0.0), 
    RGB{Float64}(0.9090909090909091,0.40909090909090906,0.0), 
    RGB{Float64}(0.9292929292929293,0.4292929292929293,0.0), 
    RGB{Float64}(0.9494949494949495,0.4494949494949495,0.0), 
    RGB{Float64}(0.9696969696969697,0.4696969696969697,0.0), 
    RGB{Float64}(0.98989898989899,0.48989898989898994,0.0), 
    RGB{Float64}(1.0,0.5101010101010102,0.010101010101010166), 
    RGB{Float64}(1.0,0.5303030303030303,0.030303030303030276), 
    RGB{Float64}(1.0,0.5505050505050506,0.05050505050505061), 
    RGB{Float64}(1.0,0.5707070707070707,0.07070707070707072), 
    RGB{Float64}(1.0,0.5909090909090908,0.09090909090909083), 
    RGB{Float64}(1.0,0.6111111111111112,0.11111111111111116), 
    RGB{Float64}(1.0,0.6313131313131313,0.13131313131313127), 
    RGB{Float64}(1.0,0.6515151515151516,0.1515151515151516), 
    RGB{Float64}(1.0,0.6717171717171717,0.1717171717171717), 
    RGB{Float64}(1.0,0.6919191919191918,0.19191919191919182),
    RGB{Float64}(1.0,0.7121212121212122,0.21212121212121215), 
    RGB{Float64}(1.0,0.7323232323232323,0.23232323232323226), 
    RGB{Float64}(1.0,0.7525252525252526,0.2525252525252526), 
    RGB{Float64}(1.0,0.7727272727272727,0.2727272727272727), 
    RGB{Float64}(1.0,0.792929292929293,0.29292929292929304), 
    RGB{Float64}(1.0,0.8131313131313131,0.31313131313131315), 
    RGB{Float64}(1.0,0.8333333333333333,0.33333333333333326),
    RGB{Float64}(1.0,0.8535353535353536,0.3535353535353536), 
    RGB{Float64}(1.0,0.8737373737373737,0.3737373737373737), 
    RGB{Float64}(1.0,0.893939393939394,0.39393939393939403), 
    RGB{Float64}(1.0,0.9141414141414141,0.41414141414141414), 
    RGB{Float64}(1.0,0.9343434343434343,0.43434343434343425), 
    RGB{Float64}(1.0,0.9545454545454546,0.4545454545454546), 
    RGB{Float64}(1.0,0.9747474747474747,0.4747474747474747), 
    RGB{Float64}(1.0,0.994949494949495,0.49494949494949503), 
    RGB{Float64}(1.0,1.0,0.5151515151515151), 
    RGB{Float64}(1.0,1.0,0.5353535353535352), 
    RGB{Float64}(1.0,1.0,0.5555555555555556), 
    RGB{Float64}(1.0,1.0,0.5757575757575757), 
    RGB{Float64}(1.0,1.0,0.595959595959596), 
    RGB{Float64}(1.0,1.0,0.6161616161616161), 
    RGB{Float64}(1.0,1.0,0.6363636363636365), 
    RGB{Float64}(1.0,1.0,0.6565656565656566), 
    RGB{Float64}(1.0,1.0,0.6767676767676767), 
    RGB{Float64}(1.0,1.0,0.696969696969697), 
    RGB{Float64}(1.0,1.0,0.7171717171717171), 
    RGB{Float64}(1.0,1.0,0.7373737373737375), 
    RGB{Float64}(1.0,1.0,0.7575757575757576), 
    RGB{Float64}(1.0,1.0,0.7777777777777777), 
    RGB{Float64}(1.0,1.0,0.797979797979798), 
    RGB{Float64}(1.0,1.0,0.8181818181818181), 
    RGB{Float64}(1.0,1.0,0.8383838383838385), 
    RGB{Float64}(1.0,1.0,0.8585858585858586), 
    RGB{Float64}(1.0,1.0,0.8787878787878789), 
    RGB{Float64}(1.0,1.0,0.898989898989899), 
    RGB{Float64}(1.0,1.0,0.9191919191919191), 
    RGB{Float64}(1.0,1.0,0.9393939393939394), 
    RGB{Float64}(1.0,1.0,0.9595959595959596), 
    RGB{Float64}(1.0,1.0,0.9797979797979799), 
    RGB{Float64}(1.0,1.0,1.0)] begin
    append!(
        RGB{Float64}[
            RGB{Float64}(1., 1., 1.)
            for i in 1:50
        ]
    )
    reverse
end

color2 = @chain RGB{Float64}[
    RGB{Float64}(0.0,0.0,0.0), 
    RGB{Float64}(0.020202020202020204,0.0,0.0), 
    RGB{Float64}(0.04040404040404041,0.0,0.0), 
    RGB{Float64}(0.06060606060606061,0.0,0.0), 
    RGB{Float64}(0.08080808080808081,0.0,0.0), 
    RGB{Float64}(0.10101010101010101,0.0,0.0), 
    RGB{Float64}(0.12121212121212122,0.0,0.0), 
    RGB{Float64}(0.1414141414141414,0.0,0.0), 
    RGB{Float64}(0.16161616161616163,0.0,0.0),
    RGB{Float64}(0.18181818181818182,0.0,0.0), 
    RGB{Float64}(0.20202020202020202,0.0,0.0), 
    RGB{Float64}(0.2222222222222222,0.0,0.0), 
    RGB{Float64}(0.24242424242424243,0.0,0.0), 
    RGB{Float64}(0.26262626262626265,0.0,0.0), 
    RGB{Float64}(0.2828282828282828,0.0,0.0), 
    RGB{Float64}(0.30303030303030304,0.0,0.0), 
    RGB{Float64}(0.32323232323232326,0.0,0.0), 
    RGB{Float64}(0.3434343434343434,0.0,0.0), 
    RGB{Float64}(0.36363636363636365,0.0,0.0), 
    RGB{Float64}(0.3838383838383838,0.0,0.0), 
    RGB{Float64}(0.40404040404040403,0.0,0.0), 
    RGB{Float64}(0.42424242424242425,0.0,0.0), 
    RGB{Float64}(0.4444444444444444,0.0,0.0), 
    RGB{Float64}(0.46464646464646464,0.0,0.0), 
    RGB{Float64}(0.48484848484848486,0.0,0.0), 
    RGB{Float64}(0.5050505050505051,0.005050505050505083,0.0), 
    RGB{Float64}(0.5252525252525253,0.025252525252525304,0.0), 
    RGB{Float64}(0.5454545454545454,0.045454545454545414,0.0), 
    RGB{Float64}(0.5656565656565656,0.06565656565656564,0.0), 
    RGB{Float64}(0.5858585858585859,0.08585858585858586,0.0), 
    RGB{Float64}(0.6060606060606061,0.10606060606060608,0.0), 
    RGB{Float64}(0.6262626262626263,0.1262626262626263,0.0), 
    RGB{Float64}(0.6464646464646465,0.14646464646464652,0.0), 
    RGB{Float64}(0.6666666666666666,0.16666666666666663,0.0), 
    RGB{Float64}(0.6868686868686869,0.18686868686868685,0.0), 
    RGB{Float64}(0.7070707070707071,0.20707070707070707,0.0), 
    RGB{Float64}(0.7272727272727273,0.2272727272727273,0.0), 
    RGB{Float64}(0.7474747474747475,0.24747474747474751,0.0), 
    RGB{Float64}(0.7676767676767676,0.2676767676767676,0.0),
    RGB{Float64}(0.7878787878787878,0.28787878787878785,0.0),
    RGB{Float64}(0.8080808080808081,0.30808080808080807,0.0), 
    RGB{Float64}(0.8282828282828283,0.3282828282828283,0.0), 
    RGB{Float64}(0.8484848484848485,0.3484848484848485,0.0), 
    RGB{Float64}(0.8686868686868687,0.36868686868686873,0.0), 
    RGB{Float64}(0.8888888888888888,0.38888888888888884,0.0), 
    RGB{Float64}(0.9090909090909091,0.40909090909090906,0.0), 
    RGB{Float64}(0.9292929292929293,0.4292929292929293,0.0), 
    RGB{Float64}(0.9494949494949495,0.4494949494949495,0.0), 
    RGB{Float64}(0.9696969696969697,0.4696969696969697,0.0), 
    RGB{Float64}(0.98989898989899,0.48989898989898994,0.0), 
    RGB{Float64}(1.0,0.5101010101010102,0.010101010101010166), 
    RGB{Float64}(1.0,0.5303030303030303,0.030303030303030276), 
    RGB{Float64}(1.0,0.5505050505050506,0.05050505050505061), 
    RGB{Float64}(1.0,0.5707070707070707,0.07070707070707072), 
    RGB{Float64}(1.0,0.5909090909090908,0.09090909090909083), 
    RGB{Float64}(1.0,0.6111111111111112,0.11111111111111116), 
    RGB{Float64}(1.0,0.6313131313131313,0.13131313131313127), 
    RGB{Float64}(1.0,0.6515151515151516,0.1515151515151516), 
    RGB{Float64}(1.0,0.6717171717171717,0.1717171717171717), 
    RGB{Float64}(1.0,0.6919191919191918,0.19191919191919182),
    RGB{Float64}(1.0,0.7121212121212122,0.21212121212121215), 
    RGB{Float64}(1.0,0.7323232323232323,0.23232323232323226), 
    RGB{Float64}(1.0,0.7525252525252526,0.2525252525252526), 
    RGB{Float64}(1.0,0.7727272727272727,0.2727272727272727), 
    RGB{Float64}(1.0,0.792929292929293,0.29292929292929304), 
    RGB{Float64}(1.0,0.8131313131313131,0.31313131313131315), 
    RGB{Float64}(1.0,0.8333333333333333,0.33333333333333326),
    RGB{Float64}(1.0,0.8535353535353536,0.3535353535353536), 
    RGB{Float64}(1.0,0.8737373737373737,0.3737373737373737), 
    RGB{Float64}(1.0,0.893939393939394,0.39393939393939403), 
    RGB{Float64}(1.0,0.9141414141414141,0.41414141414141414), 
    RGB{Float64}(1.0,0.9343434343434343,0.43434343434343425), 
    RGB{Float64}(1.0,0.9545454545454546,0.4545454545454546), 
    RGB{Float64}(1.0,0.9747474747474747,0.4747474747474747), 
    RGB{Float64}(1.0,0.994949494949495,0.49494949494949503), 
    RGB{Float64}(1.0,1.0,0.5151515151515151), 
    RGB{Float64}(1.0,1.0,0.5353535353535352), 
    RGB{Float64}(1.0,1.0,0.5555555555555556), 
    RGB{Float64}(1.0,1.0,0.5757575757575757), 
    RGB{Float64}(1.0,1.0,0.595959595959596), 
    RGB{Float64}(1.0,1.0,0.6161616161616161), 
    RGB{Float64}(1.0,1.0,0.6363636363636365), 
    RGB{Float64}(1.0,1.0,0.6565656565656566), 
    RGB{Float64}(1.0,1.0,0.6767676767676767), 
    RGB{Float64}(1.0,1.0,0.696969696969697), 
    RGB{Float64}(1.0,1.0,0.7171717171717171), 
    RGB{Float64}(1.0,1.0,0.7373737373737375), 
    RGB{Float64}(1.0,1.0,0.7575757575757576), 
    RGB{Float64}(1.0,1.0,0.7777777777777777), 
    RGB{Float64}(1.0,1.0,0.797979797979798), 
    RGB{Float64}(1.0,1.0,0.8181818181818181), 
    RGB{Float64}(1.0,1.0,0.8383838383838385), 
    RGB{Float64}(1.0,1.0,0.8585858585858586), 
    RGB{Float64}(1.0,1.0,0.8787878787878789), 
    RGB{Float64}(1.0,1.0,0.898989898989899), 
    RGB{Float64}(1.0,1.0,0.9191919191919191), 
    RGB{Float64}(1.0,1.0,0.9393939393939394), 
    RGB{Float64}(1.0,1.0,0.9595959595959596), 
    RGB{Float64}(1.0,1.0,0.9797979797979799), 
    RGB{Float64}(1.0,1.0,1.0)] begin
        # add some more black color for the outside of the poster 
        reverse
        append!(
            RGB{Float64}[
                RGB{Float64}(0, 0, 0)
                for i in 1:50
            ]
        )
    end


nothing