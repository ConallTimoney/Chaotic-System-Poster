##
using DynamicalSystems
using DataFrames
using DataFramesMeta
using Chain
using CSV
using ProgressMeter
using Unitful
include("./functions.jl")
##
#=
we will recreate figure 5 (e) in this paper 
https://www.ijert.org/research/some-dynamical-properties-of-the-duffing-equation-IJERTV5IS120339.pdf

duffing equation in paper 
ẍ + γ(p1+p2x)ẋ + αx + βx³ = βFcos(ωt) 
 
ẍ = βFcos(ωt) - γ(p1+p2x)ẋ - αx - βx³

ẋ = (βFcos(ωt) - ẍ- αx - βx³)/
    γ(p1 + p2x)

x = (β(Fcos(ωt) - x³) - ẍ - γp1ẋ)/
    (γp2ẋ + α)
 
we need to create a custom dynamical system as the duffing oscillator included in the paper does 
have the same form as duffing oscillator incuded in DynamicalSystems 

We will modify the duffing oscilliator in the source code of DynamicalSystems
=#

# -- config options --  
##
power = 0.00000001 # adjusts how much the resulting poster glows 
extra_black_levels = 50 # number of extra black colors to add to the color gradient 
N = 2 * 1e6 # number of cyclic iterations to complete 
ω = 1.2 # drving ferquency of the oscillator

# height and width of the poster/canvas
width = (60u"cm" |> u"inch") / u"inch"
height = (40u"cm" |> u"inch") / u"inch"
dpi = 300
# R² - R > |c|
y_pixels = (height * dpi + 1) |> Float64 |> floor |> Int
x_pixels = (width * dpi + 1) |> Float64 |> floor |> Int

# Here is the our modified duffing oscillator
function modified_duffing(u0 = [1, 0.00001], ω = ω, f = 14.86, 
                          γ = 0.1, p1 = 1, p2 = 0.4, α = -2, 
                          β = 2)
    return ContinuousDynamicalSystem(modified_duffing_eom, u0, [ω, f, γ, p1, p2, α, β])
end
function modified_duffing_eom(x, p, t)
    # ω, f, d, β = p
    ω, f, γ, p1, p2, α, β = p
    dx1 = x[2]
    dx2 = β*f*cos(ω*t) - α*x[1] - β*x[1]^3 - γ*(p1 + p2*x[1])*x[2]
    return SVector{2}(dx1, dx2)
end



function modified_duffing_with_jacob(u0 = [1, 0.00001], ω = ω, f = 14.86, 
                                     γ = 0.1, p1 = 1, p2 = 0.4, α = -2, 
                                     β = 2)
    t = 0
    x = u0[1]
    ẋ = u0[2]
    ẍ = β*f*cos(ω*t) - γ*(p1+p2*x)*ẋ - α*x - β*x^3

    J = zeros(eltype(u0), 2, 2)
    J[1,1] = (p2*(ẍ + α*x + β*x^3 - β*f*cos(ω*t)) - (p1 + p2*x)*(α + 3*β^2))/(γ*(p1 + p2*x)^2)
    J[1,2] = 1
    J[2,1] = -γ*p2*ẋ-α-3*β^2
    J[2,2] = -γ*(p1 + p2*x)
    
    return ContinuousDynamicalSystem(modified_duffing_eom, u0, [ω, f, γ, p1, p2, α, β], modified_duffing_jacob, J)
end
@inbounds function modified_duffing_jacob(u, p, t)
    ω, f, γ, p1, p2, α, β = p
    x = u[1]
    ẋ = u[2]
    ẍ = β*f*cos(ω*t) - γ*(p1+p2*x)*ẋ - α*x - β*x^3
    return @SMatrix [
        (p2*(ẍ + α*x + β*x^3 - β*f*cos(ω*t)) - (p1 + p2*x)*(α + 3*β^2))/( γ*(p1 + p2*x)^2)                 1 ;
        -γ*p2*ẋ-α-3*β^2                                                                                   -γ*(p1 + p2*x)
    ]
end

##

# poincare_data = make_poincare_data(;N = N)

# mkpath("./sim_data")
# CSV.write("./sim_data/" * 
#           "mod_duffing,N=$N.csv",
#           DataFrame(poincare_data))

##

poincare_data = CSV.read(
    "./sim_data/mod_duffing,N=$N.csv",
    DataFrame
)

##

matrix_to_plot = distances_matrix(poincare_data,
                                  x_pixels, 
                                  y_pixels,
                                  x_range_mulitplier = 1.1,
                                  y_range_mulitplier = 1.1,
                                  y_column = 2)

##
mkpath("./plots_data")
CSV.write("./plots_data/" * 
          "mod_duffing,N=$N,x=$x_pixels.csv",
          DataFrame(matrix_to_plot))
##

matrix_to_plot = CSV.read("./plots_data/mod_duffing,N=$N,x=$x_pixels.csv",
                          DataFrame) |>
                    Array


plot = faded_plot(matrix_to_plot
                  , x_pixels
                  , y_pixels 
                  , power  = 0.08
                  , extra_black_levels = extra_black_levels 
                  , calculate_distances = false
                  );
plot
# matrix_to_plot = nothing 

##
mkpath("duffing_plots")
savefig(
    plot
    ,"./duffing_plots/mod_duffing,power=$power,black_levels=$extra_black_levels,x=$x_pixels,N=$N.svg"
)


savefig(
    plot
    ,"./duffing_plots/mod_duffing,power=$power,black_levels=$extra_black_levels,x=$x_pixels,N=$N.png"
)




savefig(
    plot
    ,"./duffing_plots/mod_duffing,power=$power,black_levels=$extra_black_levels,x=$x_pixels,N=$N.ps"
)
