# Poster
This repo uses Julia's DynamicalSystems.jl to simulate a modified non linear duffing oscillator and create a phase portrait of the resulting motion. The oscillator is described in more detail in this [paper](https://www.ijert.org/research/some-dynamical-properties-of-the-duffing-equation-IJERTV5IS120339.pdf). The resulting image is the distance to the nearest point in the phase portrait to a power specified by the user. Hence the heatmap is of
$$
distance^{power}.
$$
The distance is found using a KDTree from NearestNeighbors.jl. 

This repo also contains code to create and plot julia sets using the same effect previously described. 

## Examples
![](plots/mod_duffing,power=1.0e-8,black_levels=50,x=1080,N=2.0e6.svg)
![](plots/../julia_sets/C=-0.835%20-%200.2321im/R=2/x=1620,y=1080.png)
![](plots/mod_duffing,power=0.001,black_levels=30,x=1080,N=2.0e6.svg) 

## Scripts
**functions.jl** Contains reuseable functions that allow quick prototyping of new plots with a variety of options. 
**modified_diffing_poincare** Creates a poincare plot of the oscillator. 