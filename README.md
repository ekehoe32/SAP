# SAP
GPU accelerated julia module for computing secant avoidance projections (SAP) on large data sets.

## Requirements
* CUDA Toolkit v10.1 or greater (https://developer.nvidia.com/cuda-downloads)

* CUDA compatible GPU

* julia v1.5 or greater (https://julialang.org/downloads/)

## Installation
Simply clone the repository and then activate the environment in julia by calling 
    
    cd("/path/to/SAP/");
    Pkg.activate(".");
    using SAP;

## Example
Users can run the example script [SAP_example.jl](src/SAP_example.jl) located in the [src](src) directory to run SAP on a toy dataset. All figures and interative plots are contained in the [Figures](Figures) directory.

## Example Embedding
![Indian Pines HSI SAP](https://github.com/ekehoe32/SAP/tree/main/Figures/Indian_Pines_Hyperspectral_SAP_type_q_3.png?raw=true)
