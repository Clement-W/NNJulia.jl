module Layers

#TODO: Each layer needs to implement the function parameters(layer)
# (needs to be specified in the documentation) 

abstract type AbstractLayer end
abstract type AbstractModel end

export AbstractLayer, AbstractModel, Dense, Sequential, add!, parameters, Convolution

# two dots to import from parent module
using ..Autodiff

include("Dense.jl")
include("Sequential.jl")
include("Convolution.jl")

end