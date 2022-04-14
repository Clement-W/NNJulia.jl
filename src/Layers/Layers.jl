module Layers

# Each layer needs to implement the function parameters(layer)

abstract type AbstractLayer end
abstract type AbstractModel end

export AbstractLayer, AbstractModel, Dense, Sequential, add!, parameters, Conv2D, Flatten

# two dots to import from parent module
using ..Autodiff

include("Dense.jl")
include("Sequential.jl")
include("Conv2D.jl")
include("Flatten.jl")

end