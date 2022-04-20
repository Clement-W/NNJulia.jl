module Layers

# Each layer needs to implement the function parameters(layer)

"""
    AbstractLayer

Every layer is a subtype of this abstract type.
"""
abstract type AbstractLayer end

"""
    AbstractModel

Every trainable model is a subtype of this abstract type.
"""
abstract type AbstractModel end

export AbstractLayer, AbstractModel, Dense, Sequential, add!, parameters, Conv2D, Flatten

# two dots to import from parent module
using ..Autodiff

include("Dense.jl")
include("Sequential.jl")
include("Conv2D.jl")
include("Flatten.jl")

end