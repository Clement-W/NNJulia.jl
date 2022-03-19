module Layers

abstract type AbstractLayer end
abstract type AbstractModel end

export Dense, Sequential

# two dots to import from parent module
using ..Autodiff

include("Dense.jl")
include("Sequential.jl")

end