module Optimisers

abstract type AbstractOptimiser end

export AbstractOptimiser, GradientDescent, update!

using ..Layers

include("GradientDescent.jl")

end