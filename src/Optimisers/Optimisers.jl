module Optimisers

"""
    AbstractOptimiser

Every optimiser struct is a subtype of AbstractOptimiser
"""
abstract type AbstractOptimiser end

export AbstractOptimiser, GradientDescent, update!

using ..Layers

include("GradientDescent.jl")

end