module Loss

export AbstractLoss, MSE, BinaryCrossentropy, CategoricalCrossentropy, compute_loss

using ..Autodiff

"""
    AbstractLoss

Every loss struct is a subtype of AbstractLoss
"""
abstract type AbstractLoss end

"""
    MSE

Represents the Mean Squared Error : L(y,y^i)=∑(y-y^i)^2
"""
struct MSE <: AbstractLoss end


"""
    BinaryCrossentropy

Represents the Binary crossentropy error function
"""
struct BinaryCrossentropy <: AbstractLoss end


"""
    CategoricalCrossentropy

Represents the Binary crossentropy error function
"""
struct CategoricalCrossentropy <: AbstractLoss end


"""
    compute_loss(lossF::MSE, predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    compute_loss(lossF::BinaryCrossentropy, predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    compute_loss(lossF::CategoricalCrossentropy, predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})

Compute the loss according to the lossF given.
"""
function compute_loss(lossF::MSE, predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    #L(y,y^i)=∑(y−y^​i​)^2
    errors = predicted .- target
    return sum(errors .* errors)
end

# Compute loss for BinaryCrossentropy
# FIXME: this implementation is mayble general enough to be used for Categorical data as well
function compute_loss(lossF::BinaryCrossentropy, predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    # https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/binary-crossentropy

    clamp!(predicted.data, 1e-7, (1 - 1e-7))

    f = -1 / length(predicted)
    part1 = (1 .- target)
    part2 = log(1 .- predicted)
    # For an unknown reason, it does not return a Tensor if I don't put part1 and part2 into 2 different variables
    # (1 .- target) .* log(1 .- predicted) don't return a tensor
    # but part1 .* part2 do return a tensor
    res = f .* sum((target .* log(predicted)) .+ (part1 .* part2))
    return res
end

# Compute loss for CategoricalCrossentropy
# FIXME: My implementation of CategoricalCrossentropy do not work 
function compute_loss(lossF::CategoricalCrossentropy, predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})

    clamp!(predicted.data, 1e-7, (1 - 1e-7))

    f = -1 / size(predicted)[end]
    res = f * sum(target .* log(predicted))
    return res
end



end #module