module Metrics

export AbstractMetrics, BinaryAccuracy, CategoricalAccuracy, compute_accuracy

using ..Autodiff

"""
    AbstractMetrics

Every metrics struct is a subtype of AbstractMetrics
"""
abstract type AbstractMetrics end


"""
    BinaryCrossentropy

Represents the binary accuracy metric

# Field
- threshold: The threshold used to decide if the output is 0 or 1. Every predictions > threshold is set to 1
"""
struct BinaryAccuracy <: AbstractMetrics
    threshold::Float64
end

"""
    CategoricalAccuracy

Represents the categorical accuracy metric

"""
struct CategoricalAccuracy <: AbstractMetrics end

#TODO: implement classic accuracy to be able to compute accuracy for regression tasks

"""
    compute_accuracy(metrics::BinaryAccuracy, predictions::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    compute_accuracy(metrics::CategoricalAccuracy, predictions::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})

Compute the accuracy according to the metrics given.
"""
function compute_accuracy(metrics::BinaryAccuracy, predictions::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    size(predictions) == size(target) || throw("The predictions must have the same size of the target")

    # every predictions > threshold is set to 1
    adjustedPredictions = predictions.data .> metrics.threshold

    accuracy = 0
    # For scalars, just check if they are equal
    if (ndims(predictions) == 1 || size(predictions)[2] == 1)
        accuracy = convert(Int64, adjustedPredictions == target)
    else
        batchSize = size(predictions)[2]
        # Create a binary vector that contains 1 where the prediction is equal to target
        binaryVec = [adjustedPredictions[:, i] == target[:, i] for i = 1:batchSize]
        # Sum the correct predictions and divide it by the total number of predictions
        accuracy = sum(binaryVec) / batchSize
    end

    return accuracy
end

function compute_accuracy(metrics::CategoricalAccuracy, predictions::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    size(predictions) == size(target) || throw("The predictions must have the same size of the target")

    accuracy = 0
    if (ndims(predictions) == 1 || size(predictions)[2] == 1)
        # For scalars, just check if their argmax is equal
        accuracy = convert(Int64, argmax(predictions.data) == argmax(target))
    else
        batchSize = size(predictions)[2]
        # Create a binary vector that contains 1 where the argmax of the prediction is equal to the argmax of the target
        # if argmax are equal, the category predicted is correct.
        binaryVec = [argmax(predictions.data[:, i]) == argmax(target[:, i]) for i = 1:batchSize]
        accuracy = sum(binaryVec) / batchSize
    end
    return accuracy
end


end
