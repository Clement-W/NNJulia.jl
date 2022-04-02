module Metrics

export AbstractMetrics, Accuracy, BinaryAccuracy, CategoricalAccuracy, compute_accuracy

using ..Autodiff

abstract type AbstractMetrics end

struct Accuracy <: AbstractMetrics end

struct BinaryAccuracy <: AbstractMetrics
    threshold::Float64
end

struct CategoricalAccuracy <: AbstractMetrics end


function compute_accuracy(metrics::Accuracy, predictions::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    #TODO:
    return 0
end

function compute_accuracy(metrics::BinaryAccuracy, predictions::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    size(predictions) == size(target) || throw("The predictions must have the same size of the target")

    # every predictions > threshold is set to 1
    adjustedPredictions = predictions.data .> metrics.threshold

    accuracy = 0
    if (ndims(predictions) == 1 || size(predictions)[2] == 1)
        accuracy = convert(Int64, adjustedPredictions == target)
    else
        batchSize = size(predictions)[2]
        binaryVec = [adjustedPredictions[:, i] == target[:, i] for i = 1:batchSize]
        accuracy = sum(binaryVec) / batchSize
    end

    return accuracy
end

function compute_accuracy(metrics::CategoricalAccuracy, predictions::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    size(predictions) == size(target) || throw("The predictions must have the same size of the target")

    accuracy = 0
    if (ndims(predictions) == 1 || size(predictions)[2] == 1)
        accuracy = convert(Int64, argmax(predictions.data) == argmax(target))
    else
        batchSize = size(predictions)[2]
        binaryVec = [argmax(predictions.data[:, i]) == argmax(target[:, i]) for i = 1:batchSize]
        accuracy = sum(binaryVec) / batchSize
    end
    return accuracy
end


end
