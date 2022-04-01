module Metrics

export AbstractMetrics, Accuracy, BinaryAccuracy, CategoricalAccuracy, compute_accuracy

using ..Autodiff

abstract type AbstractMetrics end

struct Accuracy <: AbstractMetrics end

struct BinaryAccuracy <: AbstractMetrics
    threshold::Float64
end

struct CategoricalAccuracy <: AbstractMetrics end


function compute_accuracy(metrics::Accuracy, predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    #TODO:
    return 0
end

function compute_accuracy(metrics::BinaryAccuracy, predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    size(predicted) == size(target) || throw("The predictions must have the same size of the target")

    # every predictions > threshold is set to 1
    adjustedPredictions = predicted.data .> metrics.threshold

    accuracy = 0
    if (ndims(predicted) == 1 || size(predicted)[2] == 1)
        accuracy = convert(Int64, adjustedPredictions == target)
    else
        batchSize = size(predicted)[2]
        binaryVec = [adjustedPredictions[:, i] == target[:, i] for i = 1:batchSize]
        accuracy = sum(binaryVec) / batchSize
    end

    return accuracy
end

function compute_accuracy(metrics::CategoricalAccuracy, predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    return 0
end

#This metric creates two local variables, total and count that are used to compute the frequency with which y_pred matches y_true
function accuracy(predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    #mean(onecold(ŷ) .== onecold(y)
    return 0
end

function binary_accuracy(predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64}, treshold::Float64=0.5)

    return 0
end

function categorical_accuracy(predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    return 0
end


end
