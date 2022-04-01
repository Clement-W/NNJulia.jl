module Metrics

export AbstractMetrics, Accuracy, BinaryAccuracy, CategoricalAccuracy, compute_accuracy

using ..Autodiff

abstract type AbstractMetrics end

struct Accuracy <: AbstractMetrics end

struct BinaryAccuracy <: AbstractMetrics end

struct CategoricalAccuracy <: AbstractMetrics end


function compute_accuracy(metrics::Accuracy, predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    return 0
end
function compute_accuracy(metrics::BinaryAccuracy, predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    return 0
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
