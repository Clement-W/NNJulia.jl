module Loss

export MSE, BinaryCrossentropy

using ..Autodiff

function MSE(predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    #L(y,y^i)=∑(y−y^​i​)^2
    errors = predicted .- target
    return sum(errors .* errors)
end


function BinaryCrossentropy(predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})
    # https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/binary-crossentropy
    f = -1 / length(predicted)
    return f .* sum((target .* log(predicted) .+ (1 .- target) .* log(1 .- predicted)))
end


end
