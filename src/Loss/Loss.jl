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
    part1 = (1 .- target)
    part2 = log(1 .- predicted)
    # For an unknown reason, it does not return a Tensor if I don't put part1 and part2 into 2 different variables
    # (1 .- target) .* log(1 .- predicted) don't return a tensor
    # but part1 .* part2 do return a tenor
    res = f .* sum((target .* log(predicted)) .+ (part1 .* part2))

    return res
end


end
