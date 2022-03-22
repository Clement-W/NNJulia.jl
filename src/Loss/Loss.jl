module Loss

export MSE, BinaryCrossentropy

using ..Autodiff

t = Tensor(3)

#mse (regression), binarycrossentropy (classification), etc.

function MSE(predicted::Tensor, target::Tensor)
    #L(y,y^i)=∑(y−y^​i​)^2
    errors = predicted .- target
    return sum(errors .* errors)
end


function BinaryCrossentropy(predicted::Tensor, target::Tensor)
    # https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/binary-crossentropy
    f = -1 / length(predicted)
    return f .* sum((target .* log(predicted) .+ (1 .- target) .* log(1 .- predicted)))
end


end
