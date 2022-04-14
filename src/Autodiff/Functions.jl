# This file contains every functions that can be applied on tensors


"""
    Base.sum(t::Tensor)

Return the sum of the tensor's elements. 
The tensor returned requires gradient if the initial tensor requires it.

For the gradient function, incomingGradient is a one element tensor, because the output of the sum is a 
scalar tensor. In the sum function, each element has the same weight 
(1*x1 + 1*x2 + ... + 1*xn), so the gradient of this tensor wrt to the sum tensor
is a tensor composed of ones, with the shape of the original tensor.
"""
function Base.sum(t::Tensor)

    data = sum(t.data)

    if (t.requires_grad)

        # Function used to compute the gradient of t :
        gradientFunction(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = incomingGradient * ones(size(t.data))
        #=
        incomingGradient is a one element tensor, because the output of the sum is a 
        one element tensor. In the sum function, each element has the same weight 
        (1*x1 + 1*x2 + ... + 1*xn), so the gradient of this tensor wrt to the sum tensor
        is a tensor composed of ones, with the shape of the original tensor.
        d(grad)/d(thisTensor) = d(grad)/d(sum) * d(sum)/d(thisTensor) = grad * (1,1,1,...)
        =#

        dependencies = [TensorDependency(t, gradientFunction)]
    else
        dependencies = nothing
    end

    return Tensor(data, dependencies)
end

"""
    Base.:log(t1::Tensor)

Log function to perform element-wise neperian logarithm on a tensor.
The tensor returned requires gradient if the initial tensor requires it.

- d(ln(t1))/d(t1) = 1/t1 --> multiply the incoming gradient by 1/t1.
"""
function Base.:log(t1::Tensor)

    data = log.(t1.data)

    if (t1.requires_grad)
        # Function used to compute the gradient of t1 :
        gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = incomingGradient .* (1 ./ t1.data)
        # d(ln(t1))/d(t1) = 1/t1, so we just need to multiply the incoming gradient by 1/t1.
        dependencies = [TensorDependency(t1, gradientFunctionT1)]
    else
        dependencies = nothing
    end

    return Tensor(data, dependencies)
end

"""
    Base.:tanh(t1::Tensor) 

Tanh function to perform elemnt-wise tanh on a tensor.
The tensor returned requires gradient if the initial tensor requires it.

- d(tanh(t1))/d(t1) = (1-tanh^2(t1)) --> multiply the incoming gradient by (1-tanh^2(t1))
"""
function Base.:tanh(t1::Tensor)

    data = tanh.(t1.data)

    if (t1.requires_grad)
        # Function used to compute the gradient of t1 :
        gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = incomingGradient .* (1 .- data .^ 2)
        # derivataive of tanh(x) is (1-tanh^2(x))
        dependencies = [TensorDependency(t1, gradientFunctionT1)]
    else
        dependencies = nothing
    end

    return Tensor(data, dependencies)
end

"""
    sigmoid(t1::Tensor) 

Sigmoid function to perform elemnt-wise sigmoid on a tensor.
The tensor returned requires gradient if the initial tensor requires it.

- d(sigmoid(t1))/d(t1) = sigmoid(t1)*(1-sigmoid(t1)) --> multiply the incoming gradient by sigmoid(t1)*(1-sigmoid(t1))
"""
function sigmoid(t1::Tensor)

    data = 1 ./ (1 .+ exp.(.-t1.data))

    if (t1.requires_grad)
        # Function used to compute the gradient of t1 :
        gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = incomingGradient .* (data .* (1 .- data))
        # derivataive of sigmoid(x) is sigmoid(x)*(1-sigmoid(x))
        dependencies = [TensorDependency(t1, gradientFunctionT1)]
    else
        dependencies = nothing
    end

    return Tensor(data, dependencies)
end

"""
    relu(t1::Tensor) 

Relu function to perform elemnt-wise relu on a tensor.
The tensor returned requires gradient if the initial tensor requires it.

- d(relu(t1))/d(t1) =  1 if t1>0, else 0 --> multiply the incoming gradient by (t1 .> 0)
"""
function relu(t1::Tensor)

    data = max.(zero(t1.data), t1.data)

    if (t1.requires_grad)
        # Function used to compute the gradient of t1 :
        gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = incomingGradient .* ((t1.data .> 0) .* 1)
        # derivataive of relu(x) is 1 if x>0, else it is 0
        # ( x .> 0) create a boolean, or a matrix or vector of boolean, multiplying it by 1 convert it to Int64
        dependencies = [TensorDependency(t1, gradientFunctionT1)]
    else
        dependencies = nothing
    end

    return Tensor(data, dependencies)
end

"""
    leakyrelu(t1::Tensor) 

leaky relu function to perform elemnt-wise leaky relu on a tensor.
The tensor returned requires gradient if the initial tensor requires it.

- d(leakyrelu(t1,a))/d(t1) =  1 if t1>0, else a --> multiply the incoming gradient by 1 or a depending on the data
"""
function leakyrelu(t1::Tensor, a=0.01)

    # if x > 0, leakyrelu(x,a) = x
    # if x<= 0, leakyrlu(x,a) = a*x
    # element-wise if else condition on the data :
    data = ifelse.(t1.data .> 0, float.(t1.data), a .* t1.data)

    if (t1.requires_grad)
        # Function used to compute the gradient of t1 :
        gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = float(incomingGradient) .* (ifelse.(t1.data .> 0, one.(t1.data), a))
        # derivative of leakyrelu(x,a) is 1 if x>0, else it is a
        dependencies = [TensorDependency(t1, gradientFunctionT1)]
    else
        dependencies = nothing
    end

    return Tensor(data, dependencies)
end



"""
    softmax(t1::Tensor) 

Softmax function to perform softmax on a tensor.
The tensor returned requires gradient if the initial tensor requires it.

- d(softmax(t1))/d(t1) = softmax(t1)*(1-softmax(t1)) --> multiply the incoming gradient by softmax(t1)*(1-softmax(t1))
"""
function softmax(t1::Tensor)

    probas = exp.(t1.data .- maximum(t1.data, dims=1))
    data = probas ./ sum(probas, dims=1)

    if (t1.requires_grad)
        # Function used to compute the gradient of t1 :
        gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = incomingGradient .* (data .* (1 .- data))
        # derivataive of softmax(x) is softmax(x) .* (1 - softmax(x))
        dependencies = [TensorDependency(t1, gradientFunctionT1)]
    else
        dependencies = nothing
    end

    return Tensor(data, dependencies)
end