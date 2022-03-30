# This file contains every functions that can be applied on tensors

# Return the sum of the tensor's elements
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

# log function to perform element-wise neperian logarithm on a tensor
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



# tanh function to perform element-wise tanh on a tensor
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

# sigmoid function to perform element-wise sigmoid on a tensor
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

# relu function to perform element-wise relu on a tensor
function relu(t1::Tensor)

    data = max.(zero(t1.data), t1.data)

    if (t1.requires_grad)
        # Function used to compute the gradient of t1 :
        gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = incomingGradient .* ((t1.data .> 0) .* 1)
        # derivataive of relu(x) is 1 if x>0, else it is 0
        # ( x .> 0) create a boolean, or a matrix or vector of boolean, multiplying it by 1 convert it  
        dependencies = [TensorDependency(t1, gradientFunctionT1)]
    else
        dependencies = nothing
    end

    return Tensor(data, dependencies)
end

# leaky relu function to perform element-wise leaky relu on a tensor
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


