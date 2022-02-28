# To counter mutually recursive fields between TensorDependency and Tensor
abstract type AbstractTensor end

struct TensorDependency
    tensorDep::AbstractTensor
    gradFunction::Function
end


# To improve performances, it would be better to have an immutable struct...
# To improve performances, it would be better to use concrete types for AbstractArray ({Union{Float64,Int64}})
mutable struct Tensor{T<:Union{AbstractArray,Float64,Int64}} <: AbstractTensor
    data::T
    gradient::T
    dependencies::Union{Vector{TensorDependency},Nothing}
    # main constructor
    function Tensor(data::T, gradient::T, dependencies::Union{Vector{TensorDependency},Nothing}) where {T<:Union{AbstractArray,Float64,Int64}}
        if (size(data) != size(gradient))
            throw(ErrorException("The gradient's size must be equal to the size of the data."))
        end
        new{Union{AbstractArray,Float64,Int64}}(data, gradient, dependencies)
    end
end

# Aditional constructors
Tensor(data::T) where {T<:Union{AbstractArray,Float64,Int64}} = Tensor(data, zero(data), nothing)

Tensor(data::T, gradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = Tensor(data, gradient, nothing)

Tensor(data::T, dependencies::Union{Vector{TensorDependency},Nothing}) where {T<:Union{AbstractArray,Float64,Int64}} = Tensor(data, zero(data), dependencies)

# customize the set property for t.data
function Base.setproperty!(t::Tensor, prop::Symbol, val)
    if (prop == :data)
        # Reset the gradient to 0 if the data is set mannualy
        t.gradient = zero(t.data)
        setfield!(t, :data, val)
    else
        setfield!(t, prop, val)
    end
end

# return the size of the data
Base.size(t::Tensor) = size(t.data)

# return the number of dims of the data
Base.ndims(t::Tensor{Union{Float64,Int64,AbstractArray}}) = ndims(t.data)


# set the gradient to 0 
function zero_grad!(t::Tensor)
    t.gradient = zero(t.data)
end

# string representation of a tensor
function Base.show(io::IO, t::Tensor)
    println(io, "Tensor", size(t))
    print(io, "data : ")
    show(t.data)
    println()
    print(io, "grad : ")
    show(t.gradient)
end

# Backpropagate a gradient through the auto differenciation graph by
# recurcively calling this method on the tensor dependencies.
function backward!(t::Tensor, incomingGradient::Union{T,Nothing} = nothing) where {T<:Union{AbstractArray,Float64,Int64}}

    if (incomingGradient === nothing)
        # If the tensor is a scalar, and no incoming gradient is provided, then set it to 1
        if (ndims(t)) == 0
            incomingGradient = 1
        else
            throw(ErrorException("Incoming gradient must be specified for non-scalar tensors"))
        end
    end

    # Add the incoming gradient to the current tensor gradient (initially set to 0)
    # Adding the incoming gradient allow gradient accumulation
    t.gradient += incomingGradient

    # Loop recursively into each dependencies of the current tensor to go through the whole graph
    if (t.dependencies !== nothing)
        for dep::TensorDependency in t.dependencies
            # compute the gradient of the dependency wrt to this tensor
            backwardGrad = dep.gradFunction(incomingGradient)
            # backward this gradient to the dependency
            backward!(dep.tensorDep, backwardGrad)
        end
    end

end

# Return the sum of the tensor's elements
function Base.sum(t::Tensor)

    # Function used to compute the gradient of t :
    gradientFunction(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = incomingGradient * ones(size(t.data))
    #=
    incomingGradient is a one element tensor, because the output of the sum is a 
    one element tensor. In the sum function, each element has the same weight 
    (1*x1 + 1*x2 + ... + 1*xn), so the gradient of this tensor wrt to the sum tensor
    is a tensor composed of ones, with the shape of the original tensor.
    d(grad)/d(thisTensor) = d(grad)/d(sum) * d(sum)/d(thisTensor) = grad * (1,1,1,...)
    =#

    data = sum(t.data)
    dependencies = [TensorDependency(t, gradientFunction)]

    return Tensor(data, dependencies)
end

# + operator for tensors to support addition between 2 tensors
function Base.:+(t1::Tensor, t2::Tensor)

    # Function used to compute the gradient of t1 :
    gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = incomingGradient
    # d(t1+t2)/d(t1) = 1, so we just need to multiply the incoming gradient by 1.


    # Function used to compute the gradient of t2 :
    gradientFunctionT2(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = incomingGradient
    # d(t1+t2)/d(t2) = 1, so we just need to multiply the incoming gradient by 1.


    data = t1.data + t2.data
    dependencies = [TensorDependency(t1, gradientFunctionT1), TensorDependency(t2, gradientFunctionT2)]

    return Tensor(data, dependencies)
end


# Used to support gradient computation with broadcast operations made with element-wise operators such as .+
function handleBroadcasting(t::Tensor, gradient::T) where {T<:Union{AbstractArray,Float64,Int64}}
    #=
    First, sum out the dims added by the broadcast operation, so that the gradient
    has the same dimensions of the tensor
    This will handle this example : [[1,2],[3,4]] .+ [2,2] = [[3,4],[5,6]]
    Here, the gradient of [2,2] will be the sum on the first axis (sum the columns) of the gradient of [[3,4],[5,6]]
    =#
    # If nbDimsAdded is positive, the tensor is smaller than the gradient, so it has been broadcasted to apply the operation
    nbDimsAdded = ndims(gradient) - ndims(t)

    for _ = 1:nbDimsAdded
        # sum the first axis, and remove the additional dimension
        gradient = dropdims(sum(gradient, dims = 1), dims = 1)
        if (size(gradient) == ())
            # if the gradient is a one element array, convert it to a scalar
            gradient = gradient[1]
        end

    end

    #=
    Now, to deal with this case :  [[1,2],[3,4]] .+ [[2,2]] = [[3,4],[5,6]]
    where the operation is broadcasted but no dimension is added, we'll need to sum the 
    broadcasted dims by keeping the dimensions.
    =#

    # For each dimension
    for i = 1:ndims(t)

        # If the dimension is equal to 1, it means that the operation is broadcasted along this axis
        # If it's a scalar, it doesn't change anything 
        if (size(t)[i] == 1)
            gradient = sum(gradient, dims = i)
        end
    end

    return gradient
end


# Element-wise addition (perform broadcast operation)
# TODO: implement inplace broadcast operation https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting-1
function Base.:broadcasted(::typeof(+), t1::Tensor, t2::Tensor)

    # Function used to compute the gradient of t1 :
    gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = handleBroadcasting(t1, incomingGradient)
    # d(t1+t2)/d(t1) = 1, so we just need to multiply the incoming gradient by 1. (also supports broadcasting)


    # Function used to compute the gradient of t2 :
    gradientFunctionT2(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = handleBroadcasting(t2, incomingGradient)
    # d(t1+t2)/d(t2) = 1, so we just need to multiply the incoming gradient by 1. (also supports broadcasting)

    data = t1.data .+ t2.data
    dependencies = [TensorDependency(t1, gradientFunctionT1), TensorDependency(t2, gradientFunctionT2)]

    return Tensor(data, dependencies)
end


# - operator for tensors to support substraction between 2 tensors
function Base.:-(t1::Tensor, t2::Tensor)

    # Function used to compute the gradient of t1 :
    gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = incomingGradient
    # d(t1-t2)/d(t1) = 1, so we just need to multiply the incoming gradient by 1.


    # Function used to compute the gradient of t2 :
    gradientFunctionT2(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = -incomingGradient
    # d(t1-t2)/d(t2) = -1, so we just need to multiply the incoming gradient by -1.

    data = t1.data - t2.data
    dependencies = [TensorDependency(t1, gradientFunctionT1), TensorDependency(t2, gradientFunctionT2)]

    return Tensor(data, dependencies)
end


# - operator for tensors to negate a tensor
function Base.:-(t1::Tensor)

    # Function used to compute the gradient of t1 :
    gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = -incomingGradient
    # d(-t1)/d(t1) = -1, so we just need to multiply the incoming gradient by 1.

    data = -t1.data
    dependencies = [TensorDependency(t1, gradientFunctionT1)]

    return Tensor(data, dependencies)
end



# Element-wise substraction (perform broadcast operation)
function Base.:broadcasted(::typeof(-), t1::Tensor, t2::Tensor)

    # Function used to compute the gradient of t1 :
    gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = handleBroadcasting(t1, incomingGradient)
    # d(t1-t2)/d(t1) = 1, so we just need to multiply the incoming gradient by 1. (also supports broadcasting)


    # Function used to compute the gradient of t2 :
    gradientFunctionT2(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = handleBroadcasting(t2, -incomingGradient)
    # d(t1-t2)/d(t2) = -1, so we just need to multiply the incoming gradient by -1. (also supports broadcasting)

    data = t1.data .- t2.data
    dependencies = [TensorDependency(t1, gradientFunctionT1), TensorDependency(t2, gradientFunctionT2)]

    return Tensor(data, dependencies)
end


# * operator for tensors to support multiplication and matrix multiplication between 2 tensors
function Base.:*(t1::Tensor, t2::Tensor)

    # Function used to compute the gradient of t1 :
    gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = incomingGradient * t2.data'
    #= With t1 (n1,m1), t2 (m1,m2) and t3 = t1 * t2 is (n1,m2)
    So the incoming gradient wrt t3 is (n1,m2)
    d(t1*t2)/d(t1) = t2
    So we just need to matmul the incoming gradient by t2. But t2 is (m1,m2)
    and the incoming gradient is (n1,m2). So we need to do grad @ t2.Transpose
    # This also works for scalars.
    =#


    # Function used to compute the gradient of t2 :
    gradientFunctionT2(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = t1.data' * incomingGradient
    #=
    d(t1*t2)/d(t2) = t1
    So we just need to matmul the incoming gradient by t1. But t1 is (n1,m1 )
    and the incoming gradient is (n1,m2). So we need to do t1.Transpose * grad
    =#

    data = t1.data * t2.data
    dependencies = [TensorDependency(t1, gradientFunctionT1), TensorDependency(t2, gradientFunctionT2)]

    return Tensor(data, dependencies)
end

# Element-wise multiplication (perform broadcast operation)
function Base.:broadcasted(::typeof(*), t1::Tensor, t2::Tensor)

    # Function used to compute the gradient of t1 :
    gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = handleBroadcasting(t1, t2.data .* incomingGradient)
    # d(t1*t2)/d(t1) = t2, so we just need to multiply the incoming gradient by t2. (also supports broadcasting)


    # Function used to compute the gradient of t2 :
    gradientFunctionT2(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = handleBroadcasting(t2, t1.data .* incomingGradient)
    # d(t1*t2)/d(t2) = t1, so we just need to multiply the incoming gradient by t1. (also supports broadcasting)

    data = t1.data .* t2.data
    dependencies = [TensorDependency(t1, gradientFunctionT1), TensorDependency(t2, gradientFunctionT2)]

    return Tensor(data, dependencies)
end
# TODO: test




#=
TODO:
mult .* and * operator
matmul * operator (use muladd ?)
div / operator
log operator
pow operator
sin,cos,tan,tanh ?
slice function
=#



