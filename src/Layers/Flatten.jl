using .Autodiff

"""
    Flatten()

This struct represents a Flatten layer 
It flatten the input data into a 2 dimentional tensor of shape (n,batchSize)
with n = the multiplication of every other dimension's size (except the batchsize) of the original tensor
This operation preserve the size of the last dimension.
"""
struct Flatten <: AbstractLayer end

"""
    Base.show(io::IO, d::Flatten)

String representation of a Flatten layer
"""
function Base.show(io::IO, d::Flatten)
    println("Flatten layer.")
end

"""
    zero_grad!(d::Flatten)

Does nothing for a flatten layer
"""
function Autodiff.zero_grad!(d::Flatten) end

"""
    parameters(d::Flatten) 

Does nothing for a flatten layer
"""
function parameters(d::Flatten)
    return []
end


"""
    (a::Flatten)(x::Union{Tensor,AbstractArray})

The Flatten struct is callable, and flatten the input data
"""
function (a::Flatten)(x::Union{Tensor,AbstractArray})
    return flatten(x)
end

function flatten(x::Tensor)
    data = flatten(x.data)
    dependencies = x.requires_grad ? TensorDependency[] : nothing
    gradient = x.requires_grad ? flatten(x.gradient) : nothing

    if (x.requires_grad)
        # the gradient function unflatten the incoming gradient to the size of the input tensor
        gradientFunction(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = reshape(incomingGradient, size(x))
        push!(dependencies, TensorDependency(x, gradientFunction))
    end

    return Tensor(data, gradient, dependencies, x.requires_grad)
end

function flatten(x::AbstractArray)
    return reshape(x, :, size(x)[end])
end

