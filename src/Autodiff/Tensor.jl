"""
    AbstractTensor

This type is used to counter the circular dependency between TensorDependency and Tensor.
"""
abstract type AbstractTensor end

"""
    TensorDependency(tensorDep::AbstractTensor, gradFunction::Function)

This struct represents the dependence of a tensor. This is used to keep track of the tensor's dependencies.
For example, if a tensor is made up by the sum of 2 other tensor, this tensor will have 2 TensorDependency object
in it's list of dependency. This struct also stores the derivative of the operation linking the dependencies, to be
able to compute the gradient of the resul tensor, with respect to the  dependencies.

# Fields
- tensorDep: The tensor dependence
- gradFunction: This function is used to compute the gradient of the tensor that depends on TensorDep, with respect to the dependencies.
"""
struct TensorDependency

    tensorDep::AbstractTensor
    gradFunction::Function
end

"""
    Tensor(data::T, gradient::Union{T,Nothing}, dependencies::Union{Vector{TensorDependency},Nothing}, requires_grad::Bool) where {T<:Union{AbstractArray,Float64,Int64}}
    Tensor(data::T, requires_grad::Bool=false) where {T<:Union{AbstractArray,Float64,Int64}}
    Tensor(data::T, gradient::Union{T,Nothing}) where {T<:Union{AbstractArray,Float64,Int64}}
    Tensor(data::T, dependencies::Union{Vector{TensorDependency},Nothing}) where {T<:Union{AbstractArray,Float64,Int64}} 

This mutable struct represents a Tensor, it is a scalar or an array that supports gradient computation

# Fields 
- data: The data contained in the tensor as a scalar or an array
- gradient: A gradient with respect to this tensor
- dependencies: A list that contains the tensors on which the current tensor depends
- requires_grad: Boolean which indicates if the gradient has to be computed for this tensor
"""
mutable struct Tensor{T<:Union{AbstractArray,Float64,Int64}} <: AbstractTensor
    data::T
    gradient::Union{T,Nothing}
    dependencies::Union{Vector{TensorDependency},Nothing}
    requires_grad::Bool
    " Main constructor"
    function Tensor(data::T, gradient::Union{T,Nothing}, dependencies::Union{Vector{TensorDependency},Nothing}, requires_grad::Bool) where {T<:Union{AbstractArray,Float64,Int64}}
        if (requires_grad)
            if (size(data) != size(gradient))
                throw(ErrorException("The gradient's size must be equal to the size of the data."))
            end
        else
            if (gradient !== nothing || dependencies !== nothing)
                throw(ErrorException("The tensor can't have a gradient or dependencies if requires_grad is false."))
            end
        end
        new{Union{AbstractArray,Float64,Int64}}(data, gradient, dependencies, requires_grad)
    end
end

# Aditional constructors
Tensor(data::T, requires_grad::Bool=false) where {T<:Union{AbstractArray,Float64,Int64}} = Tensor(data, (requires_grad != false) ? zero(data) : nothing, nothing, requires_grad)
# if requires_grad is true, the gradient is set to zero, else it is set to nothing

Tensor(data::T, gradient::Union{T,Nothing}) where {T<:Union{AbstractArray,Float64,Int64}} = Tensor(data, gradient, nothing, gradient !== nothing)
# if gradient == nothing, requires_grad is set to false, else it is set to true

Tensor(data::T, dependencies::Union{Vector{TensorDependency},Nothing}) where {T<:Union{AbstractArray,Float64,Int64}} = Tensor(data, (dependencies !== nothing) ? zero(data) : nothing, dependencies, dependencies !== nothing)
# if dependencies == nothing, requires_grad is set to false and gradient is set to nothing, else it is set to true


# customize the set property for t.data
"""
    Base.setproperty!(t::Tensor, prop::Symbol, val)

If the data property is modified, the gradient is set to 0
"""
function Base.setproperty!(t::Tensor, prop::Symbol, val)
    if (prop == :data)
        # Reset the gradient to 0 if the data is set mannualy
        if (t.requires_grad)
            t.gradient = zero(t.data)
        end
        setfield!(t, :data, val)
    elseif (prop == :gradient)
        if (t.requires_grad == false)
            throw(ErrorException("Can't change the gradient of a tensor with requires_grad = false"))
        end
        size(t.data) == size(val) || error("The gradient must have the same shape as the data")
        setfield!(t, :gradient, val)
    else
        setfield!(t, prop, val)
    end
end


"""
    Base.size(t::Tensor)

Return the size of the tensor's data
"""
Base.size(t::Tensor) = size(t.data)


"""
    Base.ndims(t::Tensor{Union{Float64,Int64,AbstractArray}})

Return the number of dimensions of the tensor's data
"""
Base.ndims(t::Tensor{Union{Float64,Int64,AbstractArray}}) = ndims(t.data)

"""
    Base.length(t::Tensor{Union{Float64,Int64,AbstractArray}})

Return the length of the tensor's data
"""
Base.length(t::Tensor{Union{Float64,Int64,AbstractArray}}) = length(t.data)

"""
    Base.iterate(t::Tensor{Union{Float64,Int64,AbstractArray}})
    Base.iterate(t::Tensor{Union{Float64,Int64,AbstractArray}}, state)

Iterate on the tensor's data
"""
Base.iterate(t::Tensor{Union{Float64,Int64,AbstractArray}}) = iterate(t.data)
Base.iterate(t::Tensor{Union{Float64,Int64,AbstractArray}}, state) = iterate(t.data, state)

"""
    Base.show(io::IO, t::Tensor)

String representation of a tensor
"""
function Base.show(io::IO, t::Tensor)
    println(io, "Tensor", size(t))
    print(io, "data : ")
    show(t.data)
    println()
    print(io, "grad : ")
    show(t.gradient)
end

"""
    zero_grad!(t::Tensor)

Set the gradient with respect to this tensor to 0
"""
function zero_grad!(t::Tensor)
    if (t.requires_grad == false)
        throw(ErrorException("Can't call zero_grad on a tensor with requires_grad = false"))
    end
    t.gradient = zero(t.data)
end



"""
    backward!(t::Tensor, incomingGradient::Union{T,Nothing}=nothing) where {T<:Union{AbstractArray,Float64,Int64}}

Backpropagate a gradient through the auto differenciation graph by
recurcively calling this method on the tensor dependencies.
The gradient don't need to be specified if the current tensor is a scalar 
"""
function backward!(t::Tensor, incomingGradient::Union{T,Nothing}=nothing) where {T<:Union{AbstractArray,Float64,Int64}}

    if (t.requires_grad == false)
        throw(ErrorException("Can't call backward on a tensor that don't requires gradient"))
    end

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
    t.gradient = t.gradient .+ incomingGradient


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


"""
    handle_broadcasting!(t::Tensor, gradient::T) where {T<:Union{AbstractArray,Float64,Int64}}

Used to support gradient computation with broadcast operations made with broadcasted operators 

First, sum out the dims added by the broadcast operation, so that the gradient has the same dimensions of the tensor.
To compute the gradient when a dimension is added by the broadcast operation, 
the gradient is summed along the batch axis (the dimension added).
This will handle this example : [1 2 ; 3 4] .+ [2,2] = [3 4; 5 6]

Then, when the operation is broadcasted but no dimension is added, the 
broadcasted dims are summed by keeping the dimensions.
This will handle this example : [1 2 ; 3 4] .+ [2;2] = [3 4 ; 5 6]

"""
function handle_broadcasting!(t::Tensor, gradient::T) where {T<:Union{AbstractArray,Float64,Int64}}
    #=
    First, sum out the dims added by the broadcast operation, so that the gradient
    has the same dimensions of the tensor.
    To compute the gradient when a dimension is added by the broadcast operation, we need
    to sum the gradient along the batch axis (the dimension added).
    This will handle this example : [1 2 ; 3 4] .+ [2,2] = [3 4; 5 6]
    Here, the gradient of [2,2] will be the sum on the last axis of the gradient of [3 4; 5 6]
    =#

    # If nbDimsAdded is positive, the tensor is smaller than the gradient, so it has been broadcasted to apply the operation
    nbDimsAdded = ndims(gradient) - ndims(t)

    # for each demansion added, sum the added dimension
    for _ = 1:nbDimsAdded

        # If the tensor is a scalar, remove the last dimension of the gradient 
        # At the end of the loop, every dimensions will be summed so the gradient will also be a scalar
        addedDimIndex = ndims(gradient)


        # If the tensor is a scalar, this loop is useless
        if (ndims(t) > 0)
            # Find the added dimension, starting from the end
            for (ind, dimension) in enumerate(reverse(size(gradient)))
                # If the dimension in the gradient is not in the tensor, it is the added dimension
                if ((dimension in size(t) == false))
                    addedDimIndex = ind
                    break
                end
            end
        end

        # sum the axis of the added dimension, and remove the additional dimension
        gradient = dropdims(sum(gradient, dims=addedDimIndex), dims=addedDimIndex)
        if (size(gradient) == ())
            # if the gradient is a one element array, convert it to a scalar
            gradient = gradient[1]
        end

    end

    #=
    Now, to deal with this case :  [1 2 ; 3 4] .+ [2;2] = [3 4 ; 5 6]
    where the operation is broadcasted but no dimension is added, we'll need to sum the 
    broadcasted dims by keeping the dimensions.
    =#

    # For each dimension
    for i = 1:ndims(t)

        # If the dimension is equal to 1, it means that the operation is broadcasted along this axis
        # If it's a scalar, it doesn't change anything 
        if (size(t)[i] == 1)
            gradient = sum(gradient, dims=i)
        end
    end

    return gradient
end


"""
    Base.:+(t1::Tensor, t2::Tensor)
    Base.:+(t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}}
    Base.:+(notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}}

\\+ operator for tensors to support addition between 2 tensors
This method will add the 2 tensor's data , and then if one of the
two tensors requires gradient computation, the result of t1+t2 will also requires gradient computation.
Then, t1 and t2 is added in the list of dependencies of the resulting tensor, with the corresponding gradient functions.

- d(t1+t2)/d(t1) = 1 --> multiply the incoming gradient by 1.
- d(t1+t2)/d(t2) = 1, --> multiply the incoming gradient by 1.
"""
function Base.:+(t1::Tensor, t2::Tensor)

    data = t1.data + t2.data
    # initialise dependencies at nothing if none of the 2 tensors needs gradient computation
    dependencies = (t1.requires_grad || t2.requires_grad) ? TensorDependency[] : nothing

    if (t1.requires_grad)
        # Function used to compute the gradient of t1 :
        gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = incomingGradient
        # d(t1+t2)/d(t1) = 1, so we just need to multiply the incoming gradient by 1.
        push!(dependencies, TensorDependency(t1, gradientFunctionT1))
    end

    if (t2.requires_grad)
        # Function used to compute the gradient of t2 :
        gradientFunctionT2(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = incomingGradient
        # d(t1+t2)/d(t2) = 1, so we just need to multiply the incoming gradient by 1.
        push!(dependencies, TensorDependency(t2, gradientFunctionT2))
    end


    return Tensor(data, dependencies)
end

Base.:+(t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}} = t1 + Tensor(notATensor)
Base.:+(notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}} = Tensor(notATensor) + t1


"""
    Base.:broadcasted(::typeof(+), t1::Tensor, t2::Tensor)
    Base.:broadcasted(::typeof(+), t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}}
    Base.:broadcasted(::typeof(+), notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}}

Broadcast the + operator (perform element-wise addition).
This works in the same way as the Base.:+ operator, but the method handle_broadcasting! is called  
"""
function Base.:broadcasted(::typeof(+), t1::Tensor, t2::Tensor)

    data = t1.data .+ t2.data
    # iniialise dependencies at nothing if none of the 2 tensors needs gradient computation
    dependencies = (t1.requires_grad || t2.requires_grad) ? TensorDependency[] : nothing

    if (t1.requires_grad)
        # Function used to compute the gradient of t1 :
        gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = handle_broadcasting!(t1, incomingGradient)
        # d(t1+t2)/d(t1) = 1, so we just need to multiply the incoming gradient by 1. (also supports broadcasting)
        push!(dependencies, TensorDependency(t1, gradientFunctionT1))
    end

    if (t2.requires_grad)
        # Function used to compute the gradient of t2 :
        gradientFunctionT2(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = handle_broadcasting!(t2, incomingGradient)
        # d(t1+t2)/d(t2) = 1, so we just need to multiply the incoming gradient by 1. (also supports broadcasting)
        push!(dependencies, TensorDependency(t2, gradientFunctionT2))
    end

    return Tensor(data, dependencies)
end

Base.:broadcasted(::typeof(+), t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}} = t1 .+ Tensor(notATensor)
Base.:broadcasted(::typeof(+), notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}} = Tensor(notATensor) .+ t1

"""
    Base.:-(t1::Tensor, t2::Tensor)
    Base.:-(t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}}
    Base.:-(notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}}
    Base.:-(t2::Tensor)

\\- operator for tensors to support substraction between 2 tensors
This method will substract the 2 tensor's data, and then if one of the
two tensors requires gradient computation, the result of t1-t2 will also requires gradient computation.
Then, t1 and t2 is added in the list of dependencies of the resulting tensor, with the corresponding gradient functions.

- d(t1-t2)/d(t1) = 1 --> multiply the incoming gradient by 1.
- d(t1-t2)/d(t2) = -1, --> multiply the incoming gradient by -1.
- d(-t2)/d(t2) = -1 --> multiply the incoming gradient by -1.
"""
function Base.:-(t1::Tensor, t2::Tensor)

    data = t1.data - t2.data
    # iniialise dependencies at nothing if none of the 2 tensors needs gradient computation
    dependencies = (t1.requires_grad || t2.requires_grad) ? TensorDependency[] : nothing

    if (t1.requires_grad)
        # Function used to compute the gradient of t1 :
        gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = incomingGradient
        # d(t1-t2)/d(t1) = 1, so we just need to multiply the incoming gradient by 1.
        push!(dependencies, TensorDependency(t1, gradientFunctionT1))
    end

    if (t2.requires_grad)
        # Function used to compute the gradient of t2 :
        gradientFunctionT2(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = -incomingGradient
        # d(t1-t2)/d(t2) = -1, so we just need to multiply the incoming gradient by -1.
        push!(dependencies, TensorDependency(t2, gradientFunctionT2))
    end

    return Tensor(data, dependencies)
end

Base.:-(t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}} = t1 - Tensor(notATensor)
Base.:-(notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}} = Tensor(notATensor) - t1


# - operator for tensors to negate a tensor
function Base.:-(t1::Tensor)

    data = -t1.data

    if (t1.requires_grad)
        # Function used to compute the gradient of t1 :
        gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = -incomingGradient
        # d(-t1)/d(t1) = -1, so we just need to multiply the incoming gradient by 1.
        dependencies = [TensorDependency(t1, gradientFunctionT1)]
    else
        dependencies = nothing
    end

    return Tensor(data, dependencies)
end

"""
    Base.:broadcasted(::typeof(-), t1::Tensor, t2::Tensor)
    Base.:broadcasted(::typeof(-), t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}}
    Base.:broadcasted(::typeof(-), notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}}

Broadcast the - operator (perform element-wise substraction).
This works in the same way as the Base.:- operator, but the method handle_broadcasting! is called  
"""
function Base.:broadcasted(::typeof(-), t1::Tensor, t2::Tensor)

    data = t1.data .- t2.data
    # iniialise dependencies at nothing if none of the 2 tensors needs gradient computation
    dependencies = (t1.requires_grad || t2.requires_grad) ? TensorDependency[] : nothing

    if (t1.requires_grad)
        # Function used to compute the gradient of t1 :
        gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = handle_broadcasting!(t1, incomingGradient)
        # d(t1-t2)/d(t1) = 1, so we just need to multiply the incoming gradient by 1. (also supports broadcasting)
        push!(dependencies, TensorDependency(t1, gradientFunctionT1))
    end

    if (t2.requires_grad)
        # Function used to compute the gradient of t2 :
        gradientFunctionT2(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = handle_broadcasting!(t2, -incomingGradient)
        # d(t1-t2)/d(t2) = -1, so we just need to multiply the incoming gradient by -1. (also supports broadcasting)
        push!(dependencies, TensorDependency(t2, gradientFunctionT2))
    end

    return Tensor(data, dependencies)
end

Base.:broadcasted(::typeof(-), t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}} = t1 .- Tensor(notATensor)
Base.:broadcasted(::typeof(-), notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}} = Tensor(notATensor) .- t1

# * operator for tensors to support multiplication and matrix multiplication between 2 tensors
"""
    Base.:*(t1::Tensor, t2::Tensor)
    Base.:*(t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}}
    Base.:*(notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}}

\\* operator for tensors to support multiplication and matrix multiplication between 2 tensors
This method will multiply the 2 tensor's data , and then if one of the
two tensors requires gradient computation, the result of t1*t2 will also requires gradient computation.
Then, t1 and t2 is added in the list of dependencies of the resulting tensor, with the corresponding gradient functions.

With t1 = (n1,m1), t2 =(m1,m2) and t3 = t1 * t2 is (n1,m2) so the gradient coming from t3 is (n1,m2)
- d(t1*t2)/d(t1) = t2 --> multiply the incoming gradient transpose(t2.data)
- d(t1*t2)/d(t2) = t1, --> multiply transpose(t1) by the gradient

"""
function Base.:*(t1::Tensor, t2::Tensor)

    data = t1.data * t2.data
    # iniialise dependencies at nothing if none of the 2 tensors needs gradient computation
    dependencies = (t1.requires_grad || t2.requires_grad) ? TensorDependency[] : nothing

    if (t1.requires_grad)
        # Function used to compute the gradient of t1 :
        gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = incomingGradient * t2.data'
        #= With t1 (n1,m1), t2 (m1,m2) and t3 = t1 * t2 is (n1,m2)
        So the incoming gradient wrt t3 is (n1,m2)
        d(t1*t2)/d(t1) = t2
        So we just need to matmul the incoming gradient by t2. But t2 is (m1,m2)
        and the incoming gradient is (n1,m2). So we need to do grad * transpose(t2)
        This also works for scalars.
        =#
        push!(dependencies, TensorDependency(t1, gradientFunctionT1))
    end


    if (t2.requires_grad)
        # Function used to compute the gradient of t2 :
        gradientFunctionT2(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = t1.data' * incomingGradient
        #=
        d(t1*t2)/d(t2) = t1
        So we just need to matmul the incoming gradient by t1. But t1 is (n1,m1 )
        and the incoming gradient is (n1,m2). So we need to do transpose(t1) * grad
        =#
        push!(dependencies, TensorDependency(t2, gradientFunctionT2))
    end

    return Tensor(data, dependencies)
end

Base.:*(t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}} = t1 * Tensor(notATensor)
Base.:*(notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}} = Tensor(notATensor) * t1


"""
    Base.:broadcasted(::typeof(*), t1::Tensor, t2::Tensor)
    Base.:broadcasted(::typeof(*), t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}}
    Base.:broadcasted(::typeof(*), notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}}

Broadcast the * operator (perform element-wise multiplication).
This works in the same way as the Base.:* operator, but the method handle_broadcasting! is called  
"""
function Base.:broadcasted(::typeof(*), t1::Tensor, t2::Tensor)
    data = t1.data .* t2.data
    # iniialise dependencies at nothing if none of the 2 tensors needs gradient computation
    dependencies = (t1.requires_grad || t2.requires_grad) ? TensorDependency[] : nothing

    if (t1.requires_grad)
        # Function used to compute the gradient of t1 :
        gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = handle_broadcasting!(t1, t2.data .* incomingGradient)
        # d(t1*t2)/d(t1) = t2, so we just need to multiply the incoming gradient by t2. (also supports broadcasting)
        push!(dependencies, TensorDependency(t1, gradientFunctionT1))
    end

    if (t2.requires_grad)
        # Function used to compute the gradient of t2 :
        gradientFunctionT2(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = handle_broadcasting!(t2, t1.data .* incomingGradient)
        # d(t1*t2)/d(t2) = t1, so we just need to multiply the incoming gradient by t1. (also supports broadcasting)
        push!(dependencies, TensorDependency(t2, gradientFunctionT2))
    end

    return Tensor(data, dependencies)
end

Base.:broadcasted(::typeof(*), t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}} = t1 .* Tensor(notATensor)
Base.:broadcasted(::typeof(*), notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}} = Tensor(notATensor) .* t1


"""
    Base.:broadcasted(::typeof(/), t1::Tensor, t2::Tensor)
    Base.:broadcasted(::typeof(/), t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}}
    Base.:broadcasted(::typeof(/), notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}}

Broadcast the / operator (perform element-wise multiplication) between 2 tensors.
- d(t1/t2)/d(t1) = 1/t2 --> multiply the incoming gradient by 1/t2
- d(t1/t2)/d(t2) = -t1/t2^2, --> multiply the incoming gradient by -t1/t2^2
Then, the method handle_broadcasting! is called on the result of the gradient computation wrt to t1 and/or t2 
"""
function Base.:broadcasted(::typeof(/), t1::Tensor, t2::Tensor)

    data = t1.data ./ t2.data
    # iniialise dependencies at nothing if none of the 2 tensors needs gradient computation
    dependencies = (t1.requires_grad || t2.requires_grad) ? TensorDependency[] : nothing

    if (t1.requires_grad)
        # Function used to compute the gradient of t1 :
        gradientFunctionT1(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = handle_broadcasting!(t1, incomingGradient .* 1 ./ t2.data)
        #d(t1/t2)/d(t2) = 1/t2, so we just need to multiply the incoming gradient by 1/t2
        push!(dependencies, TensorDependency(t1, gradientFunctionT1))
    end

    if (t2.requires_grad)
        # Function used to compute the gradient of t2 :
        gradientFunctionT2(incomingGradient::T) where {T<:Union{AbstractArray,Float64,Int64}} = handle_broadcasting!(t2, (-t1.data ./ t2.data .^ 2) .* incomingGradient)
        #d(t1/t2)/d(t2) = -t1/t2^2, so we just need to multiply the incoming gradient by -t1/t2^2.
        push!(dependencies, TensorDependency(t2, gradientFunctionT2))
    end

    return Tensor(data, dependencies)
end

Base.:broadcasted(::typeof(/), t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}} = t1 ./ Tensor(notATensor)
Base.:broadcasted(::typeof(/), notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}} = Tensor(notATensor) ./ t1
