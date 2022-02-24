# To counter mutually recursive fields between TensorDependency and Tensor
abstract type AbstractTensor end

struct TensorDependency
    TensorDep::AbstractTensor
    GradFunction::Function
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
function Tensor(data::T) where {T<:Union{AbstractArray,Float64,Int64}}
    Tensor(data, zero(data), nothing)
end

function Tensor(data::T, gradient::T) where {T<:Union{AbstractArray,Float64,Int64}}
    Tensor(data, gradient, nothing)
end

# customize the set property for t.data
function Base.setproperty!(t::Tensor, prop::Symbol, val)
    if (prop == :data)
        # Invalidate the gradient if the data is set mannualy
        t.gradient = zero(t.data)
        setfield!(t, :data, val)
    else
        setfield!(t, prop, val)
    end
end

# return the size of the data
function Base.size(t::Tensor)
    return size(t.data)
end

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

