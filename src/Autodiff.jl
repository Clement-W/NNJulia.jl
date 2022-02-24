#module Autodiff

# To counter mutually recursive fields between TensorDependency and Tensor
abstract type AbstractTensor end



struct TensorDependency
    TensorDep::AbstractTensor
    GradFunction::Function
end


# To improve performances, it would be better to have an immutable struct...
# To improve performances, it would be better to use concrete types for AbstractArray ({Union{Float64,Int64}})
mutable struct Tensor{T<:Union{AbstractArray,Float64,Int64}}
    data::T
    gradient::T
    dependencies::Union{TensorDependency,Nothing}
end

function Tensor(data::T) where {T<:Union{AbstractArray,Float64,Int64}}
    Tensor{Union{AbstractArray,Float64,Int64}}(data, zero(data), nothing)
end


#TODO: implement other functions for abstract arrays : https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array
function Base.size(t::Tensor)
    return size(t.data)
end

#=
function Base.getindex(t::Tensor, i::Int)
    return getindex(t.data, i)
end

# doesn't work
function Base.getindex(t::Tensor, I::Vararg{Int,N}) where {N<:Integer}
    return getindex(t.data, I)
end=#

function zero_grad!(t::Tensor)
    t.gradient = zero(t.data)
end

function Base.show(io::IO, t::Tensor)
    println(io, "Tensor", size(t))
    print(io, "data : ")
    show(t.data)
    println()
    print(io, "grad : ")
    show(t.gradient)
end



#end
