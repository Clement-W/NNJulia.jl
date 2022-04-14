using .Autodiff


"""
    Sequential(layers::Vector{T}) where {T<:AbstractLayer}
    Sequential(layers::Vararg{T}) where {T<:AbstractLayer}
    Sequential()  

This struct represents a Sequential, a list of sequential layers.

# Field
- layers: A list of layers
"""
struct Sequential{T<:AbstractLayer} <: AbstractModel
    #TODO: assert that the input and output size of the layers are compatible
    layers::Vector{T}
end

# Constructor accepting an arbitrary number of arguments (of abstract layer)
Sequential(layers::Vararg{AbstractLayer}) = Sequential([layers...])

# This constructor creates an empty list of layers
Sequential() = Sequential(AbstractLayer[])

"""
    (s::Sequential)(x::Union{Tensor,AbstractArray,Int64,Float64})
    
The Sequential struct is callable and forward the input into the first layer, forwarding the result in the next layer, etc. until the last layer.

"""
function (s::Sequential)(x::Union{Tensor,AbstractArray,Int64,Float64})
    for layer in s.layers
        x = layer(x)
    end
    return x
end


"""
    zero_grad!(s::Sequential)
    
Set the gradient of every tensors contained in the layers in the sequential to 0
"""
function Autodiff.zero_grad!(s::Sequential)
    for layer in s.layers
        Autodiff.zero_grad!(layer)
    end
end

"""
    Base.show(io::IO, s::Sequential)

String representation of a Sequentail
"""
function Base.show(io::IO, s::Sequential)
    println(io, "Sequential with : " * string(size(s.layers)[1]) * " layer")
    for layer in s.layers
        print(layer)
    end
end

# add a layer to the sequential layers
"""
    add!(model::Sequential, layer::AbstractLayer)

Add a layer into the list of layers of the sequential
"""
function add!(model::Sequential, layer::AbstractLayer)
    push!(model.layers, layer)
end

"""
    parameters(s::Sequential)

Return an array that contains a reference to every parameters of the layers
"""
function parameters(s::Sequential)
    p = Tensor[]
    for layer in s.layers
        append!(p, parameters(layer))
    end
    return p
end


