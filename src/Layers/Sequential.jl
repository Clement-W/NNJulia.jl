using .Autodiff

struct Sequential{T<:AbstractLayer} <: AbstractModel
    #TODO: assert that the input and output size of the layers are compatible
    layers::Vector{T}
end

# Constructor accepting an arbitrary number of arguments (of abstract layer)
Sequential(layers::Vararg{T}) where {T<:AbstractLayer} = Sequential([layers...])

# THis constructor creates an empty list of layers
Sequential() = Sequential(AbstractLayer[])

# Make the Sequential struct callable to forward the input into the first layer, forwarding the result in the second layer, etc.
function (s::Sequential)(x::Union{Tensor,AbstractArray,Int64,Float64})
    for layer in s.layers
        x = layer(x)
    end
    return x
end

function Autodiff.zero_grad!(s::Sequential)
    for layer in s.layers
        Autodiff.zero_grad!(layer)
    end
end

function Base.show(io::IO, s::Sequential)
    println(io, "Sequential with : " * string(size(s.layers)[1]) * " layer")
    for layer in s.layers
        print(layer)
    end
end

# add a layer to the sequential layers
function add!(model::Sequential, layer::AbstractLayer)
    push!(model.layers, layer)
end

# return an array containing a reference to every parameters of the layers
function parameters(s::Sequential)
    p = Tensor[]
    for layer in s.layers
        append!(p, parameters(layer))
    end
    return p
end


