using .Autodiff


"""
    Dense(w::Tensor, b::Tensor, f::Function=identity)
    Dense(in::Int64, out::Int64, activ::Function=identity)

This struct represents a Dense layer (fully connected neurons). 
To initialise a Dense layer, the simplest way is to use the second constructor by giving
the input and output size of the layer.

# Fields 
- weight: A tensor that contains the weight of the neurones of the layer
- bias: A tensor that contains the biases of the neurons of the layer
- activation: The activation function of the layer
"""
struct Dense{T<:Tensor,F<:Function} <: AbstractLayer
    # The struct is imutable but the values of the matrix weight, and vector bias can be changed 
    weight::T
    bias::T
    activation::F
    function Dense(w::Tensor, b::Tensor, f::Function=identity)
        # Check that the size of the weight and biases are compatible
        if (size(w) != size(b) && size(w) != ())
            if (size(w)[1] != size(b)[1])
                throw(ErrorException("Weights and Biases dimensions are not compatible : W=" * string(size(w)) * ", B=" * string(size(b)) * " and " * string(size(w)[1]) * "!=" * string(size(b)[1])))
            end
        end

        # Check that the parameters requires gradient computation
        if (!w.requires_grad || !b.requires_grad)
            throw(ErrorException("The weights and biases tensors must have requires_grad at true to compute the gradients"))
        end
        new{Tensor,Function}(w, b, f)
    end
end

# alternative contrsuctor
function Dense(in::Int64, out::Int64, activ::Function=identity)
    # Initialise weights and bias with random Float64 between -1 and 1
    w = Tensor(rand(out, in) * 2 .- 1, true)
    b = Tensor(rand(out) * 2 .- 1, true)
    f = activ
    Dense(w, b, f)
end

"""
    Base.show(io::IO, d::Dense)

String representation of a Dense layer
"""
function Base.show(io::IO, d::Dense)
    if (ndims(d.weight) != 0)
        print(io, "Dense: ", string(size(d.weight)[2]), " --> " * string(size(d.weight)[1]))
    else
        print(io, "Dense: 1 --> 1 ")
    end
    println(", Activation = " * string(d.activation))
end

"""
    zero_grad!(d::Dense)

Set the gradient with respect to the weights and biases of this layer to 0
"""
function Autodiff.zero_grad!(d::Dense)
    Autodiff.zero_grad!(d.weight)
    Autodiff.zero_grad!(d.bias)
end

"""
    (a::Dense)(x::Union{Tensor,AbstractArray,Int64,Float64})

The dense struct is callable, and compute f(W*x + b) with f the activation function
"""
(a::Dense)(x::Union{Tensor,AbstractArray,Int64,Float64}) = a.activation(a.weight * x .+ a.bias)



"""
    parameters(d::Dense) 

Return every trainable tensors of a dense layer (weight and biases)
"""
parameters(d::Dense) = Tensor[d.weight, d.bias]

