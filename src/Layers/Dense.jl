using .Autodiff

# The struct is imutable but the values of the matrix weight, and vector bias can be changed 
struct Dense{T<:Tensor,F<:Function} <: AbstractLayer
    weight::T
    bias::T
    activation::F
    function Dense(w::Tensor, b::Tensor, f::Function = identity)
        if (size(w) != size(b) && size(w) != ())
            if (size(w)[1] != size(b)[1])
                throw(ErrorException("Weights and Biases dimensions are not compatible : W=" * string(size(w)) * ", B=" * string(size(b)) * " and " * string(size(w)[1]) * "!=" * string(size(b)[1])))
            end
        end

        if (!w.requires_grad || !b.requires_grad)
            throw(ErrorException("The weights and biases tensors must have requires_grad at true to compute the gradients"))
        end
        new{Tensor,Function}(w, b, f)
    end
end

# alternative contrsuctor
function Dense(in::Int64, out::Int64, activ::Function = identity)
    # Initialise weights and bias with random Float64 between -1 and 1
    w = Tensor(rand(out, in) * 2 .- 1, true)
    b = Tensor(rand(out) * 2 .- 1, true)
    f = activ
    Dense(w, b, f)
end

function Base.show(io::IO, d::Dense)
    println(io, "Dense: ", string(size(d.weight)[2]), " --> " * string(size(d.weight)[1]))
    print(io, "weight: " * string(size(d.weight)))
    println()
    print(io, "bias: " * string(size(d.bias)))
    println()
    print(io, "activation function: " * string(d.activation))
    println()
end

function Autodiff.zero_grad!(d::Dense)
    Autodiff.zero_grad!(d.weight)
    Autodiff.zero_grad!(d.bias)
end

# Make the Dense struct callable to compute f(W*x+b)
(a::Dense)(x::Union{Tensor,AbstractArray,Int64,Float64}) = a.activation(a.weight * x .+ a.bias)
# the activation function needs to be be applied element-wise
# for exemple : (x)-> x .+ 2


