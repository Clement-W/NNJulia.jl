#Prototype d'une classe Dense :
# Ne sera probablement pas utilisée




# The struct is imutable but the values of the matrix W, and vector B can be changed 
struct Dense{W<:AbstractMatrix,B<:AbstractVector,F<:Function}
    weight::W
    bias::B
    activation::F
    function Dense(w::AbstractMatrix, b::AbstractVector, f::Function = identity)
        if (size(w)[1] != size(b)[1])
            throw(ErrorException("Weights and Biases dimensions are not compatible : W=" * string(size(w)) * ", B=" * string(size(b)) * " and " * string(size(w)[1]) * "!=" * string(size(b)[1])))
        end
        new{AbstractMatrix,AbstractVector,Function}(w, b, f)
    end
end

# Alternative constructor that initialise randomly the parameters
function Dense(in::Int64, out::Int64, activ::Function = identity)
    # Initialise weights and bias with random Float64 between -1 and 1
    w = rand(out, in) * 2 .- 1
    b = rand(out) * 2 .- 1
    f = activ
    Dense{AbstractMatrix,AbstractVector,Function}(w, b, f)
end


# Make the Dense struct callable to compute f(W*x+b)
(a::Dense)(x::AbstractVecOrMat) = a.activation.(a.weight * x .+ a.bias)
# the dot after activation is important to apply the function element-wise


function Base.show(io::IO, l::Dense)
    print(io, "Dense(", size(l.weight, 2), ", ", size(l.weight, 1))
    print(io, ", ", l.activation)
    print(io, ")")
end

function parameters(l::Dense)
    return [l.weight, l.bias]
end


#tests
using Zygote
a = Dense(2, 5)
x = [2; 1]
loss() = sum(a(x))
grads = gradient(loss, Params([a.weight, a.bias]))

#println(grads[a.weight])
#println(grads[a.bias])

y, da = pullback(a, x)
#println(da([1; 1; 1; 1; 1])) # incoming gradients from the 5 output of the dense layer
# this output the gradientxs at the 2 intput

a.weight .-= grads[a.weight]

# superclasse module pour tous les trucs differentiable, avec une méthode parameters qui retourne les
# les parameters et qui doit être redéfinie