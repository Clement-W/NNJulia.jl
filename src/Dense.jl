
# The struct is imutable but the values of the matrix W, and vector B can be changed 
struct Dense{W<:AbstractMatrix,B<:AbstractVector,F<:Function}
    weight::W
    bias::B
    activation::F
    function Dense(in::Int64, out::Int64, activ::Function = identity)
        # Initialise weights and bias with random Float64 between -1 and 1
        w = rand(out, in) * 2 .- 1
        b = rand(out) * 2 .- 1
        f = activ
        new{AbstractMatrix,AbstractVector,Function}(w, b, f)
    end
end

# Make the Dense struct callable to compute f(W*x+b)
(a::Dense)(x::AbstractVecOrMat) = a.activation(a.weight * x .+ a.bias)


function Base.show(io::IO, l::Dense)
    print(io, "Dense(", size(l.weight, 2), ", ", size(l.weight, 1))
    print(io, ", ", l.activation)
    print(io, ")")
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
println(da([1; 1; 1; 1; 1])) # incoming gradients from the 5 output of the dense layer
# this output the gradients at the 2 intput