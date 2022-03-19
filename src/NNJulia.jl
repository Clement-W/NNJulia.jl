module NNJulia

include("Autodiff/Autodiff.jl")
export AbstractTensor, Tensor, TensorDependency, zero_grad!, backward!
using .Autodiff

include("Layers/Layers.jl")
export Dense, Sequential, zero_grad!, add!, parameters, Convolution
using .Layers

end