module NNJulia

include("Autodiff/Autodiff.jl")
export AbstractTensor, Tensor, TensorDependency, zero_grad!, backward!
using .Autodiff

include("Layers/Layers.jl")
export AbstractLayer, AbstractModel, Dense, Sequential, zero_grad!, add!, parameters, Convolution
using .Layers

include("Optimisers/Optimisers.jl")
export AbstractOptimiser, GradientDescent, update!
using .Optimisers


end