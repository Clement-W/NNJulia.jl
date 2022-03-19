using .Autodiff


# TODO:
struct Convolution{T<:Tensor,F<:Function} <: AbstractLayer
    filters::Vector{T} # list of convolution filters
    bias::T # list of biases associated to each convolution filters
    activation::F



end