module NNJulia

include("Autodiff/Autodiff.jl")
export AbstractTensor, Tensor, TensorDependency, zero_grad!, backward!, handle_broadcasting!, sigmoid, relu, leakyrelu
using .Autodiff

include("Layers/Layers.jl")
export AbstractLayer, AbstractModel, Dense, Sequential, zero_grad!, add!, parameters, Convolution
using .Layers

include("Optimisers/Optimisers.jl")
export AbstractOptimiser, GradientDescent, update!
using .Optimisers

include("Loss/Loss.jl")
export AbstractLoss, MSE, BinaryCrossentropy
using .Loss

include("Metrics/Metrics.jl")
export AbstractMetrics, BinaryAccuracy, CategoricalAccuracy, compute_accuracy
using .Metrics

include("DataLoader.jl")
export DataLoader

include("Model.jl")
export TrainParameters, train!, predict, evaluate

include("Utils.jl")
export split_train_test, plot_decision_boundary


end