module NNJulia

include("Autodiff/Autodiff.jl")
export AbstractTensor, Tensor, TensorDependency, zero_grad!, backward!, handle_broadcasting!, sigmoid, relu, leakyrelu, softmax
using .Autodiff

include("Layers/Layers.jl")
export AbstractLayer, AbstractModel, Dense, Sequential, zero_grad!, add!, parameters, Conv2D, Flatten
using .Layers

include("Optimisers/Optimisers.jl")
export AbstractOptimiser, GradientDescent, update!
using .Optimisers

include("Loss/Loss.jl")
export AbstractLoss, MSE, BinaryCrossentropy, CategoricalCrossentropy, compute_loss
using .Loss

include("Metrics/Metrics.jl")
export AbstractMetrics, BinaryAccuracy, CategoricalAccuracy, Accuracy, compute_accuracy
using .Metrics

include("DataLoader.jl")
export DataLoader

include("Model.jl")
export TrainParameters, train!, predict, evaluate

include("Utils.jl")
export split_train_test, plot_decision_boundary, to_one_hot



end