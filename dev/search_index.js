var documenterSearchIndex = {"docs":
[{"location":"loss/#Loss","page":"Loss","title":"Loss","text":"","category":"section"},{"location":"loss/","page":"Loss","title":"Loss","text":"CurrentModule = NNJulia.Loss","category":"page"},{"location":"loss/#Loss-functions","page":"Loss","title":"Loss functions","text":"","category":"section"},{"location":"loss/","page":"Loss","title":"Loss","text":"AbstractLoss\nMSE\nBinaryCrossentropy\nCategoricalCrossentropy\ncompute_loss","category":"page"},{"location":"loss/#NNJulia.Loss.AbstractLoss","page":"Loss","title":"NNJulia.Loss.AbstractLoss","text":"AbstractLoss\n\nEvery loss struct is a subtype of AbstractLoss\n\n\n\n\n\n","category":"type"},{"location":"loss/#NNJulia.Loss.MSE","page":"Loss","title":"NNJulia.Loss.MSE","text":"MSE\n\nRepresents the Mean Squared Error : L(y,y^i)=∑(y-y^i)^2\n\n\n\n\n\n","category":"type"},{"location":"loss/#NNJulia.Loss.BinaryCrossentropy","page":"Loss","title":"NNJulia.Loss.BinaryCrossentropy","text":"BinaryCrossentropy\n\nRepresents the Binary crossentropy error function\n\n\n\n\n\n","category":"type"},{"location":"loss/#NNJulia.Loss.CategoricalCrossentropy","page":"Loss","title":"NNJulia.Loss.CategoricalCrossentropy","text":"CategoricalCrossentropy\n\nRepresents the Categorical crossentropy error function\n\n\n\n\n\n","category":"type"},{"location":"loss/#NNJulia.Loss.compute_loss","page":"Loss","title":"NNJulia.Loss.compute_loss","text":"compute_loss(lossF::MSE, predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})\ncompute_loss(lossF::BinaryCrossentropy, predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})\ncompute_loss(lossF::CategoricalCrossentropy, predicted::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})\n\nCompute the loss according to the lossF given.\n\n\n\n\n\n","category":"function"},{"location":"metrics/#Metrics","page":"Metrics","title":"Metrics","text":"","category":"section"},{"location":"metrics/","page":"Metrics","title":"Metrics","text":"CurrentModule = NNJulia.Metrics","category":"page"},{"location":"metrics/","page":"Metrics","title":"Metrics","text":"AbstractMetrics\nAccuracy\nBinaryAccuracy\nCategoricalAccuracy\ncompute_accuracy","category":"page"},{"location":"metrics/#NNJulia.Metrics.AbstractMetrics","page":"Metrics","title":"NNJulia.Metrics.AbstractMetrics","text":"AbstractMetrics\n\nEvery metrics struct is a subtype of AbstractMetrics\n\n\n\n\n\n","category":"type"},{"location":"metrics/#NNJulia.Metrics.Accuracy","page":"Metrics","title":"NNJulia.Metrics.Accuracy","text":"Accuracy\n\nRepresents the classic accuracy metric\n\n\n\n\n\n","category":"type"},{"location":"metrics/#NNJulia.Metrics.BinaryAccuracy","page":"Metrics","title":"NNJulia.Metrics.BinaryAccuracy","text":"BinaryCrossentropy\n\nRepresents the binary accuracy metric\n\nField\n\nthreshold: The threshold used to decide if the output is 0 or 1. Every predictions > threshold is set to 1\n\n\n\n\n\n","category":"type"},{"location":"metrics/#NNJulia.Metrics.CategoricalAccuracy","page":"Metrics","title":"NNJulia.Metrics.CategoricalAccuracy","text":"CategoricalAccuracy\n\nRepresents the categorical accuracy metric\n\n\n\n\n\n","category":"type"},{"location":"metrics/#NNJulia.Metrics.compute_accuracy","page":"Metrics","title":"NNJulia.Metrics.compute_accuracy","text":"compute_accuracy(metrics::BinaryAccuracy, predictions::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})\ncompute_accuracy(metrics::CategoricalAccuracy, predictions::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})\ncompute_accuracy(metrics::Accuracy, predictions::Tensor, target::Union{Tensor,AbstractArray,Float64,Int64})\n\nCompute the accuracy according to the metrics given.\n\n\n\n\n\n","category":"function"},{"location":"dataloader/#DataLoader","page":"DataLoader","title":"DataLoader","text":"","category":"section"},{"location":"dataloader/","page":"DataLoader","title":"DataLoader","text":"See Base.iterate in General Methods","category":"page"},{"location":"dataloader/","page":"DataLoader","title":"DataLoader","text":"DataLoader","category":"page"},{"location":"dataloader/#NNJulia.DataLoader","page":"DataLoader","title":"NNJulia.DataLoader","text":"DataLoader(XData::T1, YData::T2, batchSize=1::Int, shuffle=false::Bool) where {T1<:Union{AbstractArray,Float64,Int64},T2<:Union{AbstractArray,Float64,Int64}}\n\nThis struct represent a DataLoader. A DataLoader is used to load a dataset with inputs and label, shuffle the data, separate it into batches, etc. It is possible to iterate over this struct to access the batches of the dataset.\n\nFor the first iteration, the list of indices is shuffled (if required). Then, it return a subpart of the dataset according to the batch size with this size : (dataSize...,batchSize)\n\nFor example, if the input data contains two values for a batch size of 4, this matrix will be returned :\n\n[\n    1 0 1 1\n    0 1 0 1\n]\n\n\n\n\n\n","category":"type"},{"location":"general/#General-methods","page":"General methods","title":"General methods","text":"","category":"section"},{"location":"general/#Base-methods-dispatched","page":"General methods","title":"Base methods dispatched","text":"","category":"section"},{"location":"general/","page":"General methods","title":"General methods","text":"Base.setproperty!\nBase.size\nBase.ndims\nBase.length\nBase.iterate\nBase.show","category":"page"},{"location":"general/#Base.setproperty!","page":"General methods","title":"Base.setproperty!","text":"Base.setproperty!(t::Tensor, prop::Symbol, val)\n\nIf the data property is modified, the gradient is set to 0\n\n\n\n\n\n","category":"function"},{"location":"general/#Base.size","page":"General methods","title":"Base.size","text":"Base.size(t::Tensor)\n\nReturn the size of the tensor's data\n\n\n\n\n\n","category":"function"},{"location":"general/#Base.ndims","page":"General methods","title":"Base.ndims","text":"Base.ndims(t::Tensor{Union{Float64,Int64,AbstractArray}})\n\nReturn the number of dimensions of the tensor's data\n\n\n\n\n\n","category":"function"},{"location":"general/#Base.length","page":"General methods","title":"Base.length","text":"Base.length(t::Tensor{Union{Float64,Int64,AbstractArray}})\n\nReturn the length of the tensor's data\n\n\n\n\n\nBase.length(d::DataLoader)\n\nThe length of a DataLoader is it's number of batches.\n\n\n\n\n\n","category":"function"},{"location":"general/#Base.iterate","page":"General methods","title":"Base.iterate","text":"Base.iterate(t::Tensor{Union{Float64,Int64,AbstractArray}})\nBase.iterate(t::Tensor{Union{Float64,Int64,AbstractArray}}, state)\n\nIterate on the tensor's data\n\n\n\n\n\nBase.iterate(d::DataLoader, state=1)\n\nIterate over the DataLoader.  For the first iteration, the list of indices is shuffled (if required). Then, return a subpart of the dataset according to the batch size with this size : (dataSize...,batchSize)\n\nFor example, if the input data contains two values for a batch size of 4, this matrix will be returned :\n\n[\n    1 0 1 1\n    0 1 0 1\n]\n\n\n\n\n\n","category":"function"},{"location":"general/#Base.show","page":"General methods","title":"Base.show","text":"Base.show(io::IO, d::Dense)\n\nString representation of a Dense layer\n\n\n\n\n\nBase.show(io::IO, s::Sequential)\n\nString representation of a Sequentail\n\n\n\n\n\nBase.show(io::IO, d::Flatten)\n\nString representation of a Flatten layer\n\n\n\n\n\nBase.show(io::IO, t::Tensor)\n\nString representation of a tensor\n\n\n\n\n\n","category":"function"},{"location":"general/#Methods-for-the-gradient","page":"General methods","title":"Methods for the gradient","text":"","category":"section"},{"location":"general/","page":"General methods","title":"General methods","text":"zero_grad!","category":"page"},{"location":"general/#NNJulia.Autodiff.zero_grad!","page":"General methods","title":"NNJulia.Autodiff.zero_grad!","text":"zero_grad!(d::Dense)\n\nSet the gradient with respect to the weights and biases of this layer to 0\n\n\n\n\n\nzero_grad!(s::Sequential)\n\nSet the gradient of every tensors contained in the layers in the sequential to 0\n\n\n\n\n\nzero_grad!(d::Flatten)\n\nDoes nothing for a flatten layer\n\n\n\n\n\nzero_grad!(t::Tensor)\n\nSet the gradient with respect to this tensor to 0\n\n\n\n\n\n","category":"function"},{"location":"layers/#Layers","page":"Layers","title":"Layers","text":"","category":"section"},{"location":"layers/","page":"Layers","title":"Layers","text":"CurrentModule = NNJulia.Layers","category":"page"},{"location":"layers/#Structs-and-types","page":"Layers","title":"Structs and types","text":"","category":"section"},{"location":"layers/","page":"Layers","title":"Layers","text":"AbstractLayer\nAbstractModel\nDense\nSequential\nFlatten","category":"page"},{"location":"layers/#NNJulia.Layers.AbstractLayer","page":"Layers","title":"NNJulia.Layers.AbstractLayer","text":"AbstractLayer\n\nEvery layer is a subtype of this abstract type.\n\n\n\n\n\n","category":"type"},{"location":"layers/#NNJulia.Layers.AbstractModel","page":"Layers","title":"NNJulia.Layers.AbstractModel","text":"AbstractModel\n\nEvery trainable model is a subtype of this abstract type.\n\n\n\n\n\n","category":"type"},{"location":"layers/#NNJulia.Layers.Dense","page":"Layers","title":"NNJulia.Layers.Dense","text":"Dense(w::Tensor, b::Tensor, f::Function=identity)\nDense(in::Int64, out::Int64, activ::Function=identity)\n\nThis struct represents a Dense layer (fully connected neurons).  To initialise a Dense layer, the simplest way is to use the second constructor by giving the input and output size of the layer.\n\nFields\n\nweight: A tensor that contains the weight of the neurones of the layer\nbias: A tensor that contains the biases of the neurons of the layer\nactivation: The activation function of the layer\n\n\n\n\n\n","category":"type"},{"location":"layers/#NNJulia.Layers.Sequential","page":"Layers","title":"NNJulia.Layers.Sequential","text":"Sequential(layers::Vector{T}) where {T<:AbstractLayer}\nSequential(layers::Vararg{T}) where {T<:AbstractLayer}\nSequential()\n\nThis struct represents a Sequential, a list of sequential layers.\n\nField\n\nlayers: A list of layers\n\n\n\n\n\n","category":"type"},{"location":"layers/#NNJulia.Layers.Flatten","page":"Layers","title":"NNJulia.Layers.Flatten","text":"Flatten()\n\nThis struct represents a Flatten layer  It flatten the input data into a 2 dimentional tensor of shape (n,batchSize) with n = the multiplication of every other dimension's size (except the batchsize) of the original tensor This operation preserve the size of the last dimension.\n\n\n\n\n\n","category":"type"},{"location":"layers/#Methods-for-layers","page":"Layers","title":"Methods for layers","text":"","category":"section"},{"location":"layers/","page":"Layers","title":"Layers","text":"parameters\nadd!","category":"page"},{"location":"layers/#NNJulia.Layers.parameters","page":"Layers","title":"NNJulia.Layers.parameters","text":"parameters(d::Dense)\n\nReturn every trainable tensors of a dense layer (weight and biases)\n\n\n\n\n\nparameters(s::Sequential)\n\nReturn an array that contains a reference to every parameters of the layers\n\n\n\n\n\nparameters(d::Flatten)\n\nDoes nothing for a flatten layer\n\n\n\n\n\n","category":"function"},{"location":"layers/#NNJulia.Layers.add!","page":"Layers","title":"NNJulia.Layers.add!","text":"add!(model::Sequential, layer::AbstractLayer)\n\nAdd a layer into the list of layers of the sequential\n\n\n\n\n\n","category":"function"},{"location":"model/#Model","page":"Model","title":"Model","text":"","category":"section"},{"location":"model/","page":"Model","title":"Model","text":"TrainParameters\ntrain!\nevaluate","category":"page"},{"location":"model/#NNJulia.TrainParameters","page":"Model","title":"NNJulia.TrainParameters","text":"TrainParameters(opt::AbstractOptimiser, lossFunction::AbstractLoss, metrics::AbstractMetrics)\n\nThis struct store the important parameters used to train the model.\n\nFields\n\nopt: The optimiser used to optimise the loss\nlossFunction: The function used to compute the loss\nmetrics: The metrics used to compute the accuracy of the model\n\n\n\n\n\n","category":"type"},{"location":"model/#NNJulia.train!","page":"Model","title":"NNJulia.train!","text":"train!(model::AbstractModel, trainParams::TrainParameters, trainData::DataLoader, nbEpochs::Int, verbose::Bool=true)\n\nThis method train a model on the trainData. The accuracy and the loss computed at each epoch is stored into a dictionnary that is returned at the end of the training.\n\nThe dictionnary returned looks like this :  history = Dict(\"accuracy\" => Float64[], \"loss\" => Float64[])\n\n\n\n\n\n","category":"function"},{"location":"model/#NNJulia.evaluate","page":"Model","title":"NNJulia.evaluate","text":"evaluate(model::AbstractModel, metrics::BinaryAccuracy, xData::Union{Tensor,AbstractArray,Float64,Int64}, yData::Union{Tensor,AbstractArray,Float64,Int64})\n\nThis method evaluate a model by returning the accuracy computed with the given metrics\n\n\n\n\n\n","category":"function"},{"location":"optimisers/#Optimisers","page":"Optimisers","title":"Optimisers","text":"","category":"section"},{"location":"optimisers/","page":"Optimisers","title":"Optimisers","text":"CurrentModule = NNJulia.Optimisers","category":"page"},{"location":"optimisers/","page":"Optimisers","title":"Optimisers","text":"AbstractOptimiser\nGradientDescent\nupdate!","category":"page"},{"location":"optimisers/#NNJulia.Optimisers.AbstractOptimiser","page":"Optimisers","title":"NNJulia.Optimisers.AbstractOptimiser","text":"AbstractOptimiser\n\nEvery optimiser struct is a subtype of AbstractOptimiser\n\n\n\n\n\n","category":"type"},{"location":"optimisers/#NNJulia.Optimisers.GradientDescent","page":"Optimisers","title":"NNJulia.Optimisers.GradientDescent","text":"GradientDescent(lr::Float64)\nGradientDescent()\n\nRepresents the vanilla gradient descent optimiser. The default constructor initialise the learning rate at 0.1\n\nField\n\nlr: The learning rate\n\n\n\n\n\n","category":"type"},{"location":"optimisers/#NNJulia.Optimisers.update!","page":"Optimisers","title":"NNJulia.Optimisers.update!","text":"update!(opt::GradientDescent, model::AbstractModel)\n\nUpdate the parameters of the model using the given optimiser.\n\n\n\n\n\n","category":"function"},{"location":"#NNJulia-Documentation","page":"NNJulia Documentation","title":"NNJulia Documentation","text":"","category":"section"},{"location":"","page":"NNJulia Documentation","title":"NNJulia Documentation","text":"Minimalist neural network library made in Julia for educational purposes","category":"page"},{"location":"","page":"NNJulia Documentation","title":"NNJulia Documentation","text":"Pages = [\"general.md\"]\nDepth = 1","category":"page"},{"location":"","page":"NNJulia Documentation","title":"NNJulia Documentation","text":"Pages = [\"autodiff.md\"]\nDepth = 1","category":"page"},{"location":"","page":"NNJulia Documentation","title":"NNJulia Documentation","text":"Pages = [\"metrics.md\"]\nDepth = 1","category":"page"},{"location":"","page":"NNJulia Documentation","title":"NNJulia Documentation","text":"Pages = [\"loss.md\"]\nDepth = 1","category":"page"},{"location":"","page":"NNJulia Documentation","title":"NNJulia Documentation","text":"Pages = [\"metrics.md\"]\nDepth = 1","category":"page"},{"location":"","page":"NNJulia Documentation","title":"NNJulia Documentation","text":"Pages = [\"optimisers.md\"]\nDepth = 1","category":"page"},{"location":"","page":"NNJulia Documentation","title":"NNJulia Documentation","text":"Pages = [\"dataloader.md\"]\nDepth = 1","category":"page"},{"location":"","page":"NNJulia Documentation","title":"NNJulia Documentation","text":"Pages = [\"utils.md\"]\nDepth = 1","category":"page"},{"location":"","page":"NNJulia Documentation","title":"NNJulia Documentation","text":"Pages = [\"model.md\"]\nDepth = 1","category":"page"},{"location":"autodiff/#Autodiff","page":"Autodiff","title":"Autodiff","text":"","category":"section"},{"location":"autodiff/","page":"Autodiff","title":"Autodiff","text":"CurrentModule = NNJulia.Autodiff","category":"page"},{"location":"autodiff/#Structs-and-types","page":"Autodiff","title":"Structs and types","text":"","category":"section"},{"location":"autodiff/","page":"Autodiff","title":"Autodiff","text":"AbstractTensor\nTensorDependency\nTensor\n","category":"page"},{"location":"autodiff/#NNJulia.Autodiff.AbstractTensor","page":"Autodiff","title":"NNJulia.Autodiff.AbstractTensor","text":"AbstractTensor\n\nThis type is used to counter the circular dependency between TensorDependency and Tensor.\n\n\n\n\n\n","category":"type"},{"location":"autodiff/#NNJulia.Autodiff.TensorDependency","page":"Autodiff","title":"NNJulia.Autodiff.TensorDependency","text":"TensorDependency(tensorDep::AbstractTensor, gradFunction::Function)\n\nThis struct represents the dependence of a tensor. This is used to keep track of the tensor's dependencies. For example, if a tensor is made up by the sum of 2 other tensor, this tensor will have 2 TensorDependency object in it's list of dependency. This struct also stores the derivative of the operation linking the dependencies, to be able to compute the gradient of the resul tensor, with respect to the  dependencies.\n\nFields\n\ntensorDep: The tensor dependence\ngradFunction: This function is used to compute the gradient of the tensor that depends on TensorDep, with respect to the dependencies.\n\n\n\n\n\n","category":"type"},{"location":"autodiff/#NNJulia.Autodiff.Tensor","page":"Autodiff","title":"NNJulia.Autodiff.Tensor","text":"Tensor(data::T, gradient::Union{T,Nothing}, dependencies::Union{Vector{TensorDependency},Nothing}, requires_grad::Bool) where {T<:Union{AbstractArray,Float64,Int64}}\nTensor(data::T, requires_grad::Bool=false) where {T<:Union{AbstractArray,Float64,Int64}}\nTensor(data::T, gradient::Union{T,Nothing}) where {T<:Union{AbstractArray,Float64,Int64}}\nTensor(data::T, dependencies::Union{Vector{TensorDependency},Nothing}) where {T<:Union{AbstractArray,Float64,Int64}}\n\nThis mutable struct represents a Tensor, it is a scalar or an array that supports gradient computation\n\nFields\n\ndata: The data contained in the tensor as a scalar or an array\ngradient: A gradient with respect to this tensor\ndependencies: A list that contains the tensors on which the current tensor depends\nrequires_grad: Boolean which indicates if the gradient has to be computed for this tensor\n\n\n\n\n\n","category":"type"},{"location":"autodiff/#Methods-for-the-gradient","page":"Autodiff","title":"Methods for the gradient","text":"","category":"section"},{"location":"autodiff/","page":"Autodiff","title":"Autodiff","text":"backward!\nhandle_broadcasting!","category":"page"},{"location":"autodiff/#NNJulia.Autodiff.backward!","page":"Autodiff","title":"NNJulia.Autodiff.backward!","text":"backward!(t::Tensor, incomingGradient::Union{T,Nothing}=nothing) where {T<:Union{AbstractArray,Float64,Int64}}\n\nBackpropagate a gradient through the auto differenciation graph by recurcively calling this method on the tensor dependencies. The gradient don't need to be specified if the current tensor is a scalar \n\n\n\n\n\n","category":"function"},{"location":"autodiff/#NNJulia.Autodiff.handle_broadcasting!","page":"Autodiff","title":"NNJulia.Autodiff.handle_broadcasting!","text":"handle_broadcasting!(t::Tensor, gradient::T) where {T<:Union{AbstractArray,Float64,Int64}}\n\nUsed to support gradient computation with broadcast operations made with broadcasted operators \n\nFirst, sum out the dims added by the broadcast operation, so that the gradient has the same dimensions of the tensor. To compute the gradient when a dimension is added by the broadcast operation,  the gradient is summed along the batch axis (the dimension added). This will handle this example : [1 2 ; 3 4] .+ [2,2] = [3 4; 5 6]\n\nThen, when the operation is broadcasted but no dimension is added, the  broadcasted dims are summed by keeping the dimensions. This will handle this example : [1 2 ; 3 4] .+ [2;2] = [3 4 ; 5 6]\n\n\n\n\n\n","category":"function"},{"location":"autodiff/#Operators-between-tensors","page":"Autodiff","title":"Operators between tensors","text":"","category":"section"},{"location":"autodiff/","page":"Autodiff","title":"Autodiff","text":"Base.:+\nBase.:-\nBase.:*\nBase.:broadcasted","category":"page"},{"location":"autodiff/#Base.:+","page":"Autodiff","title":"Base.:+","text":"Base.:+(t1::Tensor, t2::Tensor)\nBase.:+(t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}}\nBase.:+(notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}}\n\n+ operator for tensors to support addition between 2 tensors This method will add the 2 tensor's data , and then if one of the two tensors requires gradient computation, the result of t1+t2 will also requires gradient computation. Then, t1 and t2 is added in the list of dependencies of the resulting tensor, with the corresponding gradient functions.\n\nd(t1+t2)/d(t1) = 1 –> multiply the incoming gradient by 1.\nd(t1+t2)/d(t2) = 1, –> multiply the incoming gradient by 1.\n\n\n\n\n\n","category":"function"},{"location":"autodiff/#Base.:-","page":"Autodiff","title":"Base.:-","text":"Base.:-(t1::Tensor, t2::Tensor)\nBase.:-(t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}}\nBase.:-(notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}}\nBase.:-(t2::Tensor)\n\n- operator for tensors to support substraction between 2 tensors This method will substract the 2 tensor's data, and then if one of the two tensors requires gradient computation, the result of t1-t2 will also requires gradient computation. Then, t1 and t2 is added in the list of dependencies of the resulting tensor, with the corresponding gradient functions.\n\nd(t1-t2)/d(t1) = 1 –> multiply the incoming gradient by 1.\nd(t1-t2)/d(t2) = -1, –> multiply the incoming gradient by -1.\nd(-t2)/d(t2) = -1 –> multiply the incoming gradient by -1.\n\n\n\n\n\n","category":"function"},{"location":"autodiff/#Base.:*","page":"Autodiff","title":"Base.:*","text":"Base.:*(t1::Tensor, t2::Tensor)\nBase.:*(t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}}\nBase.:*(notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}}\n\n* operator for tensors to support multiplication and matrix multiplication between 2 tensors This method will multiply the 2 tensor's data , and then if one of the two tensors requires gradient computation, the result of t1*t2 will also requires gradient computation. Then, t1 and t2 is added in the list of dependencies of the resulting tensor, with the corresponding gradient functions.\n\nWith t1 = (n1,m1), t2 =(m1,m2) and t3 = t1 * t2 is (n1,m2) so the gradient coming from t3 is (n1,m2)\n\nd(t1*t2)/d(t1) = t2 –> multiply the incoming gradient transpose(t2.data)\nd(t1*t2)/d(t2) = t1, –> multiply transpose(t1) by the gradient\n\n\n\n\n\n","category":"function"},{"location":"autodiff/#Base.Broadcast.broadcasted","page":"Autodiff","title":"Base.Broadcast.broadcasted","text":"Base.:broadcasted(::typeof(+), t1::Tensor, t2::Tensor)\nBase.:broadcasted(::typeof(+), t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}}\nBase.:broadcasted(::typeof(+), notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}}\n\nBroadcast the + operator (perform element-wise addition). This works in the same way as the Base.:+ operator, but the method handle_broadcasting! is called  \n\n\n\n\n\nBase.:broadcasted(::typeof(-), t1::Tensor, t2::Tensor)\nBase.:broadcasted(::typeof(-), t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}}\nBase.:broadcasted(::typeof(-), notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}}\n\nBroadcast the - operator (perform element-wise substraction). This works in the same way as the Base.:- operator, but the method handle_broadcasting! is called  \n\n\n\n\n\nBase.:broadcasted(::typeof(*), t1::Tensor, t2::Tensor)\nBase.:broadcasted(::typeof(*), t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}}\nBase.:broadcasted(::typeof(*), notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}}\n\nBroadcast the * operator (perform element-wise multiplication). This works in the same way as the Base.:* operator, but the method handle_broadcasting! is called  \n\n\n\n\n\nBase.:broadcasted(::typeof(/), t1::Tensor, t2::Tensor)\nBase.:broadcasted(::typeof(/), t1::Tensor, notATensor::T) where {T<:Union{AbstractArray,Float64,Int64}}\nBase.:broadcasted(::typeof(/), notATensor::T, t1::Tensor) where {T<:Union{AbstractArray,Float64,Int64}}\n\nBroadcast the / operator (perform element-wise multiplication) between 2 tensors.\n\nd(t1/t2)/d(t1) = 1/t2 –> multiply the incoming gradient by 1/t2\nd(t1/t2)/d(t2) = -t1/t2^2, –> multiply the incoming gradient by -t1/t2^2\n\nThen, the method handle_broadcasting! is called on the result of the gradient computation wrt to t1 and/or t2 \n\n\n\n\n\n","category":"function"},{"location":"autodiff/#Math-functions-between-tensors","page":"Autodiff","title":"Math functions between tensors","text":"","category":"section"},{"location":"autodiff/","page":"Autodiff","title":"Autodiff","text":"Base.sum\nBase.:log\nBase.:tanh\nsigmoid\nrelu\nleakyrelu\nsoftmax\n","category":"page"},{"location":"autodiff/#Base.sum","page":"Autodiff","title":"Base.sum","text":"Base.sum(t::Tensor)\n\nReturn the sum of the tensor's elements.  The tensor returned requires gradient if the initial tensor requires it.\n\nFor the gradient function, incomingGradient is a one element tensor, because the output of the sum is a  scalar tensor. In the sum function, each element has the same weight  (1x1 + 1x2 + ... + 1*xn), so the gradient of this tensor wrt to the sum tensor is a tensor composed of ones, with the shape of the original tensor.\n\n\n\n\n\n","category":"function"},{"location":"autodiff/#Base.log","page":"Autodiff","title":"Base.log","text":"Base.:log(t1::Tensor)\n\nLog function to perform element-wise neperian logarithm on a tensor. The tensor returned requires gradient if the initial tensor requires it.\n\nd(ln(t1))/d(t1) = 1/t1 –> multiply the incoming gradient by 1/t1.\n\n\n\n\n\n","category":"function"},{"location":"autodiff/#Base.tanh","page":"Autodiff","title":"Base.tanh","text":"Base.:tanh(t1::Tensor)\n\nTanh function to perform elemnt-wise tanh on a tensor. The tensor returned requires gradient if the initial tensor requires it.\n\nd(tanh(t1))/d(t1) = (1-tanh^2(t1)) –> multiply the incoming gradient by (1-tanh^2(t1))\n\n\n\n\n\n","category":"function"},{"location":"autodiff/#NNJulia.Autodiff.sigmoid","page":"Autodiff","title":"NNJulia.Autodiff.sigmoid","text":"sigmoid(t1::Tensor)\n\nSigmoid function to perform elemnt-wise sigmoid on a tensor. The tensor returned requires gradient if the initial tensor requires it.\n\nd(sigmoid(t1))/d(t1) = sigmoid(t1)(1-sigmoid(t1)) –> multiply the incoming gradient by sigmoid(t1)(1-sigmoid(t1))\n\n\n\n\n\n","category":"function"},{"location":"autodiff/#NNJulia.Autodiff.relu","page":"Autodiff","title":"NNJulia.Autodiff.relu","text":"relu(t1::Tensor)\n\nRelu function to perform elemnt-wise relu on a tensor. The tensor returned requires gradient if the initial tensor requires it.\n\nd(relu(t1))/d(t1) =  1 if t1>0, else 0 –> multiply the incoming gradient by (t1 .> 0)\n\n\n\n\n\n","category":"function"},{"location":"autodiff/#NNJulia.Autodiff.leakyrelu","page":"Autodiff","title":"NNJulia.Autodiff.leakyrelu","text":"leakyrelu(t1::Tensor)\n\nleaky relu function to perform elemnt-wise leaky relu on a tensor. The tensor returned requires gradient if the initial tensor requires it.\n\nd(leakyrelu(t1,a))/d(t1) =  1 if t1>0, else a –> multiply the incoming gradient by 1 or a depending on the data\n\n\n\n\n\n","category":"function"},{"location":"autodiff/#NNJulia.Autodiff.softmax","page":"Autodiff","title":"NNJulia.Autodiff.softmax","text":"softmax(t1::Tensor)\n\nSoftmax function to perform softmax on a tensor. The tensor returned requires gradient if the initial tensor requires it.\n\nd(softmax(t1))/d(t1) = softmax(t1)(1-softmax(t1)) –> multiply the incoming gradient by softmax(t1)(1-softmax(t1))\n\n\n\n\n\n","category":"function"},{"location":"utils/#Utils","page":"Utils","title":"Utils","text":"","category":"section"},{"location":"utils/","page":"Utils","title":"Utils","text":"split_train_test\nplot_decision_boundary\nto_one_hot","category":"page"},{"location":"utils/#NNJulia.split_train_test","page":"Utils","title":"NNJulia.split_train_test","text":"split_train_test(xData::AbstractArray, yData::AbstractArray, at::Float64)\n\nSplit the xData and yData (shuffled) into a train and a test set.\n\nFor example, with at=0.8, the train set will contain 80% of the original data and the test set will contain 20% of it.\n\n\n\n\n\n","category":"function"},{"location":"utils/#NNJulia.plot_decision_boundary","page":"Utils","title":"NNJulia.plot_decision_boundary","text":"plot_decision_boundary(model::AbstractModel, xData::AbstractArray, yData::AbstractArray, levels::Int64=3)\n\nPlot a decision frontier of the model for 2D data with the given number of levels.\n\n\n\n\n\n","category":"function"},{"location":"utils/#NNJulia.to_one_hot","page":"Utils","title":"NNJulia.to_one_hot","text":"to_one_hot(x::AbstractArray)\n\nConvert an array to one hot encoded format.\n\n\n\n\n\n","category":"function"}]
}
