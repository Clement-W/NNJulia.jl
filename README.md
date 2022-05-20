<br />
<h1 align="left">NNJulia</h1>


[![Build status (Github Actions)](https://github.com/Clement-W/NNJulia.jl/workflows/CI/badge.svg)](https://github.com/Clement-W/NNJulia.jl/actions)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://clement-w.github.io/NNJulia.jl/dev/)
[![codecov](https://codecov.io/gh/Clement-W/NNJulia.jl/branch/main/graph/badge.svg?token=J0K5VA4UXG)](https://codecov.io/gh/Clement-W/NNJulia.jl)

## About The Project

Using famous deep learning libraries such as Keras or PyTorch is nice, but it's even better to understand how they work!
That is why I have decided to implement a neural network library from scratch. This project is conducted for educational purposes and is not finished yet. 

The programming language used for this project is Julia. Julia is a recent programming language with a fast-growing community. It is made for scientific computation, and is famous for being faster than Python or Matlab. This open source programming language is dynamically typed and uses multiple dispatch as a paradigm. As Julia is designed for high performance, it makes it perfect for data science, machine learning and deep learning. When I started the project, I did not know anything about Julia, so it was also a good way to learn this language.

NNJulia is based on the principle of automatic differentiation (AD). AD is a general way of evaluating the derivative of a function defined by a computer program. It relies on a calculus formula named the chain rule. By applying the chain rule on every elementary mathematical operation and functions involved in a complicated mathematical function, derivatives of arbitrary order can be computed automatically. A fast reverse-mode automatic differentiation is mandatory during the backpropagation phase. It allows to compute efficiently the gradients required to update the parameters of a neural network. 

To implement AD, a structure Tensor is used. A tensor is a N-dimensional array that can be differentiated. By using mathematical operations and functions between tensors, a computational graph is implicitly created. This allows to store the dependencies between tensors, with respect to the operations that link them.

In the end the library NNJulia offers a simple programming interface to create and train sequential neural networks composed of dense layers (convolutional layers are not implemented yet). [The documentation is also available online.](https://clement-w.github.io/NNJulia.jl/dev/)

As you can see in examples/ it is possible to classify handwritten digits from the MNIST dataset with a decent accuracy. An example of spiral data points classification is also available.


## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

* [Julia 1.7](https://julialang.org/)

### Installation


1. Clone the repo and launch Julia
```sh
git clone https://github.com/Clement-W/NNJulia.jl
cd NNJulia.jl
julia
```
   
2. Instantiate the package
```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
julia> using NNJulia
```
   
3. Now you're ready to use NNJulia !


## Usage

### Create a model 

```julia
model = Sequential(
        Flatten(),
        Dense(784, 16, relu),
        Dense(16, 16, relu),
        Dense(16, 10, softmax),
    )
  ```
 
 ### Prepare for training
 
 ```julia
# Initialise the optimiser, the loss function and the metrics used to compute accuracy
opt = GradientDescent(0.05)
loss = BinaryCrossentropy()
metrics = BinaryAccuracy()

# Pass it to the TrainParameters struct that will be used during training
trainParams = TrainParameters(opt, loss, metrics)

# Training specifications
batchsize = 64
nbEpochs = 25;
 ```
 
 ### Load the data into a DataLoader
 
 ```julia
trainData = DataLoader(train_x, train_y_hot, batchsize,shuffle=true);
 ```
 
 ### Train the model
 
 ```julia
 history = train!(model, trainParams, trainData, nbEpochs)
 ```
 
Plot the evolution of accuracy and loss during training:
```julia
p1 = plot(history["accuracy"],label="Accuracy",legend=:topleft)
p2 = plot(history["loss"],label="Loss")
plot(p1,p2,layout=2)
```
 
 ###  Evaluate the model
 
 
 ```julia
acc = evaluate(model,metrics,test_x,test_y_hot)
println("accuracy on test data = " * string(acc*100) * "%")
```
 
 
