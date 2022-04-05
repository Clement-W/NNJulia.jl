# Autodiff.jl

CurrentModule = NNJulia.Autodiff

## Structs and types

```@docs
AbstractTensor
TensorDependency
Tensor

```

## Base methods dispatched

```@docs
Base.setproperty!
Base.size
Base.ndims
Base.length
Base.iterate
Base.show
```

## Methods for the gradient

```@docs
zero_grad!
backward!
handle_broadcasting!
```

## Operators between tensors

```@docs
Base.:+
Base.:-
Base.:*
Base.:broadcasted
```

## Math functions

```@docs
Base.sum
Base.:log
Base.:tanh
sigmoid
relu
leakyrelu

```
