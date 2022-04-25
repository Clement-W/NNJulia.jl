# Autodiff

## Structs and types

```@docs
AbstractTensor
TensorDependency
Tensor

```

## Methods for the gradient

```@docs
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

## Math functions between tensors

```@docs
Base.sum
Base.:log
Base.:tanh
sigmoid
relu
leakyrelu
softmax

```
