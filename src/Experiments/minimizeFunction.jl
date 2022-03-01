include("../Autodiff.jl")
using .Autodiff

x = Tensor(100 .* rand(3, 2))


lr = 0.1

for i = 1:100
    zero_grad!(x)
    sumOfSquare = sum(x .* x)
    backward!(sumOfSquare)

    x -= lr * x.gradient
    println("step " * string(i) * " :" * string(round.(x.data, digits = 3)))
end



