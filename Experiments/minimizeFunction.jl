include("../src/Autodiff/Autodiff.jl")
using .Autodiff

function minimize()

    x = Tensor(100 .* rand(3, 2), true)

    lr = 0.1

    for i = 1:100
        zero_grad!(x)
        sumOfSquare = sum(x .* x)
        backward!(sumOfSquare)

        x.data -= lr * x.gradient
        println("step " * string(i) * " :" * string(round.(x.data, digits = 3)))
    end

end

@time minimize()


