include("../src/NNJulia.jl")
using .NNJulia

function minimize()

    # Initialise the data that will be minimized
    x = Tensor(100 .* rand(3, 2), true)

    # Initialise the learning rate
    lr = 0.1

    # 100 iterations
    for i = 1:100

        # Set the gradient to 0
        zero_grad!(x)
        # sum the square of the data
        sumOfSquare = sum(x .* x)

        # backward the gradient 
        backward!(sumOfSquare)

        # update the data with their gradient
        x.data -= lr * x.gradient
        println("step " * string(i) * " :" * string(round.(x.data, digits=3)))
    end

end

@time minimize()


