include("../src/Autodiff/Autodiff.jl")
using .Autodiff

function linear_regression()
    f(a, b, x) = a .* x .+ b
    MSE(y, expected_y) = sum((expected_y .- y) .* (expected_y .- y))

    X = Tensor(rand(40) * 10)
    Y = f(2, 9, X) #2*x + 9
    data = (X, Y)

    a = Tensor(rand(), true)
    b = Tensor(rand(), true)
    lr = 0.0001

    println("real a = 2 and real b = 9")
    println("random a = " * string(a.data))
    println("random b = " * string(b.data))


    for i = 1:4000
        zero_grad!(a)
        zero_grad!(b)

        y = f(a, b, data[1])
        loss = MSE(y, data[2])

        backward!(loss)

        a.data -= lr * a.gradient
        b.data -= lr * b.gradient
        # Change the data of the tensor

        if (i == 0 || i % 100 == 0)
            println("step " * string(i) * ": a=" * string(round.(a.data, digits = 3)) * " b=" * string(round.(b.data, digits = 3)))
        end
    end

    print("a : ")
    print(a)
    print("\nb : ")
    print(b)
    println()

end

# test performances :
@time linear_regression()