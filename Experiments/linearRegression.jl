include("../src/Autodiff.jl")
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

        # ici a est un nouveau tensor à chaque fois
        if (i == 0 || i % 100 == 0)
            println("step " * string(i) * ": a=" * string(round.(a.data, digits = 3)) * " b=" * string(round.(b.data, digits = 3)))
        end
        if (i == 200)
            println(a)
            println(b)
            println(X)
            println(Y)
            println(y)
            println(loss)
        end
    end

    println("a : ")
    println(a)
    println("\nb : ")
    print(b)

end

# test performances :
@time linear_regression()