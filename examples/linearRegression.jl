using NNJulia

function linear_regression()

    # Create the function ax+b
    f(a, b, x) = a .* x .+ b
    # Loss function
    MSE(y, expected_y) = sum((expected_y .- y) .* (expected_y .- y))

    # Create input data
    X = (rand(40) * 10)
    # Output data
    Y = f(2, 9, X) #2*x + 9

    data = (X, Y)

    # Initialise a from ax+b
    a = Tensor(rand(), true)
    # Initialise b from ax+b
    b = Tensor(rand(), true)

    # Initialise learning rate
    lr = 0.0001

    println("target a = 2 and target b = 9")
    println("random a = " * string(a.data))
    println("random b = " * string(b.data))


    # 4000 iterations
    for i = 1:4000

        # Set the gradients of a and b to 0
        zero_grad!(a)
        zero_grad!(b)

        # compute output
        y = f(a, b, data[1])
        # compute the loss
        loss = MSE(y, data[2])

        # backward the gradient from the loss
        backward!(loss)

        # Update a and b with their gradient wrt the loss
        a.data -= lr * a.gradient
        b.data -= lr * b.gradient

        if (i == 0 || i % 100 == 0)
            println("step " * string(i) * ": a=" * string(round.(a.data, digits=3)) * " b=" * string(round.(b.data, digits=3)))
        end
    end

    print("a : ")
    print(a.data)
    print("\nb : ")
    print(b.data)
    println()

end

# test performances :
@time linear_regression()
# @benchmark linear_regression()