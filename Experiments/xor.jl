include("../src/NNJulia.jl")
using .NNJulia

function xor()

    xData = [0 0; 1 0; 0 1; 1 1]
    yData = reshape([0, 1, 1, 0], 4, 1)

    model = Sequential(
        Dense(2, 8, tanh),
        Dense(8, 1, sigmoid),
    )

    opt = GradientDescent(0.1)
    loss = BinaryCrossentropy
    batchsize = 4
    nbEpochs = 500

    trainData = DataLoader(xData, yData, batchsize)

    train!(model, opt, loss, trainData, nbEpochs)

end

xor()