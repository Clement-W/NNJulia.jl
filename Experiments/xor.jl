include("../src/NNJulia.jl")
using .NNJulia

function xor()


    xData = [
        0 1 0 1
        0 0 1 1]
    yData = [0 1 1 0]

    model = Sequential(
        Dense(2, 8, relu),
        Dense(8, 8, relu),
        Dense(8, 1, sigmoid),
    )

    opt = GradientDescent(0.1)
    loss = BinaryCrossentropy()
    metrics = BinaryAccuracy()
    batchsize = 4
    nbEpochs = 500

    trainParams = TrainParameters(opt, loss, metrics)

    trainData = DataLoader(xData, yData, batchsize)

    train!(model, trainParams, trainData, nbEpochs)

    println("prediction on xData : ")
    println(round.(predict(model, xData).data, digits=1))

end

xor()