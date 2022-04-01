include("../src/NNJulia.jl")
using .NNJulia

function xor()


    xData = [
        0 1 0 1
        0 0 1 1]

    # yData is one-hot encoded with [1 0] for a 0 and [0 1] for a 1
    yData = [
        1 0 0 1
        0 1 1 0]

    model = Sequential(
        Dense(2, 8, relu),
        Dense(8, 8, relu),
        Dense(8, 2, sigmoid),
    )

    opt = GradientDescent(0.1)
    loss = BinaryCrossentropy()
    metrics = BinaryAccuracy(0.9)
    batchsize = 4
    nbEpochs = 500

    trainParams = TrainParameters(opt, loss, metrics)

    trainData = DataLoader(xData, yData, batchsize)

    train!(model, trainParams, trainData, nbEpochs)

    println("\nEvaluate the model : ")
    print("Accuracy = ")
    acc = evaluate(model, metrics, xData, yData) * 100
    println(string(acc) * "%\n")

    println("prediction on xData : ")
    round.(predict(model, xData).data, digits=1)

end

xor()