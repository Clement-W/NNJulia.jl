using NNJulia

function xor(verbose::Bool=true)

    # Create xData with the 4 possibilites of input for xor
    xData = [
        0 1 0 1
        0 0 1 1]

    # yData is one-hot encoded with [1 0] for a 0 and [0 1] for a 1
    yData = [
        1 0 0 1
        0 1 1 0]


    # Create the model
    model = Sequential(
        Dense(2, 8, relu),
        Dense(8, 8, relu),
        Dense(8, 2, sigmoid),
    )

    # Initialise the optimiser, the loss function and the metrics used to compute accuracy
    opt = GradientDescent(0.1)
    loss = BinaryCrossentropy()
    metrics = BinaryAccuracy(0.9)

    # Pass it to the TrainParameters struct that will be used during training
    trainParams = TrainParameters(opt, loss, metrics)


    # Training specifications
    batchsize = 4
    nbEpochs = 500

    # Load the train data into a dataloader
    trainData = DataLoader(xData, yData, batchsize)

    # train the model
    train!(model, trainParams, trainData, nbEpochs, verbose)

    if (verbose)
        println("\nEvaluate the model : ")
        print("Accuracy = ")
    end



    if (verbose)
        # evaluate the model
        acc = evaluate(model, metrics, xData, yData) * 100
        println(string(acc) * "%\n")
    end

    if (verbose)
        println("prediction on xData : ")
        round.(model(xData).data, digits=1)
    end

end

xor()
# test performances with @benchmark xor(false)