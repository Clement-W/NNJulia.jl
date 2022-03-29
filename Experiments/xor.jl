include("../src/NNJulia.jl")
using .NNJulia

function xor()

    # yData is encoded this way : [1,0] if 0 and [0,1] if 1
    xData = [0 0; 1 0; 0 1; 1 1]
    yData = reshape([0, 1, 1, 0], 4, 1)

    model = Sequential(
        Dense(2, 4, tanh),
        Dense(4, 4, tanh),
        Dense(4, 1, sigmoid)
    )

    opt = GradientDescent(0.1)
    loss = BinaryCrossentropy
    batchsize = 4
    # PB quand le batch size est > 1, pb quand on ubdate les biais
    # ça fait (5,) + (batchsize,)
    # TODO: trouver comment regler ça (avec le dataloader ?)
    nbEpochs = 500

    trainData = DataLoader(xData, yData, batchsize)

    train!(model, opt, loss, trainData, nbEpochs)

end

xor()