include("../src/NNJulia.jl")
using .NNJulia

function xor()

    # yData is encoded this way : [1,0] if 0 and [0,1] if 1
    xData = [0 0; 1 0; 0 1; 1 1]
    yData = [1 0; 0 1; 0 1; 1 0]

    model = Sequential(
        Dense(2, 5, tanh),
        Dense(5, 5, tanh),
        Dense(5, 2, tanh)
    )

    opt = GradientDescent(0.03)
    loss = MSE
    batchsize = 1
    # PB quand le batch size est > 1, pb quand on ubdate les biais
    # ça fait (5,) + (batchsize,)
    nbEpochs = 500

    trainData = DataLoader(xData, yData, batchsize)

    train!(model, opt, loss, trainData, nbEpochs)

end

xor()