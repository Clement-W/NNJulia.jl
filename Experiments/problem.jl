include("../src/NNJulia.jl")
using .NNJulia


xData = reshape([0; 1; 1; 0], 4, 1)
yData = [1 0; 0 1; 0 1; 1 0]

model = Sequential(
    Dense(1, 2, tanh)
)

opt = GradientDescent(0.1)
batchsize = 3
lossFunction = MSE

trainData = DataLoader(xData, yData, batchsize)

loss = 0
for batch in trainData

    # Get input data for this batch
    inputs = batch[1]
    # Get actual data for this batch
    actual = batch[2]

    # Set the parameter's gradients to 0
    zero_grad!(model)

    # Model's prediction with input data
    predictions = model(inputs)

    # Compute the loss for this batch
    loss = lossFunction(predictions, actual)

    # Backpropagate the error through gradients
    backward!(loss)

    update!(opt, model)
end
