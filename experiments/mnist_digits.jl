# Import Dataset loader 
using MLDatasets
using Plots

# Import NNJulia
include("../src/NNJulia.jl")
using .NNJulia

# load full training set
train_x, train_y = MNIST.traindata();

# load full test set
test_x, test_y = MNIST.testdata();
# if this doesn't work, copy these lines il julia repl to download the data , it will work here after

#train_x = train_x[:, :, 1:7984] #FIXME: nombre environ de limite de données qui retourne pas Nan
#train_y = train_y[1:7984]

# one-hot encoding of the labels
train_y_hot = to_one_hot(train_y)
test_y_hot = to_one_hot(test_y)



model = Sequential(
    Flatten(),
    Dense(784, 16, relu),
    Dense(16, 16, relu),
    Dense(16, 10, softmax),
)

# Initialise the optimiser, the loss function and the metrics used to compute accuracy
opt = GradientDescent(0.05)
loss = BinaryCrossentropy() # FIXME: Does not work with categorical crossentropy yet
metrics = CategoricalAccuracy()

# Pass it to the TrainParameters struct that will be used during training
trainParams = TrainParameters(opt, loss, metrics)

# Training specifications
batchsize = 64
nbEpochs = 25;

trainData = DataLoader(train_x, train_y_hot, batchsize, true);

history = train!(model, trainParams, trainData, nbEpochs, true)

p1 = plot(history["accuracy"], label="Accuracy", legend=:topleft)
p2 = plot(history["loss"], label="Loss")
plot(p1, p2, layout=2)

acc = evaluate(model, metrics, test_x, test_y_hot)
println("accuracy on test data = " * string(acc * 100) * "%")

plots = []
for i in 1:6
    r = rand(1:10000)
    img = (Gray.(permutedims(test_x[:, :, r])))
    preds = model(reshape(test_x[:, :, r], :, 1))
    predicted_label = argmax(preds.data)[1] - 1
    push!(plots, plot(img, title="pred = " * string(predicted_label)))
end

plot(plots..., layout=6)