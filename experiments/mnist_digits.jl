# Import Dataset loader 
using MLDatasets

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
opt = GradientDescent(0.0001)
loss = CategoricalCrossentropy()
metrics = CategoricalAccuracy()

# Pass it to the TrainParameters struct that will be used during training
trainParams = TrainParameters(opt, loss, metrics)

# Training specifications
batchsize = 64
nbEpochs = 15;

trainData = DataLoader(train_x, train_y_hot, batchsize, true);

history = train!(model, trainParams, trainData, nbEpochs, true)
