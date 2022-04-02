#TODO: implement to_categorical

using Random

function split_train_test(xData::AbstractArray, yData::AbstractArray, at::Float64)
    size(xData)[ndims(xData)] == size(yData)[ndims(yData)] || throw("xData and yData must have the same number of samples")
    nbSamples = size(xData)[ndims(xData)]

    indexes = shuffle(1:nbSamples)
    train_idx = view(indexes, 1:floor(Int, nbSamples * at))
    test_idx = view(indexes, (floor(Int, at * nbSamples)+1):nbSamples)

    x_train = xData[:, train_idx]
    x_test = xData[:, test_idx]

    y_train = yData[:, train_idx]
    y_test = yData[:, test_idx]

    return x_train, y_train, x_test, y_test
end