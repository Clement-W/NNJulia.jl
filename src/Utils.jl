#TODO: implement to_categorical

using Random
using Plots
using .Layers

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


# plot the decision frontier for 2D data
function plot_decision_boundary(model::AbstractModel, xData::AbstractArray, yData::AbstractArray, levels::Int64=3)

    # set min and max values and give it some padding
    x_min = minimum(xData[1, :]) - 0.1
    x_max = maximum(xData[1, :]) + 0.1

    y_min = minimum(xData[2, :]) - 0.1
    y_max = maximum(xData[2, :]) + 0.1

    h = 0.1
    # Generate a grid of points with distance h between them
    xrange = range(x_min, x_max, step=h)
    yrange = range(y_min, y_max, step=h)

    xx = xrange' .* ones(length(yrange))
    yy = ones(length(xrange))' .* yrange

    # Predict the function value for the whole grid
    preds = model(permutedims([xx[:] yy[:]]))
    z = reshape(preds.data, size(xx)[1], size(xx)[2])

    # Plot the contour and data values
    contourf(xrange, yrange, z, levels=levels, c=:BuPu_9)
    scatter!(xData[1, :], xData[2, :], group=yData[:])

end
