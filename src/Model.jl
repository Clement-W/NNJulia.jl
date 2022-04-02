# train, test, eval

struct TrainParameters
    opt::AbstractOptimiser
    lossFunction::AbstractLoss
    metrics::AbstractMetrics
end


function train!(model::AbstractModel, trainParams::TrainParameters, trainData::DataLoader, nbEpochs::Int, verbose::Bool=true)
    history = Dict("accuracy" => Float64[], "loss" => Float64[])

    for epoch in 1:nbEpochs
        epochLoss = 0.0
        accuracy = 0.0

        #TODO: implement K-fold cross validation

        for batch in trainData

            # Get input data for this batch
            inputs = batch[1]
            # Get actual data for this batch
            actual = batch[2]

            # Set the parameter's gradients to 0
            zero_grad!(model)

            # Model's prediction with input data
            predictions = model(inputs)

            # Compute accuracy for this batch
            accuracy += compute_accuracy(trainParams.metrics, predictions, actual)

            # Compute the loss for this batch
            loss = compute_loss(trainParams.lossFunction, predictions, actual)

            # Backpropagate the error through gradients
            backward!(loss)

            epochLoss += loss

            update!(trainParams.opt, model)
        end
        accuracy = accuracy / length(trainData)

        push!(history["accuracy"], accuracy)
        push!(history["loss"], epochLoss.data)

        if (verbose)
            println("Epoch " * string(epoch) * " : accuracy = " * string(accuracy) * ", loss = " * string(epochLoss.data))
        end
    end

    return history

end

function evaluate(model::AbstractModel, metrics::BinaryAccuracy, xData::Union{Tensor,AbstractArray,Float64,Int64}, yData::Union{Tensor,AbstractArray,Float64,Int64})
    predictions = model(xData)
    return compute_accuracy(metrics, predictions, yData)
end

function predict(model::AbstractModel, inputs::Union{Tensor,AbstractArray,Int64,Float64})
    return model(inputs)
end
