# train, test, eval

"""
    TrainParameters(opt::AbstractOptimiser, lossFunction::AbstractLoss, metrics::AbstractMetrics)

This struct store the important parameters used to train the model.

# Fields
- opt: The optimiser used to optimise the loss
- lossFunction: The function used to compute the loss
- metrics: The metrics used to compute the accuracy of the model
"""
struct TrainParameters
    opt::AbstractOptimiser
    lossFunction::AbstractLoss
    metrics::AbstractMetrics
end

"""
    train!(model::AbstractModel, trainParams::TrainParameters, trainData::DataLoader, nbEpochs::Int, verbose::Bool=true)

This method train a model on the trainData.
The accuracy and the loss computed at each epoch is stored into a dictionnary that is returned at the end
of the training.

The dictionnary returned looks like this : 
``` history = Dict("accuracy" => Float64[], "loss" => Float64[]) ```

"""
function train!(model::AbstractModel, trainParams::TrainParameters, trainData::DataLoader, nbEpochs::Int, verbose::Bool=true)
    history = Dict("accuracy" => Float64[], "loss" => Float64[])

    for epoch in 1:nbEpochs
        epochLoss = 0.0
        accuracy = 0.0

        #TODO: implement K-fold cross validation
        cpt = 0
        for batch in trainData
            cpt += 1

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
            #println(loss)
            #println(predictions)



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

"""
    evaluate(model::AbstractModel, metrics::BinaryAccuracy, xData::Union{Tensor,AbstractArray,Float64,Int64}, yData::Union{Tensor,AbstractArray,Float64,Int64})

This method evaluate a model by returning the accuracy computed with the given metrics
"""
function evaluate(model::AbstractModel, metrics::AbstractMetrics, xData::Union{Tensor,AbstractArray,Float64,Int64}, yData::Union{Tensor,AbstractArray,Float64,Int64})
    predictions = model(xData)
    return compute_accuracy(metrics, predictions, yData)
end