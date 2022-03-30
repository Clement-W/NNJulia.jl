# train, test, eval

function train!(model::AbstractModel, opt::AbstractOptimiser, lossFunction::Function, trainData::DataLoader, nbEpochs::Int, verbose::Bool=true)
    #history = [] TODO: history array (like keras)
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
            accuracy += computeAccuracy(predictions, actual)

            # Compute the loss for this batch
            loss = lossFunction(predictions, actual)

            # Backpropagate the error through gradients
            backward!(loss)

            epochLoss += loss

            update!(opt, model)
        end
        accuracy = accuracy / length(trainData)

        if (verbose)
            println("Epoch " * string(epoch) * " : accuracy = " * string(accuracy) * "%, loss = " * string(epochLoss.data))
        end
    end

end


function computeAccuracy(predictions::Tensor, actual::T) where {T<:Union{AbstractArray,Float64,Int64}}
    #return sum(round.(predictions.data, digits=1) == actual)
    #TODO:
end