using .Layers

"""
    GradientDescent(lr::Float64)
    GradientDescent()

Represents the vanilla gradient descent optimiser.
The default constructor initialise the learning rate at 0.1

# Field
- lr: The learning rate
"""
struct GradientDescent <: AbstractOptimiser
    lr::Float64
end

# default constructor, initialising the learning rate at 0.1
GradientDescent() = GradientDescent(0.1)

# update the parameters of a model with gradient descent algorithm 

"""
    update!(opt::GradientDescent, model::AbstractModel)

Update the parameters of the model using the given optimiser.
"""
function update!(opt::GradientDescent, model::AbstractModel)
    # vanilla gradient descent
    for p in parameters(model)
        p.data = p.data .- opt.lr .* p.gradient
    end
end