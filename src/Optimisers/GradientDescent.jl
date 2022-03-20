using .Layers

# struct storing the learning rate for gradient descent optimiser
struct GradientDescent <: AbstractOptimiser
    lr::Float64
end

# default constructor, initialising the learning rate at 0.1
GradientDescent() = GradientDescent(0.1)

# update the parameters of a model with gradient descent algorithm 
function update!(opt::GradientDescent, model::AbstractModel)
    for p in parameters(model)
        p.data = p.data .- opt.lr .* p.gradient
    end
end