using .Layers

struct GradientDescent <: AbstractOptimiser
    lr::Float64
end

GradientDescent() = GradientDescent(0.1)

function update!(opt::GradientDescent, model::AbstractModel)
    for p in parameters(model)
        p.data = p.data .- opt.lr .* p.gradient
    end
end