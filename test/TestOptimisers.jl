include("../src/NNJulia.jl")
using .NNJulia
using Test

@testset "Test Gradient Descent" begin
    @testset "Constructors" begin
        @testset "Base constructor" begin
            opt = GradientDescent(0.5)
            @test opt.lr == 0.5
        end

        @testset "Consturctor taking no arguments" begin
            opt = GradientDescent()
            @test opt.lr == 0.1
        end
    end

    @testset "Update parameters of a sequential model" begin
        w1 = Tensor(2, 100)
        b1 = Tensor(1, 200)
        d1 = Dense(w1, b1)

        w2 = Tensor(4, 300)
        b2 = Tensor(7, 400)
        d2 = Dense(w2, b2)

        s = Sequential(d1, d2)

        opt = GradientDescent(1)

        update!(opt, s)

        @test s.layers[1].weight.data == -98
        @test s.layers[1].bias.data == -199

        @test s.layers[2].weight.data == -296
        @test s.layers[2].bias.data == -393

    end
end;