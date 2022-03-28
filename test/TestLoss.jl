include("../src/NNJulia.jl")
using .NNJulia
using Test


@testset "Test MSE function" begin
    @testset "Predicted and actual are tensors" begin
        actual = Tensor([1, 2, 3], true)
        predicted = Tensor([1.1, 2.1, 3.1], true)

        loss = MSE(predicted, actual)
        @test isapprox(loss.data, 0.03)

        backward!(loss)

        @test isapprox(predicted.gradient, [0.2, 0.2, 0.2])
        @test isapprox(actual.gradient, [-0.2, -0.2, -0.2])
    end

    @testset "Predicted is a tensor, and actual is a vector" begin
        actual = [1, 2, 3]
        predicted = Tensor([1.1, 2.1, 3.1], true)

        loss = MSE(predicted, actual)
        @test isapprox(loss.data, 0.03)

        backward!(loss)

        @test isapprox(predicted.gradient, [0.2, 0.2, 0.2])
    end
end

@testset "Test Binary Crossentropy function" begin

    @testset "Predicted and actual are tensors" begin
        actual = Tensor([0, 0, 1], true)
        predicted = Tensor([0.05, 0.05, 0.9], true)

        loss = BinaryCrossentropy(predicted, actual)

        expectedLoss = log(1 - 0.05) + log(1 - 0.05) + log(0.9)
        expectedLoss = expectedLoss * -1 / 3 # 0.06931570147764247
        @test isapprox(loss.data, expectedLoss)

        backward!(loss)
        # actual.gradient = [0.9814796597221467, 0.9814796597221467, -0.7324081924454066]
        # predicted.gradient = [0.3508771929824561, 0.3508771929824561, -0.37037037037037035]

    end

    @testset "Predicted is a tensor, and actual is a vector" begin
        actual = [0, 0, 1]
        predicted = Tensor([0.05, 0.05, 0.9], true)

        loss = BinaryCrossentropy(predicted, actual)

        expectedLoss = log(1 - 0.05) + log(1 - 0.05) + log(0.9)
        expectedLoss = expectedLoss * -1 / 3 # 0.06931570147764247
        @test isapprox(loss.data, expectedLoss)

        backward!(loss)
        # actual.gradient = [0.9814796597221467, 0.9814796597221467, -0.7324081924454066]
        # predicted.gradient = [0.3508771929824561, 0.3508771929824561, -0.37037037037037035]

    end
end