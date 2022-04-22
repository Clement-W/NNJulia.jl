
@testset "Test loss package" begin
    @testset "Test MSE function" begin
        @testset "Predicted and actual are tensors" begin
            actual = Tensor([1, 2, 3], true)
            predicted = Tensor([1.1, 2.1, 3.1], true)

            loss = compute_loss(MSE(), predicted, actual)
            @test isapprox(loss.data, 0.03)

            backward!(loss)

            @test isapprox(predicted.gradient, [0.2, 0.2, 0.2])
            @test isapprox(actual.gradient, [-0.2, -0.2, -0.2])
        end

        @testset "Predicted is a tensor, and actual is a vector" begin
            actual = [1, 2, 3]
            predicted = Tensor([1.1, 2.1, 3.1], true)

            loss = compute_loss(MSE(), predicted, actual)
            @test isapprox(loss.data, 0.03)

            backward!(loss)

            @test isapprox(predicted.gradient, [0.2, 0.2, 0.2])
        end
    end

    @testset "Test Binary Crossentropy function" begin

        @testset "Predicted and actual are tensors" begin
            actual = Tensor([0, 0, 1], true)
            predicted = Tensor([0.05, 0.05, 0.9], true)

            loss = compute_loss(BinaryCrossentropy(), predicted, actual)

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

            loss = compute_loss(BinaryCrossentropy(), predicted, actual)

            expectedLoss = log(1 - 0.05) + log(1 - 0.05) + log(0.9)
            expectedLoss = expectedLoss * -1 / 3 # 0.06931570147764247
            @test isapprox(loss.data, expectedLoss)

            backward!(loss)
            # predicted.gradient = [0.3508771929824561, 0.3508771929824561, -0.37037037037037035]

        end
    end

    @testset "Test Categorical Crossentropy function" begin

        @testset "Predicted and actual are tensors" begin
            actual = Tensor([0 0; 1 0; 0 1], true)
            predicted = Tensor([0.05 0.1; 0.95 0.8; 0 0.1], true)

            loss = compute_loss(CategoricalCrossentropy(), predicted, actual)

            expectedLoss = 1.176939193690798
            @test isapprox(loss.data, expectedLoss)

            backward!(loss)
            #grad actual = [1.4978661367769954 1.1512925464970227; 0.02564664719377529 0.11157177565710485; 8.05904782547916 1.1512925464970227]
            #grad predicted = [0.0 0.0; -0.5263157894736842 0.0; 0.0 -5.0]

        end

        @testset "Predicted is a tensor, and actual is a vector" begin
            actual = [0 0; 1 0; 0 1]
            predicted = Tensor([0.05 0.1; 0.95 0.8; 0 0.1], true)

            loss = compute_loss(CategoricalCrossentropy(), predicted, actual)

            expectedLoss = 1.176939193690798
            @test isapprox(loss.data, expectedLoss)

            backward!(loss)
            #grad predicted = [0.0 0.0; -0.5263157894736842 0.0; 0.0 -5.0]
        end
    end
end;