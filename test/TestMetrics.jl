include("../src/NNJulia.jl")
using .NNJulia
using Test


@testset "Test binary accuracy" begin
    @testset "100% accuracy with a batch size of 4" begin
        pred = [
            0.8 0.2 0.1 0.9
            0 0.98 0.78 0.12
            0 0.1 0.2 0.9]

        target = [
            1 0 0 1
            0 1 1 0
            0 0 0 1]

        predicted = Tensor(pred)

        metrics = BinaryAccuracy(0.7)

        acc = compute_accuracy(metrics, predicted, target)
        @test acc == 1
    end


    @testset "75% accuracy with a batch size of 4" begin
        pred = [
            0.8 0.2 0.1 0.9
            0 0.98 0.68 0.12
            0 0.1 0.2 0.9]

        target = [
            1 0 0 1
            0 1 1 0
            0 0 0 1]

        predicted = Tensor(pred)

        metrics = BinaryAccuracy(0.7)

        acc = compute_accuracy(metrics, predicted, target)
        @test acc == 0.75
    end

    @testset "0% accuracy with a batch size of 1" begin
        pred = [
            0.78
            0
            0]

        target = [
            1
            0
            0]

        predicted = Tensor(pred)

        metrics = BinaryAccuracy(0.79)

        acc = compute_accuracy(metrics, predicted, target)
        @test acc == 0
    end

    @testset "100% accuracy with a batch size of 1 (with vectors)" begin
        pred = [0.8, 0, 0]

        target = [1, 0, 0]

        predicted = Tensor(pred)

        metrics = BinaryAccuracy(0.79)

        acc = compute_accuracy(metrics, predicted, target)
        @test acc == 1
    end
end
