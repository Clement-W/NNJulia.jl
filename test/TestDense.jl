include("../src/NNJulia.jl")
using .NNJulia
using Test



@testset "Constructors" begin

    @testset "Base constructor" begin
        d = Dense(Tensor([1 1 1; 2 2 2], true), Tensor([3, 3], true))

        @test d.weight.data == [1 1 1; 2 2 2]
        @test d.weight.gradient == [0 0 0; 0 0 0]

        @test d.bias.data == [3, 3]
        @test d.bias.gradient == [0, 0]

        @test d.activation == identity

        @test_throws ErrorException Dense(Tensor([1 1; 2 2; 3 3]), Tensor([1 1]))
        @test_throws ErrorException Dense(Tensor([1 1 1; 2 2 2]), Tensor([3, 3]))
        @test_throws ErrorException Dense(Tensor([1 1 1; 2 2 2]), Tensor([3, 3], true))
        @test_throws ErrorException Dense(Tensor([1 1 1; 2 2 2], true), Tensor([3, 3]))

        d2 = Dense(Tensor(3, true), Tensor(1, true))
        @test d2.weight.data == 3
        @test d2.weight.gradient == 0
        @test d2.bias.data == 1
        @test d2.bias.gradient == 0

    end


    @testset "Test constructor taking input and output size" begin
        d = Dense(2, 3)
        @test size(d.weight) == (3, 2)
        @test size(d.bias) == (3,)
        @test d.activation == identity

        d = Dense(1, 1, (x) -> x .+ 2)
        @test size(d.weight) == (1, 1)
        @test size(d.bias) == (1,)
        @test d.activation([1 1]) == [3 3]
    end
end

@testset "Forward a tensor into the dense layer" begin
    @testset "Identity activation function" begin
        # 3 --> 2
        d = Dense(Tensor([1 1 1; 2 2 2], true), Tensor([4, 4], true))

        @test_throws DimensionMismatch d([2 2 2])
        @test_throws DimensionMismatch d([2])
        @test_throws DimensionMismatch d([2, 2])
        @test_throws DimensionMismatch d([2 2])

        input = Tensor([10, 10, 10])
        output = d(input)


        @test output.data == [34, 64]

        d2 = Dense(Tensor(1, true), Tensor(2, true))
        input = Tensor(10)
        output = d2(input)

        @test output.data == 12

        d3 = Dense(3, 4)
        input = Tensor([1, 1, 1])
        output = d3(input)

        @test output.data == (d3.weight * input .+ d3.bias).data

    end

    @testset "Other activation function" begin
        f(x) = x .* x
        d = Dense(Tensor([1 1; 2 2], true), Tensor([4, 4], true), f)
        input = Tensor([10, 10])
        output = d(input)

        @test output.data == [24^2, 44^2]
    end

end

@testset "Test zero_grad on a Dense layer" begin
    d = Dense(Tensor(2, true), Tensor(1, true))
    d.weight.gradient = 100
    d.bias.gradient = 200

    @test d.weight.gradient == 100
    @test d.bias.gradient == 200

    zero_grad!(d)

    @test d.weight.gradient == 0
    @test d.bias.gradient == 0

end

@testset "Backward a tensor into the dense layer" begin
    @testset "Identity activation function" begin
        d = Dense(Tensor(2, true), Tensor(1, true))

        input = Tensor(3)
        output = d(input)

        @test output.data == 7

        backward!(output, 10)

        @test d.weight.gradient == 3 * 10
        @test d.bias.gradient == 1 * 10
    end

    @testset "Other activation function" begin
        f(x) = 2 .* x
        d = Dense(Tensor(2, true), Tensor(1, true), f)

        input = Tensor(3)
        output = d(input)

        @test output.data == 14

        backward!(output, 10)

        @test d.weight.gradient == 6 * 10
        @test d.bias.gradient == 2 * 10

    end

end


@testset "Multiple dense layer" begin
    @testset "With first constuctor giving the weights and biases" begin
        d1 = Dense(Tensor([2 2], true), Tensor([1], true))
        d2 = Dense(Tensor(3, true), Tensor(2, true))
        d3 = Dense(Tensor(3, true), Tensor(1, true))

        y = d1([3, 2])
        y = d2(y)
        y = d3(y)

        @test y.data == [106]
    end

    @testset "With second constructor giving the size of input and output" begin
        d1 = Dense(10, 5)
        d2 = Dense(5, 5)
        d3 = Dense(5, 2)
        d4 = Dense(2, 1)

        y = d1([1, 2, 3, 4, -5, 6, 7, 8, 9, 10])
        y = d2(y)
        y = d3(y)
        y = d4(y)
    end

end


@testset "parameters method" begin
    w = Tensor(2, true)
    b = Tensor(1, true)

    d = Dense(w, b)
    params = parameters(d)

    @test params[1] == w
    @test params[2] == b
end;