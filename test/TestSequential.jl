include("../src/NNJulia.jl")
using .NNJulia
using Test


@testset "Constructors" begin

    @testset "Base constructor" begin
        layers = [Dense(2, 5), Dense(5, 2), Dense(2, 1)]
        seq = Sequential(layers)

        @test length(seq.layers) == 3

        seq2 = Sequential([
            Dense(100, 1000),
            Dense(1000, 1000),
            Dense(1000, 500),
            Dense(500, 100),
            Dense(100, 10)
        ])

        @test length(seq2.layers) == 5

    end


    @testset "Constructor taking an arbitrary number of argument" begin

        seq = Sequential(
            Dense(2, 5),
            Dense(5, 2),
            Dense(2, 1)
        )

        @test length(seq.layers) == 3

    end

    @testset "Constructor taking taking no argument" begin

        seq = Sequential()

        @test length(seq.layers) == 0

    end
end

@testset "Forward a tensor into the sequential " begin

    @testset "Sequential having only Dense layers" begin

        seq = Sequential(
            Dense(Tensor([2 2], true), Tensor([1], true)),
            Dense(Tensor(3, true), Tensor(2, true)),
            Dense(Tensor(3, true), Tensor(1, true))
        )

        # forward a vector
        y = seq([3, 2])
        @test y.data == [106]

        # forward a tensor
        y = seq(Tensor([3, 2]))
        @test y.data == [106]

    end

end

@testset "Test zero_grad on a Sequential" begin
    d1 = Dense(Tensor(2, true), Tensor(1, true))
    d1.weight.gradient = 100
    d1.bias.gradient = 200

    d2 = Dense(Tensor(8, true), Tensor(2, true))
    d2.weight.gradient = 99
    d2.bias.gradient = 150

    seq = Sequential(d1, d2)

    @test seq.layers[1].weight.gradient == 100
    @test seq.layers[1].bias.gradient == 200

    @test seq.layers[2].weight.gradient == 99
    @test seq.layers[2].bias.gradient == 150

    zero_grad!(seq)

    @test seq.layers[1].weight.gradient == 0
    @test seq.layers[1].bias.gradient == 0

    @test seq.layers[2].weight.gradient == 0
    @test seq.layers[2].bias.gradient == 0

end

@testset "Backward a tensor into the Sequential" begin
    @testset "The sequential only have dense layers " begin
        @testset "Dense with one neuron in the layers" begin

            seq = Sequential(
                Dense(Tensor(2, true), Tensor(1, true)),
                Dense(Tensor(4, true), Tensor(5, true))
            )


            input = Tensor(3)
            output = seq(input)

            @test output.data == 33

            backward!(output, 1)

            @test seq.layers[2].weight.gradient == 7
            @test seq.layers[2].bias.gradient == 1

            @test seq.layers[1].weight.gradient == 3 * 4
            @test seq.layers[1].bias.gradient == 1 * 4
        end

        @testset "Dense with multiple neurons in the layers w and b" begin

            seq = Sequential(
                Dense(Tensor([2 -1; -3 2], true), Tensor([0, 0], true)),
                Dense(Tensor([4 5], true), Tensor([-1], true))
            )

            input = Tensor([4, 7])
            output = seq(input)

            @test output.data == [13]

            backward!(output, [1])

            @test seq.layers[2].weight.gradient == [1 2]
            @test seq.layers[2].bias.gradient == [1]

            @test seq.layers[1].weight.gradient == [16 28; 20 35]
            @test seq.layers[1].bias.gradient == [4, 5]
        end

    end

end

@testset "Test add! method" begin
    s = Sequential()
    add!(s, Dense(2, 4))

    @test length(s.layers) == 1
end


@testset "Test parameters method" begin
    @testset "Check if every parameters is well returned" begin
        w1 = Tensor(2, true)
        b1 = Tensor(1, true)
        d1 = Dense(w1, b1)

        w2 = Tensor(4, true)
        b2 = Tensor(7, true)
        d2 = Dense(w2, b2)

        s = Sequential(d1, d2)
        params = parameters(s)

        @test length(params) == 4

        @test params[1] == w1
        @test params[2] == b1

        @test params[3] == w2
        @test params[4] == b2
    end

    @testset "Update parameters with their gradient" begin
        w1 = Tensor(2, 100)
        b1 = Tensor(1, 200)
        d1 = Dense(w1, b1)

        w2 = Tensor(4, 300)
        b2 = Tensor(7, 400)
        d2 = Dense(w2, b2)

        s = Sequential(d1, d2)

        params = parameters(s)
        for p in params
            p.data = p.data .+ p.gradient
        end

        @test s.layers[1].weight.data == 102
        @test s.layers[1].bias.data == 201

        @test s.layers[2].weight.data == 304
        @test s.layers[2].bias.data == 407


    end


end;
