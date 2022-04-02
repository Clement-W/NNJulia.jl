include("../src/NNJulia.jl")
using .NNJulia
using Test

@testset "Test split_train_test" begin
    x_data = rand(3, 100)
    y_data = rand(1, 100)

    x_train, y_train, x_test, y_test = split_train_test(x_data, y_data, 0.8)

    @test size(x_train) == (3, 80)
    @test size(y_train) == (1, 80)

    @test size(x_test) == (3, 20)
    @test size(y_test) == (1, 20)
end;