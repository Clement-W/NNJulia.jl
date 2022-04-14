include("../src/NNJulia.jl")
using .NNJulia
using Test

@testset "Test zero_grad " begin
    f = Flatten()
    zero_grad!(f)
end

@testset "Test parameters" begin
    f = Flatten()
    @test parameters(f) == []
end

@testset "Forward a tensor into the flatten layer" begin

    f = Flatten()
    t = Tensor(rand(28, 28, 3, 100), true)
    t2 = f(t)

    @test size(t2.data) == (2352, 100)
    @test size(t2.gradient) == (2352, 100)
    @test length(t2.dependencies) == 1

end

@testset "Forward an array into the flatten layer" begin
    f = Flatten()
    x = rand(28, 28, 3, 100)
    y = f(x)
    @test size(y) == (2352, 100)
end

@testset "Backward a layer that passed through a Flatten layer" begin
    f = Flatten()
    t = Tensor(ones(3, 3, 10) .* 2, true)
    t2 = f(t)

    @test size(t2.data) == (9, 10)
    @test size(t2.gradient) == (9, 10)
    @test length(t2.dependencies) == 1

    backward!(t2, ones(9, 10))

    @test t.gradient == ones(3, 3, 10)

end