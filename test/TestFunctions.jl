include("../src/NNJulia.jl")
using .NNJulia
using Test


@testset "Tensor sum" begin
    t1 = Tensor([1, 2, 3], true)
    t2 = sum(t1)

    @test t2.requires_grad == true

    backward!(t2)
    @test t1.data == [1, 2, 3]
    @test t2.data == 6

    @test t1.gradient == [1, 1, 1]
    @test t2.gradient == 1

    zero_grad!(t1)
    zero_grad!(t2)

    backward!(t2, -10)
    @test t1.gradient == [-10, -10, -10]
    @test t2.gradient == -10

    t3 = Tensor([1 2; 3 4])
    t4 = sum(t3)

    @test t3.gradient === nothing
    @test t3.dependencies === nothing
    @test t3.requires_grad == false

    @test t4.data == 10
    @test t4.gradient === nothing
    @test t4.dependencies === nothing
    @test t4.requires_grad == false
end

@testset "Tensor log function" begin
    t1 = Tensor([1, 2, 3], true)
    t2 = log(t1)

    @test t2.requires_grad == true

    backward!(t2, [1, 1, 1])

    @test t1.data == [1, 2, 3]
    @test t2.data == [log(1), log(2), log(3)]

    @test t1.gradient == [1, 0.5, 1 / 3]
    @test t2.gradient == [1, 1, 1]

    zero_grad!(t1)
    zero_grad!(t2)

    backward!(t2, [-10, -10, -10])
    @test t1.gradient ≈ [-10, -5, -10 / 3]
    @test t2.gradient == [-10, -10, -10]


    t3 = Tensor([1 2; 3 4])
    t4 = log(t3)

    @test t3.gradient === nothing
    @test t3.dependencies === nothing
    @test t3.requires_grad == false

    @test t4.data == [log(1) log(2); log(3) log(4)]
    @test t4.gradient === nothing
    @test t4.dependencies === nothing
    @test t4.requires_grad == false
end


@testset "Tensor tanh function" begin
    t1 = Tensor([1 2; 3 4], true)
    t2 = tanh(t1)

    backward!(t2, [1 1; 1 1])

    @test t2.data == [tanh(1) tanh(2); tanh(3) tanh(4)]

    @test t2.gradient == [1 1; 1 1]
    @test t1.gradient == [1-tanh(1)^2 1-tanh(2)^2; 1-tanh(3)^2 1-tanh(4)^2]

end

@testset "Tensor sigmoid function" begin
    t1 = Tensor([1 2; 3 4], true)
    t2 = sigmoid(t1)

    backward!(t2, [1 1; 1 1])

    @test t2.data == [(1/(1+exp(-1))) (1/(1+exp(-2))); (1/(1+exp(-3))) (1/(1+exp(-4)))]

    @test t2.gradient == [1 1; 1 1]
    @test t1.gradient == [(1/(1+exp(-1)))*(1-(1/(1+exp(-1)))) (1/(1+exp(-2)))*(1-(1/(1+exp(-2)))); (1/(1+exp(-3)))*(1-(1/(1+exp(-3)))) (1/(1+exp(-4)))*(1-(1/(1+exp(-4))))]

end;

@testset "Tensor sigmoid function" begin
    t1 = Tensor([1 2; 3 4], true)
    t2 = sigmoid(t1)

    backward!(t2, [1 1; 1 1])

    @test t2.data == [(1/(1+exp(-1))) (1/(1+exp(-2))); (1/(1+exp(-3))) (1/(1+exp(-4)))]

    @test t2.gradient == [1 1; 1 1]
    @test t1.gradient == [(1/(1+exp(-1)))*(1-(1/(1+exp(-1)))) (1/(1+exp(-2)))*(1-(1/(1+exp(-2)))); (1/(1+exp(-3)))*(1-(1/(1+exp(-3)))) (1/(1+exp(-4)))*(1-(1/(1+exp(-4))))]

end;


@testset "Tensor relu function" begin
    t1 = Tensor([1 2; -1 4], true)
    t2 = relu(t1)

    backward!(t2, [10 10; 10 10])

    @test t2.data == [1 2; 0 4]

    @test t2.gradient == [10 10; 10 10]
    @test t1.gradient == [10 10; 0 10]

end;

@testset "Tensor leaky relu function" begin
    t1 = Tensor([1 2; -1 4], true)
    t2 = leakyrelu(t1)

    backward!(t2, [10 10; 10 10])

    @test t2.data == [1 2; -0.01 4]

    @test t2.gradient == [10 10; 10 10]
    @test t1.gradient == [10 10; 0.1 10]

end;