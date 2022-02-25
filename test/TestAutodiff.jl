include("../src/Autodiff.jl")
using .Autodiff
using Test

@testset "Constructors" begin
    @testset "Constructor taking only data in argument" begin
        t = Tensor(3)
        @test t.data == 3
        @test t.gradient == 0
        @test t.dependencies === nothing

        t1 = Tensor([2 2; 3 3])
        @test t1.data == [2 2; 3 3]
        @test t1.gradient == [0 0; 0 0]
        @test t1.dependencies === nothing
    end

    @testset "Base constructor" begin
        dep = [TensorDependency(Tensor([3, 3]), () -> ())]
        t = Tensor([2, 2], [0, 0], dep)
        @test t.data == [2, 2]
        @test t.gradient == [0, 0]
        @test t.dependencies == dep

        @test_throws ErrorException Tensor([2, 2], [0], nothing)
        @test_throws MethodError Tensor([2, 2], 1, nothing)
    end

    @testset "Constructor taking data and gradient in argmuent" begin
        t = Tensor(3, 0)
        @test t.data == 3
        @test t.gradient == 0
        @test t.dependencies === nothing

        t1 = Tensor([2 2; 3 3], [0 0; 0 0])
        @test t1.data == [2 2; 3 3]
        @test t1.gradient == [0 0; 0 0]
        @test t1.dependencies === nothing
    end

    @testset "Constructor raking data and TensorDependency in argument" begin
        dep = [TensorDependency(Tensor([3, 3]), () -> ())]
        t = Tensor([2, 2], dep)
        @test t.data == [2, 2]
        @test t.gradient == [0, 0]
        @test t.dependencies == dep
    end
end

@testset "Set Tensor property" begin
    t = Tensor(3, 1)
    t.data = 3
    @test t.gradient == 0

    t1 = Tensor(3, 1)
    t1.gradient = 5
    @test t1.data == 3
    @test t1.gradient == 5
    @test t1.dependencies === nothing
end

@testset "Tensor size" begin
    t = Tensor(3, 1)
    @test size(t) == ()

    t1 = Tensor([3 3 3; 2 2 2])
    @test size(t1) == (2, 3)
end

@testset "Tensor ndims" begin
    t = Tensor(rand(3, 3, 3))
    @test ndims(t) == 3

    t1 = Tensor(3)
    @test ndims(t1) == 0
end

@testset "Test zero_grad" begin
    t = Tensor(3, 1)
    @test t.gradient == 1
    zero_grad!(t)
    @test t.gradient == 0

    t1 = Tensor([3 3 3; 2 2 2], [1 1 1; 5 5 5])
    @test t1.gradient == [1 1 1; 5 5 5]
    zero_grad!(t1)
    @test t1.gradient == [0 0 0; 0 0 0]
end

@testset "Simple Backward" begin
    t = Tensor([2, 2, 2])
    @test t.gradient == [0, 0, 0]
    backward(t, [5, 5, 5])
    @test t.gradient == [5, 5, 5]

end

@testset "Tensor sum" begin
    t1 = Tensor([1, 2, 3])
    t2 = sum(t1)

    backward(t2)
    @test t1.data == [1, 2, 3]
    @test t2.data == 6

    @test t1.gradient == [1, 1, 1]
    @test t2.gradient == 1

    zero_grad!(t1)
    zero_grad!(t2)

    backward(t2, -10)
    @test t1.gradient == [-10, -10, -10]
    @test t2.gradient == -10
end