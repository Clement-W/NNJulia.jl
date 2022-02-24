include("../src/Autodiff.jl")
using .Autodiff
using Test

@testset "Constructors" begin

    @testset "Constructor with one argument" begin
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


    end




end