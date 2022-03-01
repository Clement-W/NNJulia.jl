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
    backward!(t, [5, 5, 5])
    @test t.gradient == [5, 5, 5]

end

@testset "Tensor sum" begin
    t1 = Tensor([1, 2, 3])
    t2 = sum(t1)

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
end




@testset "Test + operator" begin
    @testset "Simple binary addition" begin


        t1 = Tensor([1, 2, 3])
        t2 = Tensor([2, 2, 2])
        t3 = t1 + t2

        @test t1.data == [1, 2, 3]
        @test t2.data == [2, 2, 2]
        @test t3.data == [3, 4, 5]
        @test t1.gradient == [0, 0, 0]
        @test t2.gradient == [0, 0, 0]
        @test t3.gradient == [0, 0, 0]
        @test size(t3.dependencies) == (2,)
        @test t1.dependencies === nothing
        @test t2.dependencies === nothing

        backward!(t3, [10, 20, 30])

        @test t3.gradient == [10, 20, 30]
        @test t2.gradient == [10, 20, 30]
        @test t1.gradient == [10, 20, 30]

    end

    @testset "Simple unary addition" begin
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([2, 2, 2])
        t3 = t1 + t2

        backward!(t3, [10, 20, 30])

        @test t3.data == [3, 4, 5]
        @test t3.gradient == [10, 20, 30]

        t3 += Tensor([0.5, 0.5, 0.5])
        @test t3.data == [3.5, 4.5, 5.5]
        @test t3.gradient == [0, 0, 0] # A new tensor has been created and assigned to t3
        @test size(t3.dependencies) == (2,)
    end


    @testset "Test element-wise addition between tensors" begin
        @testset "Broadcast element-wise addition that adds a dimension " begin
            t1 = Tensor([1 2 3; 4 5 6])
            t2 = Tensor(1)
            t3 = t1 .+ t2

            @test t1.data == [1 2 3; 4 5 6]
            @test t2.data == 1
            @test t3.data == [2 3 4; 5 6 7]
            @test t1.gradient == [0 0 0; 0 0 0]
            @test t2.gradient == 0
            @test t3.gradient == [0 0 0; 0 0 0]
            @test size(t3.dependencies) == (2,)
            @test t1.dependencies === nothing
            @test t2.dependencies === nothing
            @test size(t3) == (2, 3)

            backward!(t3, [1 1 1; 2 2 2])


            @test t3.gradient == [1 1 1; 2 2 2]
            @test t2.gradient == 9
            @test t1.gradient == [1 1 1; 2 2 2]
        end

        @testset "Broadcast element-wise addition with no dimensions added " begin
            t1 = Tensor([1 2 3; 4 5 6])
            t2 = Tensor([2 2 2])
            t3 = t1 .+ t2

            @test t1.data == [1 2 3; 4 5 6]
            @test t2.data == [2 2 2]
            @test t3.data == [3 4 5; 6 7 8]
            @test t1.gradient == [0 0 0; 0 0 0]
            @test t2.gradient == [0 0 0]
            @test t3.gradient == [0 0 0; 0 0 0]
            @test size(t3.dependencies) == (2,)
            @test t1.dependencies === nothing
            @test t2.dependencies === nothing
            @test size(t3) == (2, 3)

            backward!(t3, [1 1 1; 2 2 2])


            @test t3.gradient == [1 1 1; 2 2 2]
            @test t2.gradient == [3 3 3]
            @test t1.gradient == [1 1 1; 2 2 2]
        end
    end
end



@testset "Test - operator" begin
    @testset "Simple binary substraction" begin


        t1 = Tensor([1, 2, 3])
        t2 = Tensor([2, 2, 2])
        t3 = t1 - t2

        @test t1.data == [1, 2, 3]
        @test t2.data == [2, 2, 2]
        @test t3.data == [-1, 0, 1]
        @test t1.gradient == [0, 0, 0]
        @test t2.gradient == [0, 0, 0]
        @test t3.gradient == [0, 0, 0]
        @test size(t3.dependencies) == (2,)
        @test t1.dependencies === nothing
        @test t2.dependencies === nothing

        backward!(t3, [10, 20, 30])

        @test t3.gradient == [10, 20, 30]
        @test t2.gradient == [-10, -20, -30]
        @test t1.gradient == [10, 20, 30]

    end

    @testset "Simple unary substraction" begin
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([2, 2, 2])
        t3 = t1 - t2

        backward!(t3, [10, 20, 30])

        @test t3.data == [-1, 0, 1]
        @test t3.gradient == [10, 20, 30]

        t3 -= Tensor([0.5, 0.5, 0.5])
        @test t3.data == [-1.5, -0.5, 0.5]
        @test t3.gradient == [0, 0, 0] # A new tensor has been created and assigned to t3
        @test size(t3.dependencies) == (2,)
    end

    @testset "Negation operation" begin
        t1 = Tensor([1, 2, 3])
        t3 = -t1

        @test t1.data == [1, 2, 3]
        @test t3.data == [-1, -2, -3]
        @test t1.gradient == [0, 0, 0]
        @test t3.gradient == [0, 0, 0]
        @test size(t3.dependencies) == (1,)
        @test t1.dependencies === nothing

        backward!(t3, [10, 20, 30])

        @test t3.gradient == [10, 20, 30]
        @test t1.gradient == [-10, -20, -30]
    end

    @testset "Test element-wise substraction between tensors" begin
        @testset "Broadcast element-wise substraction that adds a dimension " begin
            t1 = Tensor([1 2 3; 4 5 6])
            t2 = Tensor(1)
            t3 = t1 .- t2

            @test t1.data == [1 2 3; 4 5 6]
            @test t2.data == 1
            @test t3.data == [0 1 2; 3 4 5]
            @test t1.gradient == [0 0 0; 0 0 0]
            @test t2.gradient == 0
            @test t3.gradient == [0 0 0; 0 0 0]
            @test size(t3.dependencies) == (2,)
            @test t1.dependencies === nothing
            @test t2.dependencies === nothing
            @test size(t3) == (2, 3)

            backward!(t3, [1 1 1; 2 2 2])


            @test t3.gradient == [1 1 1; 2 2 2]
            @test t2.gradient == -9
            @test t1.gradient == [1 1 1; 2 2 2]
        end

        @testset "Broadcast element-wise substraction with no dimensions added " begin
            t1 = Tensor([1 2 3; 4 5 6])
            t2 = Tensor([2 2 2])
            t3 = t1 .- t2

            @test t1.data == [1 2 3; 4 5 6]
            @test t2.data == [2 2 2]
            @test t3.data == [-1 0 1; 2 3 4]
            @test t1.gradient == [0 0 0; 0 0 0]
            @test t2.gradient == [0 0 0]
            @test t3.gradient == [0 0 0; 0 0 0]
            @test size(t3.dependencies) == (2,)
            @test t1.dependencies === nothing
            @test t2.dependencies === nothing
            @test size(t3) == (2, 3)

            backward!(t3, [1 1 1; 2 2 2])


            @test t3.gradient == [1 1 1; 2 2 2]
            @test t2.gradient == [-3 -3 -3]
            @test t1.gradient == [1 1 1; 2 2 2]
        end
    end
end


@testset "Test * operator" begin
    @testset "Test matrix multiplication" begin
        t1 = Tensor([1 2; 3 4; 5 6])
        t2 = Tensor([10, 20])

        t3 = t1 * t2
        @test size(t3) == (3,)
        @test t3.data == [50, 110, 170]

        backward!(t3, [-1, -2, -3])

        @test t3.gradient == [-1, -2, -3]
        @test t2.gradient == [-22, -28]
        @test t1.gradient == [-10 -20; -20 -40; -30 -60]

    end

    @testset "Test element-wise multiplication" begin
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        t3 = t1 .* t2

        @test size(t3) == (3,)
        @test t3.data == [4, 10, 18]

        backward!(t3, [10, 20, 30])

        @test t3.gradient == [10, 20, 30]
        @test t1.gradient == [40, 100, 180]
        @test t2.gradient == [10, 40, 90]
    end

    @testset "Broadcast element-wise multiplication" begin
        @testset "Broadcast adding a dimension" begin
            t1 = Tensor([1 2 3; 4 5 6])
            t2 = Tensor(2)

            t3 = t1 .* t2
            @test t3.data == [2 4 6; 8 10 12]

            backward!(t3, [1 1 1; 2 2 2])

            @test t1.gradient == [2 2 2; 4 4 4]
            @test t2.gradient == 36 # 9 + 12 +15
        end

        @testset "Broadcast with no dimension added" begin
            t1 = Tensor([1 2 3; 4 5 6])
            t2 = Tensor([7 8 9])

            t3 = t1 .* t2
            @test t3.data == [7 16 27; 28 40 54]

            backward!(t3, [1 1 1; 2 2 2])

            @test t1.gradient == [7 8 9; 14 16 18]
            @test t2.gradient == [9 12 15]
        end
    end

end


@testset "Test ./ operator " begin
    @testset "Element-wise true division" begin
        t1 = Tensor([1 2 3])
        t2 = Tensor([2 2 2])
        t3 = t1 ./ t2

        @test t3.data == [0.5 1 1.5]

        backward!(t3, [10 20 30])

        @test t3.gradient == [10 20 30]
        @test t1.gradient == [5 10 15]
        @test t2.gradient == [-2.5 -10 -22.5]
    end
    @testset "Broadcast element-wise true division" begin
        @testset "Broadcast adding a dimension" begin
            t1 = Tensor([1 2 3; 4 5 6])
            t2 = Tensor(2)

            t3 = t1 ./ t2
            @test t3.data == [0.5 1 1.5; 2 2.5 3]

            backward!(t3, [1 1 1; 2 2 2])

            @test t3.gradient == [1 1 1; 2 2 2]
            @test t1.gradient == [0.5 0.5 0.5; 1 1 1]
            @test t2.gradient == -9
        end

        @testset "Broadcast with no dimension added" begin
            t1 = Tensor([2 4 6; 8 10 12])
            t2 = Tensor([2 2 2])

            t3 = t1 ./ t2
            @test t3.data == [1 2 3; 4 5 6]

            backward!(t3, [1 1 1; 2 2 2])

            @test t1.gradient == [0.5 0.5 0.5; 1 1 1]
            @test t2.gradient == [-4.5 -6 -7.5]
        end
    end
end


@testset "Test log operator " begin
    t1 = Tensor([1 2 3])

    t3 = log(t1)
    @test t3.data == [log(1) log(2) log(3)]

    backward!(t3, [2 2 2])

    @test t1.gradient == [2 1 2 / 3]
end
