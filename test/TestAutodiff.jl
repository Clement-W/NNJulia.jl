@testset "Test Autodiff package" begin
    @testset "Constructors" begin
        @testset "Constructor taking only data in argument + requires_grad" begin
            t = Tensor(3)
            @test t.data == 3
            @test t.dependencies === nothing
            @test t.requires_grad == false

            t1 = Tensor([2 2; 3 3], true)
            @test t1.data == [2 2; 3 3]
            @test t1.gradient == [0 0; 0 0]
            @test t1.dependencies === nothing
            @test t1.requires_grad == true
        end

        @testset "Base constructor" begin
            dep = [TensorDependency(Tensor([3, 3]), () -> ())]
            t = Tensor([2, 2], [0, 0], dep, true)
            @test t.data == [2, 2]
            @test t.gradient == [0, 0]
            @test t.dependencies == dep
            @test t.requires_grad == true

            @test_throws ErrorException Tensor([2, 2], [0], nothing, true)
            @test_throws ErrorException Tensor([2, 2], [0, 0], nothing, false)
            @test_throws ErrorException Tensor([2, 2], nothing, dep, false)

            @test_throws MethodError Tensor([2, 2], 1, nothing, true)
        end

        @testset "Constructor taking data and gradient in argmuent + requires_grad" begin
            t = Tensor(3, 0)
            @test t.data == 3
            @test t.gradient == 0
            @test t.dependencies === nothing
            @test t.requires_grad == true

            t1 = Tensor([2 2; 3 3], [0 0; 0 0])
            @test t1.data == [2 2; 3 3]
            @test t1.gradient == [0 0; 0 0]
            @test t1.dependencies === nothing
            @test t1.requires_grad == true
        end

        @testset "Constructor taking data and TensorDependency in argument + requires_grad" begin
            dep = [TensorDependency(Tensor([3, 3]), () -> ())]
            t = Tensor([2, 2], dep)
            @test t.data == [2, 2]
            @test t.gradient == [0, 0]
            @test t.dependencies == dep
            @test t.requires_grad == true
        end
    end

    @testset "Set Tensor property" begin
        t = Tensor(3, 1)
        t.data = 3
        @test t.gradient == 0
        @test t.requires_grad == true
        @test t.dependencies === nothing

        t1 = Tensor(3, 1)
        t1.gradient = 5
        @test t1.data == 3
        @test t1.gradient == 5
        @test t1.dependencies === nothing
        @test t1.requires_grad == true

        t1 = Tensor(3, false)
        @test_throws ErrorException t1.gradient = 3
        t1.data = 2
        @test t1.data == 2

        t1 = Tensor(5, true)
        t1.data = 2
        t1.gradient = 10


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
        @test t.requires_grad == true

        t1 = Tensor([3 3 3; 2 2 2], [1 1 1; 5 5 5])
        @test t1.gradient == [1 1 1; 5 5 5]
        zero_grad!(t1)
        @test t1.gradient == [0 0 0; 0 0 0]
        @test t1.requires_grad == true
    end


    @testset "Simple Backward" begin
        t = Tensor([2, 2, 2], true)
        @test t.gradient == [0, 0, 0]
        backward!(t, [5, 5, 5])
        @test t.gradient == [5, 5, 5]

        t1 = Tensor([2 2 2])
        @test_throws ErrorException backward!(t1, [5 5 5])

    end



    @testset "Test + operator" begin
        @testset "Simple binary addition" begin

            @testset "2 tensors requiring gradient" begin
                t1 = Tensor([1, 2, 3], true)
                t2 = Tensor([2, 2, 2], true)
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

            @testset "2 tensors that does not requires gradient" begin

                t4 = Tensor([1, 2, 3])
                t5 = Tensor([2, 2, 2])
                t6 = t4 + t5

                @test t6.data == [3, 4, 5]
                @test t6.requires_grad == false
                @test t6.gradient === nothing
                @test t6.dependencies === nothing
            end

            @testset "1 tensor requiring gradient and 1 that does not" begin
                t1 = Tensor([1, 2, 3], true)
                t2 = Tensor([2, 2, 2])
                t3 = t1 + t2

                @test t1.data == [1, 2, 3]
                @test t2.data == [2, 2, 2]
                @test t3.data == [3, 4, 5]
                @test t1.gradient == [0, 0, 0]
                @test t2.gradient === nothing
                @test t3.gradient == [0, 0, 0]
                @test size(t3.dependencies) == (1,)
                @test t1.dependencies === nothing
                @test t2.dependencies === nothing

                backward!(t3, [10, 20, 30])

                @test t3.gradient == [10, 20, 30]
                @test t1.gradient == [10, 20, 30]
                @test t2.gradient === nothing
            end
        end

        @testset "Simple unary addition" begin
            t1 = Tensor([1, 2, 3], true)
            t2 = Tensor([2, 2, 2], true)
            t3 = t1 + t2

            backward!(t3, [10, 20, 30])

            @test t3.data == [3, 4, 5]
            @test t3.gradient == [10, 20, 30]
            @test size(t3.dependencies) == (2,)

            t3 += Tensor([0.5, 0.5, 0.5])
            @test t3.data == [3.5, 4.5, 5.5]
            @test t3.gradient == [0, 0, 0] # A new tensor has been created and assigned to t3
            @test size(t3.dependencies) == (1,)
            @test t3.dependencies[1].tensorDep.data == [3, 4, 5]
            @test t3.requires_grad == true

            t4 = Tensor([1 1 1])
            @test t4.requires_grad == false
            t4 += Tensor([2 2 2], true)

            @test t4.data == [3 3 3]
            @test t4.gradient == [0 0 0]
            @test size(t4.dependencies) == (1,)
            @test t4.dependencies[1].tensorDep.data == [2 2 2]
            @test t4.requires_grad == true

        end


        @testset "Test element-wise addition between tensors" begin
            @testset "Element-wise addition" begin
                @testset "2 tensors requiring gradient" begin
                    t1 = Tensor([1, 2, 3], true)
                    t2 = Tensor([2, 2, 2], true)
                    t3 = t1 .+ t2

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

                @testset "2 tensors that does not requires gradient" begin

                    t4 = Tensor([1, 2, 3])
                    t5 = Tensor([2, 2, 2])
                    t6 = t4 .+ t5

                    @test t6.data == [3, 4, 5]
                    @test t6.requires_grad == false
                    @test t6.gradient === nothing
                    @test t6.dependencies === nothing
                end

                @testset "1 tensor requiring gradient and 1 that does not" begin
                    t1 = Tensor([1, 2, 3], true)
                    t2 = Tensor([2, 2, 2])
                    t3 = t1 .+ t2

                    @test t1.data == [1, 2, 3]
                    @test t2.data == [2, 2, 2]
                    @test t3.data == [3, 4, 5]
                    @test t1.gradient == [0, 0, 0]
                    @test t2.gradient === nothing
                    @test t3.gradient == [0, 0, 0]
                    @test size(t3.dependencies) == (1,)
                    @test t1.dependencies === nothing
                    @test t2.dependencies === nothing

                    backward!(t3, [10, 20, 30])

                    @test t3.gradient == [10, 20, 30]
                    @test t1.gradient == [10, 20, 30]
                    @test t2.gradient === nothing
                end
            end

            @testset "Broadcast element-wise addition that adds a dimension with a scalar " begin
                t1 = Tensor([1 2 3; 4 5 6], true)
                t2 = Tensor(1, true)
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

            @testset "Broadcast element-wise addition that adds a dimension with a vector " begin
                t1 = Tensor([1 2 3; 4 5 6], true)
                t2 = Tensor([1, 1], true)
                t3 = t1 .+ t2

                @test t1.data == [1 2 3; 4 5 6]
                @test t2.data == [1, 1]
                @test t3.data == [2 3 4; 5 6 7]
                @test t1.gradient == [0 0 0; 0 0 0]
                @test t2.gradient == [0, 0]
                @test t3.gradient == [0 0 0; 0 0 0]
                @test size(t3.dependencies) == (2,)
                @test t1.dependencies === nothing
                @test t2.dependencies === nothing
                @test size(t3) == (2, 3)

                backward!(t3, [1 1 1; 2 2 2])


                @test t3.gradient == [1 1 1; 2 2 2]
                @test t2.gradient == [3, 6]
                @test t1.gradient == [1 1 1; 2 2 2]
            end

            @testset "Broadcast element-wise addition with no dimensions added " begin
                t1 = Tensor([1 2 3; 4 5 6], true)
                t2 = Tensor([2 2 2], true)
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


        @testset "Test element-wise addition between a tensor and a scalar/vector/matrix" begin
            @testset "Element-wise addition" begin
                @testset "t1 = tensors requiring gradient and t2 = an array" begin
                    t1 = Tensor([1, 2, 3], true)
                    t2 = [2, 2, 2]
                    t3 = t1 .+ t2

                    @test t1.data == [1, 2, 3]
                    @test t3.data == [3, 4, 5]
                    @test t1.gradient == [0, 0, 0]
                    @test t3.gradient == [0, 0, 0]
                    @test size(t3.dependencies) == (1,)
                    @test t1.dependencies === nothing

                    backward!(t3, [10, 20, 30])

                    @test t3.gradient == [10, 20, 30]
                    @test t1.gradient == [10, 20, 30]
                end

                @testset "t1 = an array and t2 is a tensor requiring gradient" begin
                    t2 = Tensor([1, 2, 3], true)
                    t1 = [2, 2, 2]
                    t3 = t1 .+ t2

                    @test t2.data == [1, 2, 3]
                    @test t3.data == [3, 4, 5]
                    @test t2.gradient == [0, 0, 0]
                    @test t3.gradient == [0, 0, 0]
                    @test size(t3.dependencies) == (1,)
                    @test t2.dependencies === nothing

                    backward!(t3, [10, 20, 30])

                    @test t3.gradient == [10, 20, 30]
                    @test t2.gradient == [10, 20, 30]
                end
            end


            @testset "Broadcast element-wise addition that adds a dimension, with t2 as a scalar " begin
                t1 = Tensor([1 2 3; 4 5 6], true)
                t2 = 1
                t3 = t1 .+ t2

                @test t1.data == [1 2 3; 4 5 6]

                @test t3.data == [2 3 4; 5 6 7]
                @test t1.gradient == [0 0 0; 0 0 0]

                @test t3.gradient == [0 0 0; 0 0 0]
                @test size(t3.dependencies) == (1,)
                @test t1.dependencies === nothing

                @test size(t3) == (2, 3)

                backward!(t3, [1 1 1; 2 2 2])


                @test t3.gradient == [1 1 1; 2 2 2]

                @test t1.gradient == [1 1 1; 2 2 2]
            end

            @testset "Broadcast element-wise addition with no dimensions added with t2 as a matrix " begin
                t1 = Tensor([1 2 3; 4 5 6], true)
                t2 = [2 2 2]
                t3 = t1 .+ t2

                @test t1.data == [1 2 3; 4 5 6]

                @test t3.data == [3 4 5; 6 7 8]
                @test t1.gradient == [0 0 0; 0 0 0]

                @test t3.gradient == [0 0 0; 0 0 0]
                @test size(t3.dependencies) == (1,)
                @test t1.dependencies === nothing

                @test size(t3) == (2, 3)

                backward!(t3, [1 1 1; 2 2 2])

                @test t3.gradient == [1 1 1; 2 2 2]
                @test t1.gradient == [1 1 1; 2 2 2]
            end
        end

    end


    @testset "Test - operator" begin
        @testset "Simple binary substraction" begin

            @testset "2 tensors requiring gradient" begin
                t1 = Tensor([1, 2, 3], true)
                t2 = Tensor([2, 2, 2], true)
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

            @testset "2 tensors that does not requires gradient" begin

                t4 = Tensor([1, 2, 3])
                t5 = Tensor([2, 2, 2])
                t6 = t4 - t5

                @test t6.data == [-1, 0, 1]
                @test t6.requires_grad == false
                @test t6.gradient === nothing
                @test t6.dependencies === nothing
            end

            @testset "1 tensor requiring gradient and 1 that does not" begin
                t1 = Tensor([1, 2, 3], true)
                t2 = Tensor([2, 2, 2])
                t3 = t1 - t2

                @test t1.data == [1, 2, 3]
                @test t2.data == [2, 2, 2]
                @test t3.data == [-1, 0, 1]
                @test t1.gradient == [0, 0, 0]
                @test t2.gradient === nothing
                @test t3.gradient == [0, 0, 0]
                @test size(t3.dependencies) == (1,)
                @test t1.dependencies === nothing
                @test t2.dependencies === nothing

                backward!(t3, [10, 20, 30])

                @test t3.gradient == [10, 20, 30]
                @test t1.gradient == [10, 20, 30]
                @test t2.gradient === nothing
            end
        end

        @testset "Simple unary substraction" begin
            t1 = Tensor([1, 2, 3], true)
            t2 = Tensor([2, 2, 2], true)
            t3 = t1 - t2

            backward!(t3, [10, 20, 30])

            @test t3.data == [-1, 0, 1]
            @test t3.gradient == [10, 20, 30]
            @test size(t3.dependencies) == (2,)

            t3 -= Tensor([0.5, 0.5, 0.5])
            @test t3.data == [-1.5, -0.5, 0.5]
            @test t3.gradient == [0, 0, 0] # A new tensor has been created and assigned to t3
            @test size(t3.dependencies) == (1,)
            @test t3.dependencies[1].tensorDep.data == [-1, 0, 1]
            @test t3.requires_grad == true

            t4 = Tensor([1 1 1])
            @test t4.requires_grad == false
            t4 -= Tensor([2 2 2], true)

            @test t4.data == [-1 -1 -1]
            @test t4.gradient == [0 0 0]
            @test size(t4.dependencies) == (1,)
            @test t4.dependencies[1].tensorDep.data == [2 2 2]
            @test t4.requires_grad == true

        end

        @testset "Negation operation" begin
            t1 = Tensor([1, 2, 3], true)
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
            @testset "Element-wise substraction" begin
                @testset "2 tensors requiring gradient" begin
                    t1 = Tensor([1, 2, 3], true)
                    t2 = Tensor([2, 2, 2], true)
                    t3 = t1 .- t2

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

                @testset "2 tensors that does not requires gradient" begin

                    t4 = Tensor([1, 2, 3])
                    t5 = Tensor([2, 2, 2])
                    t6 = t4 .- t5

                    @test t6.data == [-1, 0, 1]
                    @test t6.requires_grad == false
                    @test t6.gradient === nothing
                    @test t6.dependencies === nothing
                end

                @testset "1 tensor requiring gradient and 1 that does not" begin
                    t1 = Tensor([1, 2, 3], true)
                    t2 = Tensor([2, 2, 2])
                    t3 = t1 .- t2

                    @test t1.data == [1, 2, 3]
                    @test t2.data == [2, 2, 2]
                    @test t3.data == [-1, 0, 1]
                    @test t1.gradient == [0, 0, 0]
                    @test t2.gradient === nothing
                    @test t3.gradient == [0, 0, 0]
                    @test size(t3.dependencies) == (1,)
                    @test t1.dependencies === nothing
                    @test t2.dependencies === nothing

                    backward!(t3, [10, 20, 30])

                    @test t3.gradient == [10, 20, 30]
                    @test t1.gradient == [10, 20, 30]
                end
            end

            @testset "Broadcast element-wise substraction that adds a dimension " begin
                t1 = Tensor([1 2 3; 4 5 6], true)
                t2 = Tensor(1, true)
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
                t1 = Tensor([1 2 3; 4 5 6], true)
                t2 = Tensor([2 2 2], true)
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


        @testset "Test element-wise substraction between a tensor and a scalar/vector/matrix" begin
            @testset "Element-wise substraction" begin
                @testset "t1 = tensors requiring gradient and t2 = an array" begin
                    t1 = Tensor([1, 2, 3], true)
                    t2 = [2, 2, 2]
                    t3 = t1 .- t2

                    @test t1.data == [1, 2, 3]

                    @test t3.data == [-1, 0, 1]
                    @test t1.gradient == [0, 0, 0]

                    @test t3.gradient == [0, 0, 0]
                    @test size(t3.dependencies) == (1,)
                    @test t1.dependencies === nothing

                    backward!(t3, [10, 20, 30])

                    @test t3.gradient == [10, 20, 30]
                    @test t1.gradient == [10, 20, 30]
                end

                @testset "t1 = an array and t2 is a tensor requiring gradient" begin

                    t2 = Tensor([1, 2, 3], true)
                    t1 = [2, 2, 2]
                    t3 = t1 .- t2

                    @test t2.data == [1, 2, 3]

                    @test t3.data == [1, 0, -1]
                    @test t2.gradient == [0, 0, 0]

                    @test t3.gradient == [0, 0, 0]
                    @test size(t3.dependencies) == (1,)
                    @test t2.dependencies === nothing

                    backward!(t3, [10, 20, 30])

                    @test t3.gradient == [10, 20, 30]
                    @test t2.gradient == [-10, -20, -30]
                end

            end

            @testset "Broadcast element-wise substraction that adds a dimension with t2 as a scalar " begin
                t1 = Tensor([1 2 3; 4 5 6], true)
                t2 = 1
                t3 = t1 .- t2

                @test t1.data == [1 2 3; 4 5 6]

                @test t3.data == [0 1 2; 3 4 5]
                @test t1.gradient == [0 0 0; 0 0 0]

                @test t3.gradient == [0 0 0; 0 0 0]
                @test size(t3.dependencies) == (1,)
                @test t1.dependencies === nothing

                @test size(t3) == (2, 3)

                backward!(t3, [1 1 1; 2 2 2])

                @test t3.gradient == [1 1 1; 2 2 2]
                @test t1.gradient == [1 1 1; 2 2 2]
            end

            @testset "Broadcast element-wise substraction with no dimensions added with t2 a matrix " begin
                t1 = Tensor([1 2 3; 4 5 6], true)
                t2 = [2 2 2]
                t3 = t1 .- t2

                @test t1.data == [1 2 3; 4 5 6]

                @test t3.data == [-1 0 1; 2 3 4]
                @test t1.gradient == [0 0 0; 0 0 0]

                @test t3.gradient == [0 0 0; 0 0 0]
                @test size(t3.dependencies) == (1,)
                @test t1.dependencies === nothing

                @test size(t3) == (2, 3)

                backward!(t3, [1 1 1; 2 2 2])

                @test t3.gradient == [1 1 1; 2 2 2]

                @test t1.gradient == [1 1 1; 2 2 2]
            end
        end
    end


    @testset "Test * operator" begin

        @testset "Simple matrix multiplication" begin

            @testset "2 tensors requiring gradient" begin
                t1 = Tensor([1 2; 3 4; 5 6], true)
                t2 = Tensor([10, 20], true)

                t3 = t1 * t2
                @test size(t3) == (3,)
                @test t1.data == [1 2; 3 4; 5 6]
                @test t2.data == [10, 20]
                @test t3.data == [50, 110, 170]
                @test t1.gradient == [0 0; 0 0; 0 0]
                @test t2.gradient == [0, 0]
                @test t3.gradient == [0, 0, 0]
                @test size(t3.dependencies) == (2,)
                @test t1.dependencies === nothing
                @test t2.dependencies === nothing

                backward!(t3, [-1, -2, -3])

                @test t3.gradient == [-1, -2, -3]
                @test t2.gradient == [-22, -28]
                @test t1.gradient == [-10 -20; -20 -40; -30 -60]
            end

            @testset "2 tensors that does not requires gradient" begin

                t1 = Tensor([1 2; 3 4; 5 6])
                t2 = Tensor([10, 20])

                t3 = t1 * t2

                @test t3.data == [50, 110, 170]
                @test t3.requires_grad == false
                @test t3.gradient === nothing
                @test t3.dependencies === nothing
            end

            @testset "1 tensor requiring gradient and 1 that does not" begin
                t1 = Tensor([1 2; 3 4; 5 6], true)
                t2 = Tensor([10, 20])
                t3 = t1 * t2

                @test t1.data == [1 2; 3 4; 5 6]
                @test t2.data == [10, 20]
                @test t3.data == [50, 110, 170]
                @test t1.gradient == [0 0; 0 0; 0 0]
                @test t2.gradient === nothing
                @test t3.gradient == [0, 0, 0]
                @test size(t3.dependencies) == (1,)
                @test t1.dependencies === nothing
                @test t2.dependencies === nothing

                backward!(t3, [-1, -2, -3])

                @test t3.gradient == [-1, -2, -3]
                @test t2.gradient === nothing
                @test t1.gradient == [-10 -20; -20 -40; -30 -60]

            end
        end


        @testset "Matrix multiplication where the result is a scalar" begin
            t1 = Tensor([1 2 3], true)
            t2 = Tensor([2, 2, 2], true)
            t3 = t1 * t2


            @test size(t3) == (1,)
            @test t1.data == [1 2 3]
            @test t2.data == [2, 2, 2]
            @test t3.data == [12]
            @test t1.gradient == [0 0 0]
            @test t2.gradient == [0, 0, 0]
            @test t3.gradient == [0]
            @test size(t3.dependencies) == (2,)
            @test t1.dependencies === nothing
            @test t2.dependencies === nothing

            backward!(t3, [10])

            @test t3.gradient == [10]
            @test t2.gradient == [10, 20, 30]
            @test t1.gradient == [20 20 20]
        end


        @testset "Simple unary multiplication" begin
            t1 = Tensor([1 2; 3 4; 5 6])

            t1 *= Tensor([10, 20], true)


            @test t1.data == [50, 110, 170]
            @test t1.gradient == [0, 0, 0]
            @test size(t1.dependencies) == (1,)
            @test t1.dependencies[1].tensorDep.data == [10, 20]
            @test t1.requires_grad == true

        end

        @testset "Test element-wise multiplication between tensors" begin
            @testset "Element-wise multiplication" begin
                @testset "2 tensors requiring gradient" begin
                    t1 = Tensor([1, 2, 3], true)
                    t2 = Tensor([2, 2, 2], true)
                    t3 = t1 .* t2

                    @test t1.data == [1, 2, 3]
                    @test t2.data == [2, 2, 2]
                    @test t3.data == [2, 4, 6]
                    @test t1.gradient == [0, 0, 0]
                    @test t2.gradient == [0, 0, 0]
                    @test t3.gradient == [0, 0, 0]
                    @test size(t3.dependencies) == (2,)
                    @test t1.dependencies === nothing
                    @test t2.dependencies === nothing

                    backward!(t3, [10, 20, 30])

                    @test t3.gradient == [10, 20, 30]
                    @test t2.gradient == [10, 40, 90]
                    @test t1.gradient == [20, 40, 60]
                end

                @testset "2 tensors that does not requires gradient" begin

                    t4 = Tensor([1, 2, 3])
                    t5 = Tensor([2, 2, 2])
                    t6 = t4 .* t5

                    @test t6.data == [2, 4, 6]
                    @test t6.requires_grad == false
                    @test t6.gradient === nothing
                    @test t6.dependencies === nothing
                end

                @testset "1 tensor requiring gradient and 1 that does not" begin
                    t1 = Tensor([1, 2, 3], true)
                    t2 = Tensor([2, 2, 2])
                    t3 = t1 .* t2

                    @test t1.data == [1, 2, 3]
                    @test t2.data == [2, 2, 2]
                    @test t3.data == [2, 4, 6]
                    @test t1.gradient == [0, 0, 0]
                    @test t2.gradient === nothing
                    @test t3.gradient == [0, 0, 0]
                    @test size(t3.dependencies) == (1,)
                    @test t1.dependencies === nothing
                    @test t2.dependencies === nothing

                    backward!(t3, [10, 20, 30])

                    @test t3.gradient == [10, 20, 30]
                    @test t1.gradient == [20, 40, 60]
                    @test t2.gradient === nothing
                end
            end

            @testset "Broadcast element-wise multiplication that adds a dimension " begin
                t1 = Tensor([1 2 3; 4 5 6], true)
                t2 = Tensor(2, true)
                t3 = t1 .* t2

                @test t1.data == [1 2 3; 4 5 6]
                @test t2.data == 2
                @test t3.data == [2 4 6; 8 10 12]
                @test t1.gradient == [0 0 0; 0 0 0]
                @test t2.gradient == 0
                @test t3.gradient == [0 0 0; 0 0 0]
                @test size(t3.dependencies) == (2,)
                @test t1.dependencies === nothing
                @test t2.dependencies === nothing
                @test size(t3) == (2, 3)

                backward!(t3, [1 1 1; 2 2 2])


                @test t3.gradient == [1 1 1; 2 2 2]
                @test t2.gradient == 36
                @test t1.gradient == [2 2 2; 4 4 4]
            end

            @testset "Broadcast element-wise multiplication with no dimensions added " begin
                t1 = Tensor([1 2 3; 4 5 6], true)
                t2 = Tensor([2 2 2], true)
                t3 = t1 .* t2

                @test t1.data == [1 2 3; 4 5 6]
                @test t2.data == [2 2 2]
                @test t3.data == [2 4 6; 8 10 12]
                @test t1.gradient == [0 0 0; 0 0 0]
                @test t2.gradient == [0 0 0]
                @test t3.gradient == [0 0 0; 0 0 0]
                @test size(t3.dependencies) == (2,)
                @test t1.dependencies === nothing
                @test t2.dependencies === nothing
                @test size(t3) == (2, 3)

                backward!(t3, [1 1 1; 2 2 2])


                @test t3.gradient == [1 1 1; 2 2 2]
                @test t2.gradient == [9 12 15]
                @test t1.gradient == [2 2 2; 4 4 4]
            end
        end


        @testset "Test element-wise multiplication between a tensor and a scalar/vector/matrix" begin
            @testset "Element-wise multiplication" begin
                @testset "t1 = tensors requiring gradient and t2 = an array" begin
                    t1 = Tensor([1, 2, 3], true)
                    t2 = [2, 2, 2]
                    t3 = t1 .* t2

                    @test t1.data == [1, 2, 3]

                    @test t3.data == [2, 4, 6]
                    @test t1.gradient == [0, 0, 0]

                    @test t3.gradient == [0, 0, 0]
                    @test size(t3.dependencies) == (1,)
                    @test t1.dependencies === nothing


                    backward!(t3, [10, 20, 30])

                    @test t3.gradient == [10, 20, 30]
                    @test t1.gradient == [20, 40, 60]
                end

                @testset "t1 = an array and t2 is a tensor requiring gradient" begin

                    t2 = Tensor([1, 2, 3], true)
                    t1 = [2, 2, 2]
                    t3 = t1 .* t2

                    @test t2.data == [1, 2, 3]

                    @test t3.data == [2, 4, 6]
                    @test t2.gradient == [0, 0, 0]

                    @test t3.gradient == [0, 0, 0]
                    @test size(t3.dependencies) == (1,)
                    @test t2.dependencies === nothing


                    backward!(t3, [10, 20, 30])

                    @test t3.gradient == [10, 20, 30]
                    @test t2.gradient == [20, 40, 60]
                end

            end

            @testset "Broadcast element-wise multiplication that adds a dimension with t2 as a scalar" begin
                t1 = Tensor([1 2 3; 4 5 6], true)
                t2 = 2
                t3 = t1 .* t2

                @test t1.data == [1 2 3; 4 5 6]

                @test t3.data == [2 4 6; 8 10 12]
                @test t1.gradient == [0 0 0; 0 0 0]

                @test t3.gradient == [0 0 0; 0 0 0]
                @test size(t3.dependencies) == (1,)
                @test t1.dependencies === nothing

                @test size(t3) == (2, 3)

                backward!(t3, [1 1 1; 2 2 2])


                @test t3.gradient == [1 1 1; 2 2 2]
                @test t1.gradient == [2 2 2; 4 4 4]
            end

            @testset "Broadcast element-wise multiplication with no dimensions added with t2 as a matrix " begin
                t1 = Tensor([1 2 3; 4 5 6], true)
                t2 = [2 2 2]
                t3 = t1 .* t2

                @test t1.data == [1 2 3; 4 5 6]

                @test t3.data == [2 4 6; 8 10 12]
                @test t1.gradient == [0 0 0; 0 0 0]

                @test t3.gradient == [0 0 0; 0 0 0]
                @test size(t3.dependencies) == (1,)
                @test t1.dependencies === nothing

                @test size(t3) == (2, 3)

                backward!(t3, [1 1 1; 2 2 2])


                @test t3.gradient == [1 1 1; 2 2 2]

                @test t1.gradient == [2 2 2; 4 4 4]
            end
        end

    end

    @testset "Test ./ operator" begin
        @testset "Element-wise true division between 2 tensors" begin
            @testset "2 tensors requiring gradient" begin

                t1 = Tensor([1 2 3], true)
                t2 = Tensor([2 2 2], true)
                t3 = t1 ./ t2

                @test t1.data == [1 2 3]
                @test t2.data == [2 2 2]
                @test t3.data == [0.5 1.0 1.5]
                @test t1.gradient == [0 0 0]
                @test t2.gradient == [0 0 0]
                @test t3.gradient == [0.0 0.0 0.0]
                @test size(t3.dependencies) == (2,)
                @test t1.dependencies === nothing
                @test t2.dependencies === nothing

                backward!(t3, [10 20 30])

                @test t3.gradient == [10.0 20.0 30.0]
                @test t1.gradient == [5.0 10.0 15.0]
                @test t2.gradient == [-2.5 -10.0 -22.5]
            end

            @testset "2 tensors that does not requires gradient" begin

                t4 = Tensor([1, 2, 3])
                t5 = Tensor([2, 2, 2])
                t6 = t4 ./ t5

                @test t6.data == [0.5, 1.0, 1.5]
                @test t6.requires_grad == false
                @test t6.gradient === nothing
                @test t6.dependencies === nothing
            end

            @testset "1 tensor requiring gradient and 1 that does not" begin
                t1 = Tensor([1, 2, 3], true)
                t2 = Tensor([2, 2, 2])
                t3 = t1 ./ t2

                @test t1.data == [1, 2, 3]
                @test t2.data == [2, 2, 2]
                @test t3.data == [0.5, 1, 1.5]
                @test t1.gradient == [0, 0, 0]
                @test t2.gradient === nothing
                @test t3.gradient == [0.0, 0.0, 0.0]
                @test size(t3.dependencies) == (1,)
                @test t1.dependencies === nothing
                @test t2.dependencies === nothing

                backward!(t3, [10, 20, 30])

                @test t3.gradient == [10, 20, 30]
                @test t1.gradient == [5, 10, 15]
                @test t2.gradient === nothing

            end
        end

        @testset "Broadcast element-wise true division that adds a dimension " begin
            t1 = Tensor([1 2 3; 4 5 6], true)
            t2 = Tensor(2, true)

            t3 = t1 ./ t2

            @test t1.data == [1 2 3; 4 5 6]
            @test t2.data == 2
            @test t3.data == [0.5 1 1.5; 2 2.5 3]
            @test t1.gradient == [0 0 0; 0 0 0]
            @test t2.gradient == 0
            @test t3.gradient == [0 0 0; 0 0 0]
            @test size(t3.dependencies) == (2,)
            @test t1.dependencies === nothing
            @test t2.dependencies === nothing
            @test size(t3) == (2, 3)

            backward!(t3, [1 1 1; 2 2 2])

            @test t3.gradient == [1 1 1; 2 2 2]
            @test t1.gradient == [0.5 0.5 0.5; 1 1 1]
            @test t2.gradient == -9
        end

        @testset "Broadcast element-wise true division with no dimensions added " begin
            t1 = Tensor([2 4 6; 8 10 12], true)
            t2 = Tensor([2 2 2], true)

            t3 = t1 ./ t2

            @test t1.data == [2 4 6; 8 10 12]
            @test t2.data == [2 2 2]
            @test t3.data == [1 2 3; 4 5 6]
            @test t1.gradient == [0 0 0; 0 0 0]
            @test t2.gradient == [0 0 0]
            @test t3.gradient == [0 0 0; 0 0 0]
            @test size(t3.dependencies) == (2,)
            @test t1.dependencies === nothing
            @test t2.dependencies === nothing
            @test size(t3) == (2, 3)

            backward!(t3, [1 1 1; 2 2 2])

            @test t1.gradient == [0.5 0.5 0.5; 1 1 1]
            @test t2.gradient == [-4.5 -6 -7.5]
        end



        @testset "Element-wise true division between a tensor and a scalar/vector/matrix" begin
            @testset "t1 = tensors requiring gradient and t2 = an array" begin

                t1 = Tensor([1 2 3], true)
                t2 = [2 2 2]
                t3 = t1 ./ t2

                @test t1.data == [1 2 3]

                @test t3.data == [0.5 1.0 1.5]
                @test t1.gradient == [0 0 0]

                @test t3.gradient == [0.0 0.0 0.0]
                @test size(t3.dependencies) == (1,)
                @test t1.dependencies === nothing

                backward!(t3, [10 20 30])

                @test t3.gradient == [10.0 20.0 30.0]
                @test t1.gradient == [5.0 10.0 15.0]
            end

            @testset "t1 = an array and t2 is a tensor requiring gradient" begin

                t1 = [1 2 3]
                t2 = Tensor([2 2 2], true)
                t3 = t1 ./ t2


                @test t2.data == [2 2 2]
                @test t3.data == [0.5 1.0 1.5]

                @test t2.gradient == [0 0 0]
                @test t3.gradient == [0.0 0.0 0.0]
                @test size(t3.dependencies) == (1,)
                @test t2.dependencies === nothing

                backward!(t3, [10 20 30])

                @test t3.gradient == [10.0 20.0 30.0]
                @test t2.gradient == [-2.5 -10.0 -22.5]
            end
        end


        @testset "Broadcast element-wise true division that adds a dimension with t2 as a scalar " begin
            t1 = Tensor([1 2 3; 4 5 6], true)
            t2 = 2

            t3 = t1 ./ t2

            @test t1.data == [1 2 3; 4 5 6]

            @test t3.data == [0.5 1 1.5; 2 2.5 3]
            @test t1.gradient == [0 0 0; 0 0 0]

            @test t3.gradient == [0 0 0; 0 0 0]
            @test size(t3.dependencies) == (1,)
            @test t1.dependencies === nothing

            @test size(t3) == (2, 3)

            backward!(t3, [1 1 1; 2 2 2])

            @test t3.gradient == [1 1 1; 2 2 2]
            @test t1.gradient == [0.5 0.5 0.5; 1 1 1]
        end

        @testset "Broadcast element-wise true division with no dimensions added with t2 as a matrix " begin
            t1 = Tensor([2 4 6; 8 10 12], true)
            t2 = [2 2 2]

            t3 = t1 ./ t2

            @test t1.data == [2 4 6; 8 10 12]

            @test t3.data == [1 2 3; 4 5 6]
            @test t1.gradient == [0 0 0; 0 0 0]

            @test t3.gradient == [0 0 0; 0 0 0]
            @test size(t3.dependencies) == (1,)
            @test t1.dependencies === nothing
            @test size(t3) == (2, 3)

            backward!(t3, [1 1 1; 2 2 2])

            @test t1.gradient == [0.5 0.5 0.5; 1 1 1]

        end


    end

end;
