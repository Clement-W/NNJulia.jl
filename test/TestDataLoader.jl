@testset "Test DataLoader" begin
    @testset "Constructor" begin

        x = rand(3, 5)
        y = rand(1, 5)

        d = DataLoader(x, y)

        @test d.XData == x
        @test d.YData == y
        @test d.batchSize == 1
        @test d.nbBatch == 5
        @test d.shuffle == false
        @test d.indices == [1, 2, 3, 4, 5]

        d2 = DataLoader(x, y, 2)
        @test d2.XData == x
        @test d2.YData == y
        @test d2.batchSize == 2
        @test d2.nbBatch == 3
        @test d2.shuffle == false
        @test d2.indices == [1, 3, 5]

        d3 = DataLoader(x, y, 3, true)
        @test d3.XData == x
        @test d3.YData == y
        @test d3.batchSize == 3
        @test d3.nbBatch == 2
        @test d3.shuffle == true
        @test d3.indices == [1, 4]

    end

    @testset "Length" begin

        x = rand(3, 5)
        y = rand(1, 5)

        d = DataLoader(x, y, 2)

        @test length(d) == 3

    end


    @testset "Iteration on 1D data" begin
        x = rand(100)
        y = rand(100)

        d = DataLoader(x, y, 2)

        @test d.XData == x
        @test d.YData == y
        @test d.batchSize == 2
        @test d.nbBatch == 50
        @test d.shuffle == false
        @test d.indices == [i for i in range(1, 100, step=2)]

        for data in d
            @test data[1] == [x[1], x[2]]
            @test data[2] == [y[1], y[2]]
            break
        end

    end


    @testset "Shuffle dataset" begin
        x = rand(100)
        y = rand(100)

        d = DataLoader(x, y, 2, true)
        indices = copy(d.indices)
        for data in d
            continue
        end

        @test indices != d.indices

    end

    @testset "Iteration on 2D data" begin
        x = rand(3, 100)
        y = rand(1, 100)

        d = DataLoader(x, y, 2)

        @test d.XData == x
        @test d.YData == y
        @test d.batchSize == 2
        @test d.nbBatch == 50
        @test d.shuffle == false
        @test d.indices == [i for i in range(1, 100, step=2)]

        for data in d

            #test input value
            @test data[1] == [x[:, 1] x[:, 2]]
            #test label value
            @test data[2] == [y[1] y[2]]
            break
        end

    end

    @testset "Iteration on 3D tensors" begin
        x = rand(3, 3, 100)
        y = rand(1, 1, 100)

        d = DataLoader(x, y, 2)

        @test d.XData == x
        @test d.YData == y
        @test d.batchSize == 2
        @test d.nbBatch == 50
        @test d.shuffle == false
        @test d.indices == [i for i in range(1, 100, step=2)]

        for data in d
            @test data[1][:, :, 1] == x[1:3, 1:3, 1]
            @test data[1][:, :, 2] == x[1:3, 1:3, 2]

            @test data[2][:] == y[1:2]
            break
        end

    end


end;
