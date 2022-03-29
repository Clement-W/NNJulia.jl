# define constructors of DataLoader,Base.iterate and length
using Random

struct DataLoader{T<:Union{AbstractArray,Float64,Int64}}
    XData::T
    YData::T
    batchSize::Int
    indices::Vector{Int}
    shuffle::Bool
    nbBatch::Int
    function DataLoader(XData::T, YData::T, batchSize=1::Int, shuffle=false::Bool) where {T<:Union{AbstractArray,Float64,Int64}}
        size(XData)[ndims(XData)] == size(YData)[ndims(YData)] || throw("xData and yData must have the same number of samples")
        ndims(XData) == ndims(YData) || throw("XData and YData must have the same number of dimensions")
        batchSize > 0 || throw("BatchSIze must be > 0")

        indices = range(1, size(XData)[ndims(XData)], step=batchSize)

        nbBatch = length(indices)
        new{Union{AbstractArray,Float64,Int64}}(XData, YData, batchSize, indices, shuffle, nbBatch)
    end
end

function Base.iterate(d::DataLoader, state=1)
    if (state > d.nbBatch)
        # if the number of iteration comes to the number of batches, stop iterating
        return nothing
    end

    if (d.shuffle && state == 1)
        # if it is the first iteration and shuffle is needed, shuffle the indices to access the data
        shuffle!(d.indices)
    end

    nextState = state + 1

    # second part of the slice operator  (next index)
    # the next index is the minimum between current index + batchsize and the number of data samples

    nextIndex = min((d.indices[state] + d.batchSize) - 1, size(d.XData)[1])

    # range of index going from curent index to the next index computed before
    index = d.indices[state]:nextIndex

    # The next batch is taken at the range index computed before, and then
    # a tuple of length ndims(data) is created to get every dims of the data ndim array
    # Val(xxxx) can be removed, it is juste used to compute ntuple faster

    # So here, we create a tuple of length ndims(data) that contains ':' to take every data at this dimension
    # but the last dimension corresponds to the index taken in the batch

    # permutedims is used to forward a ndims(data) x batchSize to the layers
    #TODO: make permutedims works for > 2 dim arrays
    x = (d.XData[ntuple(index -> :, Val(ndims(d.XData) - 1))..., index])
    y = (d.YData[ntuple(index -> :, Val(ndims(d.YData) - 1))..., index])
    batch = (x, y)
    # when iterating, batch is returned and is a tuple containing the input data, and the corresponding label
    return (batch, nextState)

end

Base.length(d::DataLoader) = d.nbBatch

# x[i,ntuple(i -> :, (ndims(x) - 1))...]
