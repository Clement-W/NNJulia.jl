using Random

"""
    DataLoader(XData::T1, YData::T2, batchSize=1::Int, shuffle=false::Bool) where {T1<:Union{AbstractArray,Float64,Int64},T2<:Union{AbstractArray,Float64,Int64}}

This struct represent a DataLoader. A DataLoader is used to load a dataset with inputs and label, shuffle the data, separate it into batches, etc.
It is possible to iterate over this struct to access the batches of the dataset.

For the first iteration, the list of indices is shuffled (if required).
Then, it return a subpart of the dataset according to the batch size with this size : (dataSize...,batchSize)

For example, if the input data contains two values for a batch size of 4, this matrix will be returned :

    [
        1 0 1 1
        0 1 0 1
    ]

"""
struct DataLoader{T1<:Union{AbstractArray,Float64,Int64},T2<:Union{AbstractArray,Float64,Int64}}
    XData::T1
    YData::T2
    batchSize::Int
    indices::Vector{Int}
    shuffle::Bool
    nbBatch::Int
    function DataLoader(XData::T1, YData::T2, batchSize=1::Int, shuffle=false::Bool) where {T1<:Union{AbstractArray,Float64,Int64},T2<:Union{AbstractArray,Float64,Int64}}
        size(XData)[ndims(XData)] == size(YData)[ndims(YData)] || throw("xData and yData must have the same number of samples")
        #ndims(XData) == ndims(YData) || throw("XData and YData must have the same number of dimensions")
        batchSize > 0 || throw("BatchSIze must be > 0")

        # Initialise the list of indices according to the batchsize
        indices = range(1, size(XData)[ndims(XData)], step=batchSize)

        # The number of batches is equal to the length of the indices array
        nbBatch = length(indices)
        new{Union{AbstractArray,Float64,Int64},Union{AbstractArray,Float64,Int64}}(XData, YData, batchSize, indices, shuffle, nbBatch)
    end
end

"""
    Base.iterate(d::DataLoader, state=1)

Iterate over the DataLoader. 
For the first iteration, the list of indices is shuffled (if required).
Then, return a subpart of the dataset according to the batch size with this size : (dataSize...,batchSize)

For example, if the input data contains two values for a batch size of 4, this matrix will be returned :

    [
        1 0 1 1
        0 1 0 1
    ]
"""
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


    # the next index is the minimum between current index + batchsize and the number of data samples
    # So the list of indices cannot be accessed with an index superior to it's size
    nextIndex = min((d.indices[state] + d.batchSize) - 1, size(d.XData)[ndims(d.XData)])

    # range of index going from curent index to the nextindex computed before
    index = d.indices[state]:nextIndex

    # The next batch is taken at the range index computed before, and then
    # a tuple of length ndims(data) is created to get every dims of the data ndim array
    # So here, we create a tuple of length ndims(data) that contains ':' to take every data at this dimension
    # but the last dimension corresponds to the index taken in the batch

    # permutedims is used to forward a ndims(data) x batchSize to the layers
    x = (d.XData[ntuple(index -> :, Val(ndims(d.XData) - 1))..., index]) # Val() can be removed, it is used to compute with tuples faster
    y = (d.YData[ntuple(index -> :, Val(ndims(d.YData) - 1))..., index])

    # Put input and labels into a tuple
    batch = (x, y)
    # when iterating, batch is returned and is a tuple containing the input data, and the corresponding label
    return (batch, nextState)

end

"""
    Base.length(d::DataLoader)

The length of a DataLoader is it's number of batches.
"""
Base.length(d::DataLoader) = d.nbBatch