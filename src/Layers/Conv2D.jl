using .Autodiff


# If image size is (32, 32, 3), the input shape becomes (32, 32, 3, BatchSize).
# After the convolution with a valid padding and a stride of 1, the output shape
# is (29, 29, 32, BatchSize)

# image size will be : (n_rows,n_cols,colorChannel,BatchSize)

"""
    Conv2D(filters::Tensor, b::Tensor, activation::Function=identity, padding::Int64=0, stride::Int64=1)

This struct represents a Convolution layer.

# Fields
- filters: The weights of a convolution layer are stored into filters that are convolved with the input.
The size of the filters is ((n_rows,n_cols,colorChannel,n_filters))
- bias: The biases with size (n_filters,)
- activation: The activation function
- padding: The number of 0 added on each side of image
- stride : The stride 

"""
struct Conv2D{T<:Tensor} <: AbstractLayer
    filters::T
    bias::T
    activation::Function
    padding::Int64
    stride::Int64

    function Conv2D(filters::Tensor, b::Tensor, activation::Function=identity, padding::Int64=0, stride::Int64=1)
        # Check that the size of the fiters are correct
        size(filters)[1] == size(filters)[2] || throw(ErrorException("The filters must have the same number of rows and cols"))

        # Check that the parameters requires gradient computation
        (filters.requires_grad && b.requires_grad) || throw(ErrorException("The filters and biases tensors must have requires_grad at true to compute the gradients"))

        # check  padding >= 0
        padding >= 0 || throw("Padding must be >= 0")
        # check stride is >=1
        stride >= 1 || throw("Stride must be >= 1")

        new{Tensor}(filters, b, activation, padding, stride)
    end

end


# alternative contrsuctor
function Conv2D(n_filters::Int64, filter_size::Int64, padding::Int64=0, stride::Int64=1, n_colorChannel::Int64=1, activ::Function=identity)
    # Initialise filters and biases 
    filters = Tensor(rand(filter_size, filter_size, n_colorChannel, n_filters) * 2 .- 1, true)
    bias = Tensor(rand(n_filters) * 2 .- 1, true)
    Conv2D(filters, bias, activ, padding, stride)
end



"""
(conv::Conv2D)(x::AbstractArray)
    
2D Convolution operator for tensors.

x size is (n_rows,n_cols,colorChannel,BatchSize)
"""
#TODO: restart everything with im2col : https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster 
function (conv::Conv2D)(input::AbstractArray)

    # if x has 3 dimensions, it's size is (n_rows,n_cols,BatchSize)
    # so we add a fourth dimension equal to 1 to have a color channel dimension at 1
    # this will help to generalise the next steps


    if (ndims(input) == 3)
        input = reshape(input, (size(input)[1], size(input)[2], 1, size(input)[3]))
    end

    # control the size of x
    ndims(input) == 4 || throw("The dimension of x must be (n_rows,n_cols,colorChannel,BatchSize)")

    # Initialise output 
    nrows_x, ncols_x, nColorChannel_x, batchsize_x = size(input)
    nrows_f, ncols_f, nColorChannel_f, n_f = size(conv.filters)

    nrows_out = round(Int, 1 + (nrows_x + 2 * conv.padding - nrows_f) / conv.stride)
    ncols_out = round(Int, 1 + (ncols_x + 2 * conv.padding - ncols_f) / conv.stride)

    nColorChannel_f == nColorChannel_x || throw("The input data and the filters must have the same number of color channels")

    out = zeros((nrows_out, ncols_out, n_f, batchsize_x))

    # add padding to x if necessary
    if (conv.padding != 0)
        input = padding(input, conv.padding)
    end

    # iterate over examples in the batch 
    for n in 1:batchsize_x
        # iterate over the filters
        for f in 1:n_f
            stride_h = 1
            stride_w = 1

            # for each point in the output
            for h_out in 1:nrows_out
                for w_out in 1:ncols_out
                    # for each point in the filter
                    for k in 1:nrows_f
                        for l in 1:ncols_f
                            # take the depth (color channel) into account
                            for c in 1:nColorChannel_f #TODO: test the performances if this for loop is put before "for k"
                                # compute the value in the output ndimarray
                                # here we sum the product between the filter's value and the input's value
                                # for every value in the filter and every color channel
                                out[h_out, w_out, f, n] += input[stride_h+k-1, stride_w+l-1, c, n] * conv.filters.data[k, l, c, f]
                            end
                        end
                    end

                    out[h_out, w_out, f, n] += conv.bias.data[f]

                    stride_w += 1
                end
                stride_h += 1
                stride_w = 1 # restart at 1 after completing a loop over the columns

            end
        end
    end

    out = conv.activation.(out)


    function gradFunctionFilters(incomingGradient::AbstractArray)
        # https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c

        # comment prendre en compte le padding ? on le fait avant sur x ?
        #dw = conv(x,incGrad)


    end

    function gradFunctionBiases(incomingGradient::AbstractArray)
        # the gradient is 1 * incoming gradient
        # but the gradient needs to have the same shape as the bias vector. So the data
        # is summed on every idexes except the last one that is the index of the filter.
        return dropdims(sum(r, dims=(1, 2, 3)), dims=(1, 2, 3))
    end

    dep1 = TensorDependency(conv.filters, gradFunctionFilters)
    dep2 = TensorDependency(conv.bias, gradFunctionBiases)

    outTensor = Tensor(out, TensorDependency[dep1, dep2])


    # provisoire le temps de tester si la convolution fonctionne
    return outTensor


end


"""
    padding(x::AbstractArray,pad::Int64) 

Pad is added on each side of x.
x size is (n_rows,n_cols,colorChannel,BatchSize).
"""
function padding(x::AbstractArray, pad::Int64)
    # create a new array with the same size of the original x, except that the width and the height is incremented by 2* pad 
    # because there is n=pad times 0 added on each side of the matrices
    xpadded = zeros(size(x)[1] + 2 * pad, size(x)[2] + 2 * pad, size(x)[3], size(x)[4])

    # iterate over examples in xpadded
    for bs in 1:size(xpadded)[4]
        # iterate over the color channels
        for c in 1:size(xpadded)[3]
            # iterate over the rows
            for i in 1:size(x)[1]
                # iterate over the columns
                for j in 1:size(x)[2]
                    xpadded[pad+i, pad+j, c, bs] = x[i, j, c, bs]
                end
            end
        end
    end

    return xpadded
end

