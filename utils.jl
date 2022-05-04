## Plot the Euclidean error between the two trajectories
function plotTrajectoryError(x, y)

    figure("Pointwise Trajectory Error")
    plot(sqrt.(abs2.(y[1, :]) .+ abs2.(y[2, :])) - sqrt.(abs2.(x[1, :]) + abs2.(x[2, :])))

    xlabel("Sample Index")
    ylabel("Euclidean Distance between Nominal and Actual Positions")

end

## Show reconstructed image magnitude and phase including normalization if specified
function showReconstructedImage(figID, x, sh, do_normalization)

    fig = figure("$figID: Reconstruction", figsize = (10, 4))

    x_max = maximum(abs.(x))

    reshapedMag = reshape(abs.(x), sh[1], sh[2])
    reshapedAngle = reshape(angle.(x), sh[1], sh[2])

    ## Normalize step:
    if do_normalization
        reshapedMag = reshapedMag ./ x_max
        x_max = 1.0
    end

    subplot(121)
    title("Magnitude")
    imshow(reshapedMag, vmin = 0, vmax = x_max, cmap = "gray")
    colorbar()

    subplot(122)
    title("Phase")
    imshow(ROMEO.unwrap(reshapedAngle, dims=(1,2)), vmin = -pi, vmax = pi, cmap = "seismic")
    colorbar()


end

## Function for plotting the voxel-wise errors between two Complex-valued images x and y of a given shape sh
function plotError(figID, x, y, sh)

    fig = figure("Voxel-wise Reconstruction Errors + $figID", figsize = (10, 4))
    absErrorTerm = Flux.Losses.mae.(abs.(x), abs.(y)) ./ abs.(x)
    angleErrorTerm = Flux.Losses.mae.(angle.(x), angle.(y))

    reshapedMag = reshape(absErrorTerm, sh[1], sh[2])
    reshapedAngle = reshape(angleErrorTerm, sh[1], sh[2])

    subplot(121)
    title("Magnitude Error")
    imshow(reshapedMag, vmin = 0, vmax = 1.0, cmap = "jet")
    colorbar()

    subplot(122)
    title("Phase Error")
    imshow(reshapedAngle, vmin = -pi, vmax = pi, cmap = "Spectral")
    colorbar()

end



## Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
function weighted_EMulx_Tullio_Sep(x_re, x_im, nodes, positions, weights)

    # Separation of real and imaginary parts to play well with GPU
    @tullio RE_E[k, n] := cos <| (-Float32(pi) * 2 * nodes[i, k] * $positions[i, n])
    @tullio IM_E[k, n] := sin <| (-Float32(pi) * 2 * nodes[i, k] * $positions[i, n])

    @tullio y_re[k] := (weights[k] * RE_E[k, n]) * x_re[n] - (weights[k] * IM_E[k, n]) * x_im[n]
    @tullio y_im[k] := (weights[k] * IM_E[k, n]) * x_re[n] + (weights[k] * RE_E[k, n]) * x_im[n]

    # w_re = weights .* y_re
    # w_im = weights .* y_im

    return (y_re, y_im)

end

## Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
function weighted_EMulx_Tullio_Sep(x_re, x_im, nodes, positions, weights, b0_map, times)

    # Separation of real and imaginary parts to play well with GPU
    @tullio RE_E[k, n] := cos <| (-Float32(pi) * 2 * nodes[i, k] * $positions[i, n] - times[k]*b0_map[n])
    @tullio IM_E[k, n] := sin <| (-Float32(pi) * 2 * nodes[i, k] * $positions[i, n] - times[k]*b0_map[n])

    @tullio y_re[k] := (weights[k] * RE_E[k, n]) * x_re[n] - (weights[k] * IM_E[k, n]) * x_im[n]
    @tullio y_im[k] := (weights[k] * IM_E[k, n]) * x_re[n] + (weights[k] * RE_E[k, n]) * x_im[n]

    # w_re = weights .* y_re
    # w_im = weights .* y_im

    return (y_re, y_im)

end


function get_weights(gradients)

    @tullio W[k] := sqrt <| $gradients[i, k] * $gradients[i, k] ## Define weights as magnitude of gradients
    W = W ./ maximum(W)
    return W

end

## Weighted Version of Matrix-Vector Multiplication using Tullio.jl with real matrices and CUDA compat...
function weighted_EHMulx_Tullio_Sep(x_re, x_im, nodes, positions, weights)

    # Separation of real and imaginary parts to play well with GPU
    @tullio RE_E[n, k] :=  cos <| (Float32(pi)  * 2 * $positions[i, n] * nodes[i, k])
    @tullio IM_E[n, k] :=  sin <| (Float32(pi)  * 2 * $positions[i, n] * nodes[i, k])

    @tullio y_re[n] := RE_E[n, k] * ($weights[k]*x_re[k]) - IM_E[n, k] * ($weights[k]*x_im[k])
    @tullio y_im[n] := IM_E[n, k] * ($weights[k]*x_re[k]) + RE_E[n, k] * ($weights[k]*x_im[k])

    return (y_re, y_im)

end

## Weighted Version of Matrix-Vector Multiplication using Tullio.jl with real matrices and CUDA compat...
function weighted_EHMulx_Tullio_Sep(x_re, x_im, nodes, positions, weights, b0_map, times)

    # Separation of real and imaginary parts to play well with GPU
    @tullio RE_E[n, k] := cos <| (Float32(pi) * 2 * $positions[i, n] * nodes[i, k] + b0_map[n]*times[k])
    @tullio IM_E[n, k] := sin <| (Float32(pi) * 2 * $positions[i, n] * nodes[i, k] + b0_map[n]*times[k])

    @tullio y_re[n] := RE_E[n, k] * ($weights[k]*x_re[k]) - IM_E[n, k] * ($weights[k]*x_im[k])
    @tullio y_im[n] := IM_E[n, k] * ($weights[k]*x_re[k]) + RE_E[n, k] * ($weights[k]*x_im[k])

    return (y_re, y_im)

end



## Get gradients from the trajectory
function nodes_to_gradients(nodes)

    newNodes = hcat(CuArray([0; 0]), nodes)
    gradients = diff(newNodes, dims = 2)
    return gradients

end



## Pad gradients to prepare for Tullio
function pad_gradients(gradients, kernelSize)

    padding = CuArray(zeros(Float32, kernelSize[1], kernelSize[2] - 1))
    padded = hcat(padding, gradients)
    return padded

end

## Filter gradients using Tullio for efficient convolution
function filter_gradients(gradients, kernel)

    @tullio d[b, i] := $gradients[b, i+a-1] * kernel[b, a]
    return d

end



## Convert gradients to trajectory nodes
function gradients_to_nodes(gradients)

    nodes = cumsum(gradients, dims = 2)
    return nodes

end

## Efficient function to apply a time domain gradient impulse response function kernel to the trajectory (2D only now)
function apply_td_girf(nodes, kernel)

    gradients = nodes_to_gradients(nodes)
    padded = pad_gradients(gradients, size(kernel))
    filtered = filter_gradients(padded, kernel)
    filtered_nodes = gradients_to_nodes(filtered)
    return filtered_nodes

end

## Get the padded gradient waveform
function get_padded_gradients(nodes, kernelSize)

    g = nodes_to_gradients(nodes)
    padded = pad_gradients(g, kernelSize)
    return padded

end

## Convert Vector of Vectors to Matrix
function vecvec_to_matrix(vecvec)

    dim1 = length(vecvec)
    dim2 = length(vecvec[1])

    my_array = zeros(Float32, dim1, dim2)

    for i = 1:dim1
        for j = 1:dim2
            my_array[i, j] = vecvec[i][j]
        end
    end

    return my_array

end

## Get the positions corresponding to the strong voxel condition for a given image Shape
function getPositions(sh::Tuple)

    # set up positions according to strong voxel condition
    x = collect(1:sh[1]) .- sh[1] / 2 .- 1
    y = collect(1:sh[2]) .- sh[2] / 2 .- 1

    p = Iterators.product(x, y)

    positions = collect(Float64.(vecvec_to_matrix(vec(collect.(p))))')

    return positions

end

## Helper function for reshaping nodes to size expected by Flux dataloader
function reshapeNodes(x)

    s = size(x)
    reshape(x, 1, s[2], s[1], 1)

end

## Helper function for undoing the reshaping of nodes to size expected by Flux dataloader
function undoReshape(x)

    r = size(x)
    reshape(x, r[3], r[2])

end

## Loss function for the minimization -> works over the real and imaginary parts of the image separately
function loss(x, y)

    Flux.Losses.mae(x[1],y[1]) + Flux.Losses.mae(x[2],y[2])
    
end


## Generates ground truth gaussian kernel
# TODO: Add support for variable width
function getGaussianKernel(kernel_length)

    ## Generate Ground Truth Filtering Kernel
    ker = rand(2, kernel_length)
    ker[1, :] = exp.(.-(-kernel_length÷2:kernel_length÷2) .^ 2 ./ (50))
    ker[2, :] = exp.(.-(-kernel_length÷2:kernel_length÷2) .^ 2 ./ (25))
    ker = ker ./ sum(ker, dims = 2)

end

## Generates delay kernel
function deltaKernel(kernel_length, shift)

    x = zeros(2,kernel_length)
    x[:,kernel_length - shift] .= 1.0
    return x

end

function pull_from_gpu(imTuple::Tuple)

    cpuTuple = imTuple |> cpu
    return complex.(cpuTuple...)

end

function pull_from_gpu(array::CuMatrix)

    array |> cpu

end