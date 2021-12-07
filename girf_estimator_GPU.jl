using
    MRIReco,
    DSP,
    NIfTI,
    FiniteDifferences,
    PyPlot,
    Waveforms,
    Distributions,
    ImageFiltering,
    ImageTransformations,
    Flux,
    CUDA,
    NFFT,
    Zygote,
    TestImages,
    LinearAlgebra,
    KernelAbstractions,
    CUDAKernels,
    Tullio


#use pyplot backend with interactivity turned on
pygui(true)

## Have to set this to the real number of threads because MRIReco.jl seems to set this to 1 when it's loaded :(
BLAS.set_num_threads(32)

## Plot the Euclidean error between the two trajectories
function plotTrajectoryError(x, y)

    figure("Pointwise Trajectory Error")
    plot(sqrt.(abs2.(y[1, :]) .+ abs2.(y[2, :])) - sqrt.(abs2.(x[1, :]) + abs2.(x[2, :])))

    xlabel("Sample Index")
    ylabel("Euclidean Distance between Nominal and Actual Positions")

end

## Show reconstructed image magnitude and phase including normalization if specified
function showReconstructedImage(x, sh, do_normalization)

    fig = figure("Reconstruction", figsize = (10, 4))

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
    imshow(reshapedAngle, vmin = -pi, vmax = pi, cmap = "seismic")
    colorbar()


end

## Function for plotting the voxel-wise errors between two Complex-valued images x and y of a given shape sh
function plotError(x, y, sh)

    fig = figure("Voxel-wise Reconstruction Errors", figsize = (10, 4))
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

    @tullio y_re[k] := RE_E[k, n] * x_re[n] - IM_E[k, n] * x_im[n]
    @tullio y_im[k] := IM_E[k, n] * x_re[n] + RE_E[k, n] * x_im[n]

    w_re = weights .* y_re
    w_im = weights .* y_im

    return (w_re, w_im)

end


function get_weights(gradients)

    @tullio W[k] := sqrt <| $gradients[i, k] * $gradients[i, k] ## Define weights as magnitude of gradients
    W = W ./ maximum(W)
    return W

end

## Weighted Version of Matrix-Vector Multiplication using Tullio.jl with real matrices and CUDA compat...
function weighted_EHMulx_Tullio_Sep(x_re, x_im, nodes, positions, weights)

    w_re = weights .* x_re
    w_im = weights .* x_im

    # Separation of real and imaginary parts to play well with GPU
    @tullio RE_E[n, k] := cos <| (Float32(pi) * 2 * $positions[i, n] * nodes[i, k])
    @tullio IM_E[n, k] := sin <| (Float32(pi) * 2 * $positions[i, n] * nodes[i, k])

    @tullio y_re[n] := RE_E[n, k] * w_re[k] - IM_E[n, k] * w_im[k]
    @tullio y_im[n] := IM_E[n, k] * w_re[k] + RE_E[n, k] * w_im[k]

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

    #Flux.Losses.mse(real(x), real(y)) + Flux.Losses.mse(imag(x), imag(y))
    Flux.Losses.mae(x, y)

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

function pull_from_gpu(imTuple)

    cpuTuple = imTuple |> cpu
    return complex.(cpuTuple...)

end

## DEBUG
CUDA.allowscalar(false)

## Define Kernel Length
kernel_length = 11

## Get ground truth kernel
ker = getGaussianKernel(kernel_length)

ker = CuMatrix(Float32.(ker))

## Test Setting Up Simulation (forward sim)
N = 156
M = 128
imShape = (N, M)

B = Float64.(TestImages.testimage("mri_stack"))[:, :, 17]

img_small = ImageTransformations.imresize(B, imShape)
#img_medium = ImageTransformations.restrict(img_small)

I_mage = img_small

imShape = size(I_mage)

I_mage = circularShutterFreq!(I_mage, 1)

# Simulation parameters
parameters = Dict{Symbol,Any}()
parameters[:simulation] = "fast"
parameters[:trajName] = "Spiral"
parameters[:numProfiles] = 1
parameters[:numSamplingPerProfile] = imShape[1] * imShape[2] * 2
parameters[:windings] = 86
parameters[:AQ] = 3.0e-2

## Do simulation to get the trajectory to perturb!
acqData = simulation(I_mage, parameters)
positions = getPositions(imShape)
nodesRef = deepcopy(acqData.traj[1].nodes)
signalRef = deepcopy(acqData.kdata[1])

## Put everything on the GPU
positionsRef = CuMatrix(Float32.(positions))
nodesRef = CuMatrix(Float32.(nodesRef))
image_real = CuMatrix(Float32.(real.(I_mage)))
image_imag = CuMatrix(Float32.(imag.(I_mage)))

## Make test simulation (neglecting T2 effects, etc...) using the tullio function 
@time referenceSim = weighted_EMulx_Tullio_Sep(image_real, image_imag, nodesRef, positionsRef, get_weights(nodes_to_gradients(nodesRef)))

## Define Perfect Reconstruction
@time recon1 = weighted_EHMulx_Tullio_Sep(referenceSim[1], referenceSim[2], nodesRef, positionsRef, get_weights(nodes_to_gradients(nodesRef)))
#normalizeRecon!(recon1)

## Show the reconstruction
showReconstructedImage(pull_from_gpu(recon1), imShape, true)

## Plot the actual nodes used for the perfect reconstruction
#figure()
#scatter(nodesRef[1, :], nodesRef[2, :])

perturbedNodes = apply_td_girf(nodesRef, ker)

@time perturbedSim = weighted_EMulx_Tullio_Sep(image_real, image_imag, perturbedNodes, positionsRef, get_weights(nodes_to_gradients(perturbedNodes)))

## Plot the new nodes
#scatter(perturbedNodes[1, :], perturbedNodes[2, :])

## Reconstruct with perturbed nodes
@time recon2 = weighted_EHMulx_Tullio_Sep(perturbedSim[1], perturbedSim[2], perturbedNodes, positionsRef, get_weights(nodes_to_gradients(perturbedNodes)))
showReconstructedImage(pull_from_gpu(recon2), imShape, true)
#normalizeRecon!(recon2)

#plotError(recon1, recon2, imShape)

## Input Data
reconRef = deepcopy(recon2)
positionsRef = deepcopy(collect(positions))
weights = get_weights(nodes_to_gradients(nodesRef))

initialReconstruction = weighted_EHMulx_Tullio_Sep(perturbedSim[1],perturbedSim[2], nodesRef, positionsRef, get_weights(nodes_to_gradients(perturbedNodes)))
showReconstructedImage(pull_from_gpu(initialReconstruction), imShape, true)
plotError(initialReconstruction, recon2, imShape)

# Syntax for gradients is entirely based on implicit anonymous functions in Flux, and the documentation of this syntax is implicit as well. What the Flux man!
# See example below: 

# (x...) -> loss(x..., arg1, arg2...)

## Do Training of Model for one iteration
# parameters = Flux.params(model)
# opt = ADAGrad()

# for i = 1:200
#     Flux.train!(loss, parameters, [(dataRef, reconRef, nodesRef, positions)], opt)
# end

## Gradient of the sensitivity matrix is sparse so we intuitively choose ADAM as our Optimizer
opt = ADAM() # Add 0.00001 as learning rate for better performance.

sqnorm(x) = sum(abs2, x)

## Number of iterations until convergence
numiters = 100

testKernLength = kernel_length

# @info "\nUSING CPU: "
# @benchmark weighted_EHMulx_Tullio(perturbedSim, perturbedNodes, positions,get_weights(nodes_to_gradients(perturbedNodes)))

## Prepare CUDA
kernel = CuMatrix(ones(2, testKernLength) ./ testKernLength)
dat = CuVector{Float64}(undef, numiters)
datK = CuVector{Float64}(undef, numiters)

positionsRef = CuMatrix(positionsRef)
weights = CuVector(weights)
perturbedSim = CuVector(perturbedSim)
reconRef = CuVector(reconRef)
nodesRef = CuMatrix(nodesRef)
perturbedNodes = CuMatrix(perturbedNodes)

# ## InTerp vars into benchmark
# @info "\nUSING GPU: "
# @benchmark weighted_EHMulx_Tullio(perturbedSim, perturbedNodes, positionsRef,get_weights(nodes_to_gradients(perturbedNodes)))

for i = 1:numiters

    ps = Params([kernel])

    train_loss, back = Zygote.pullback(() -> loss(weighted_EHMulx_Tullio(perturbedSim, real(apply_td_girf(nodesRef, kernel)), positionsRef, get_weights(nodes_to_gradients((apply_td_girf(nodesRef, kernel))))), reconRef), ps)

    gs = back(one(train_loss))

    println(gs)
    # CUDA.@allowscalar dat[i] = training_loss
    # CUDA.@allowscalar datK[i] = Flux.Losses.mse(ker,kernel)

    println("[ITERATION $i] Train  Loss: ", train_loss)
    # print("[ITERATION $i] Kernel Loss: ", datK[i],"\n")

    Flux.update!(opt, ps, gs)

end

# figure()
# plot(dat)
# plot(datK)

outputTrajectory = real(apply_td_girf(nodesRef, kernel))

@time finalRecon = weighted_EHMulx_Tullio(perturbedSim, outputTrajectory, positionsRef, get_weights(nodes_to_gradients(outputTrajectory))) |> cpu

plotError(finalRecon, recon2, imShape)

#showReconstructedImage(finalRecon,imShape,true)

plotError(finalRecon, recon1, imShape)
