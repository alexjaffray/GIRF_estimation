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

## Show reconstructed image magnitude and phase including normalization if specified
function compareReconstructedImages(x, y, sh, do_normalization)

    fig = figure("Reconstruction Comparison", figsize = (10, 8))

    x_max = maximum(abs.(x))
    y_max = maximum(abs.(y))

    reshapedMag_x = reshape(abs.(x), sh[1], sh[2])
    reshapedAngle_x = reshape(angle.(x), sh[1], sh[2])

    reshapedMag_y = reshape(abs.(y), sh[1], sh[2])
    reshapedAngle_y = reshape(angle.(y), sh[1], sh[2])


    ## Normalize step:
    if do_normalization
        reshapedMag_x = reshapedMag_x ./ x_max
        x_max = 1.0
        reshapedMag_y = reshapedMag_y ./ y_max
        y_max = 1.0
    end

    subplot(221)
    title("Magnitude (Initial)")
    imshow(reshapedMag_x, vmin = 0, vmax = x_max, cmap = "gray")
    colorbar()

    subplot(222)
    title("Phase (Initial)")
    imshow(reshapedAngle_x, vmin = -pi, vmax = pi, cmap = "seismic")
    colorbar()

    subplot(223)
    title("Magnitude (Final)")
    imshow(reshapedMag_y, vmin = 0, vmax = y_max, cmap = "gray")
    colorbar()

    subplot(224)
    title("Phase (Final)")
    imshow(reshapedAngle_y, vmin = -pi, vmax = pi, cmap = "seismic")
    colorbar()


end

## Plot the Euclidean error between the two trajectories
function plotTrajectories(t_nom, t_perturbed, t_solved)

    figure("Trajectory Comparison", figsize = (12, 12))

    gt_error = sqrt.(abs2.(t_nom[1, :]) .+ abs2.(t_nom[2, :])) - sqrt.(abs2.(t_perturbed[1, :]) + abs2.(t_perturbed[2, :]))
    solved_error = sqrt.(abs2.(t_solved[1, :]) .+ abs2.(t_solved[2, :])) - sqrt.(abs2.(t_perturbed[1, :]) + abs2.(t_perturbed[2, :]))

    subplot(211)
    plot(t_nom', label = ["Nominal kx Trajectory", "Nominal ky Trajectory"])
    plot(t_perturbed', label = ["Ground Truth kx Trajectory", "Ground Truth ky Trajectory"])
    plot(t_solved', label = ["Estimated kx Trajectory", "Estimated ky Trajectory"])
    xlabel("Sampling Index")
    ylabel("K-Space Position")
    legend(loc = "upper right")

    subplot(212)
    plot(gt_error, label = "Nominal Error w.r.t GT")
    plot(solved_error, label = "Estimated Error w.r.t GT")
    xlabel("Sample Index")
    ylabel("Sampling Position Error")
    legend(loc = "upper right")

end

function plotKernels(k_gt, k_est)

    kernel_size_difference = size(k_est, 2) - size(k_gt, 2)
    k_gt_padded = hcat(zeros(2, kernel_size_difference), k_gt)

    sampleIndices = -size(k_gt_padded, 2)+1:0

    figure("Kernel Comparison")
    plot(sampleIndices, k_gt_padded', label = ["Ground-truth kernel (x-dir)", "Ground-truth kernel (y-dir)"])
    plot(sampleIndices, k_est', label = ["Estimated kernel (x-dir)", "Estimated kernel (y-dir)"])
    xlabel("Sampling Index")
    legend(loc = "upper left")
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

## Loss Evolution Plotting Function
function plotLoss(loss, kernLoss, trajLoss)

    figure("Loss Over Time")
    plot(loss ./ loss[1], label = "Recon (Training) Loss")
    plot(kernLoss ./ kernLoss[1], label = "Kernel Loss")
    plot(trajLoss ./ trajLoss[1], label = "Trajectory Loss")
    legend(loc = "upper right")
    title("Normalized Loss")
    xlabel("Iteration")
    ylabel("Loss")

end

## Generic Allocator for the E system matrix
function prepareE(imShape)

    # construct E in place
    E = Array{ComplexF32}(undef, imShape[1] * imShape[2], imShape[1] * imShape[2])

    positions = getPositions(imShape)

    return E, positions

end

## Memory Efficient Multi-threaded in-place E constructor
function constructE!(E, nodes::Matrix, positions::Matrix)

    phi = nodes' * positions

    Threads.@threads for i in eachindex(phi)
        E[i] = cispi(-2 * phi[i])
    end

end

## Memory Efficient Multi-threaded in-place EH constructor
function constructEH!(EH, nodes::Matrix, positions::Matrix)

    phi = positions * nodes'

    Threads.@threads for i in eachindex(phi)
        EH[i] = cispi(2 * phi[i])
    end

end

## Single Threaded Explicit Passing Version for Autodiff compat. 
function EMulx(x, nodes::Matrix, positions::Matrix)

    E = constructE(nodes, positions)
    y = E * x
    return y

end

## Single Threaded Explicit Passing Version for Autodiff compat.
function EHMulx(x, nodes::Matrix, positions::Matrix)

    EH = constructEH(nodes, positions)
    y = EH * x
    return y

end

## Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
function EMulx_Tullio(x, nodes::Matrix{Float64}, positions::Matrix{Float64})

    @tullio E[k, n] := exp <| (-1.0im * pi * 2.0 * nodes[i, k] * $positions[i, n])
    @tullio y[k] := E[k, n] * $x[n]

    return y

end

## Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
function weighted_EMulx_Tullio(x, nodes::Matrix{Float64}, positions::Matrix{Float64}, weights::Vector{Float64})

    #@tullio W[k] := sqrt <| $gradients[i,k]*$gradients[i,k] ## Define weights as magnitude of gradients
    @tullio E[k, n] := exp <| (-1.0im * pi * 2.0 * nodes[i, k] * $positions[i, n])
    @tullio y[k] := weights[k] * E[k, n] * $x[n]

    return y

end

## Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
function EHMulx_Tullio(x, nodes::Matrix{Float64}, positions::Matrix{Float64})

    @tullio EH[n, k] := exp <| (1.0im * pi * 2.0 * $positions[i, n] * nodes[i, k])
    @tullio y[n] := EH[n, k] * $x[k]

    return y

end

## Calculates sampling weights for uniform radial sampling 
function get_weights(gradients::Matrix)

    @tullio W[k] := sqrt <| $gradients[i, k] * $gradients[i, k] ## Define weights as magnitude of gradients
    W = W / maximum(W)
    return W

end

## Weighted Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
function weighted_EHMulx_Tullio(x, nodes::Matrix{Float64}, positions::Matrix{Float64}, weights::Vector{Float64})

    #DENSITY COMPENSATION FUNCTION AS DESCRIBED IN NOLL, FESSLER and SUTTON
    #@tullio W[k] := sqrt <| $gradients[i,k]*$gradients[i,k] ## Define weights as magnitude of gradients
    @tullio EH[n, k] := exp <| (1.0im * pi * 2.0 * $positions[i, n] * nodes[i, k])
    @tullio y[n] := EH[n, k] * (weights[k] * $x[k])

    return y

end

## Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
function EHMulx_Tullio(x, nodes::Array{Float64,4}, positions::Matrix{Float64})

    nodes2 = undoReshape(nodes)

    @tullio EH[n, k] := exp <| (1.0im * pi * 2.0 * $positions[i, n] * nodes2[i, k])
    @tullio y[n] := EH[n, k] * $x[k]

    return y

end

## Get gradients from the trajectory
function nodes_to_gradients(nodes::Matrix)

    gradients = diff(hcat([0; 0], nodes), dims = 2)
    return gradients

end

## Pad gradients to prepare for Tullio
function pad_gradients(gradients::Matrix, kernelSize)

    padding = zeros(kernelSize[1], kernelSize[2] - 1)
    padded = hcat(padding, gradients)
    return padded

end

## Filter gradients using Tullio for efficient convolution
function filter_gradients(gradients::Matrix, kernel::Matrix)

    @tullio d[b, i+_] := gradients[b, i+a] * kernel[b, a] #verbose = true
    return d

end

## Convert gradients to trajectory nodes
function gradients_to_nodes(gradients::Matrix)

    nodes = cumsum(gradients, dims = 2)
    return nodes

end

## Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
function weighted_EMulx_Tullio_Sep(x_re, x_im, nodes, positions, weights)

    # Separation of real and imaginary parts to play well with GPU
    @tullio RE_E[k, n] := cos <| (-pi * 2.0 * nodes[i, k] * $positions[i, n])
    @tullio IM_E[k, n] := sin <| (-pi * 2.0 * nodes[i, k] * $positions[i, n])

    @tullio y_re[k] := RE_E[k, n] * x_re[n] - IM_E[k, n] * x_im[n]
    @tullio y_im[k] := IM_E[k, n] * x_re[n] + RE_E[k, n] * x_im[n]

    w_re = weights .* y_re
    w_im = weights .* y_im

    return (w_re, w_im)

end

## Weighted Version of Matrix-Vector Multiplication using Tullio.jl with real matrices and CUDA compat...
function weighted_EHMulx_Tullio_Sep(x_re, x_im, nodes, positions, weights)

    w_re = weights .* x_re
    w_im = weights .* x_im

    # Separation of real and imaginary parts to play well with GPU
    @tullio RE_E[n, k] := cos <| (pi * 2.0 * $positions[i, n] * nodes[i, k])
    @tullio IM_E[n, k] := sin <| (pi * 2.0 * $positions[i, n] * nodes[i, k])

    @tullio y_re[n] := RE_E[n, k] * w_re[k] - IM_E[n, k] * w_im[k]
    @tullio y_im[n] := IM_E[n, k] * w_re[k] + RE_E[n, k] * w_im[k]

    return (y_re, y_im)

end

## Efficient function to apply a time domain gradient impulse response function kernel to the trajectory (2D only now)
function apply_td_girf(nodes::Matrix, kernel::Matrix)
    gradients = nodes_to_gradients(nodes)
    padded = pad_gradients(gradients, size(kernel))
    filtered = filter_gradients(padded, kernel)
    filtered_nodes = gradients_to_nodes(filtered)
    return filtered_nodes

end

## Get the padded gradient waveform
function get_padded_gradients(nodes::Matrix, kernelSize::Tuple)

    g = nodes_to_gradients(nodes)
    padded = pad_gradients(g, kernelSize)
    return padded

end

## Single Threaded Explicit Passing Version for Autodiff compat.
function constructE(nodes::Matrix, positions::Matrix)

    phi = nodes' * positions'

    # Multithreaded
    # Threads.@threads for i in eachindex(phi)
    #     E[i] = cispi(-2 * phi[i])
    # end

    E = cispi.(-2 * phi[i])

    return E

end

## Single Threaded Explicit Passing Version for Autodiff compat.
function constructEH(nodes::Matrix, positions::Matrix)

    phi = positions * nodes

    # Multithreaded
    # Threads.@threads for i in eachindex(phi)
    #     EH[i] = cispi(2 * phi[i])
    # end

    EH = cispi.(2 * phi)

    return EH

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

## Custom simulation function 
function groundtruth_sim(nodes::Matrix, image, kernel, positions)

    outputData = EMulx_Tullio(vec(image), apply_td_girf(nodes, kernel), positions)
    return outputData

end

## Generates ground truth gaussian kernel of different width in x and y directions
# TODO: Add support for variable width
function getGaussianKernel(kernel_length)

    ## Generate Ground Truth Filtering Kernel
    ker = rand(2, kernel_length)
    ker[1, :] = exp.(.-(-kernel_length÷2:kernel_length÷2) .^ 2 ./ (5))
    ker[2, :] = exp.(.-(-kernel_length÷2:kernel_length÷2) .^ 2 ./ (20))
    ker = ker ./ sum(ker, dims = 2)

end

## Generates delay kernel
function deltaKernel(kernel_length, shift)

    x = zeros(2, kernel_length)
    x[:, kernel_length-shift] .= 1.0
    return x

end

## Define Kernel Length
kernel_length = 7

## Get ground truth kernel
DelayKernel = false

if !DelayKernel
    ker = getGaussianKernel(kernel_length)
else
    ker = deltaKernel(kernel_length, 2)
end

## Set up Simulation (forward sim)
N = 32
M = 32
imShape = (N, M)

## Read in test MRI Image
B = Float64.(TestImages.testimage("mri_stack"))[:, :, 14]

## Resize the test MRI Image and reset the new Size
img_small = ImageTransformations.imresize(B, imShape)
I_mage = img_small
imShape = size(I_mage)

## Filter the k-space of the image corresponding to circular trajectory extent
I_mage = circularShutterFreq!(I_mage, 1)

# Simulation parameters
parameters = Dict{Symbol,Any}()
parameters[:simulation] = "fast"
parameters[:trajName] = "Spiral"
parameters[:numProfiles] = 1
parameters[:numSamplingPerProfile] = imShape[1] * imShape[2] * 2
parameters[:windings] = 30
parameters[:AQ] = parameters[:numSamplingPerProfile] * 2e-6 # Set 2μs dwell time

## Do simulation to get the trajectory to perturb!
acqData = simulation(I_mage, parameters)

# Clone reference data
positions = getPositions(imShape)
nodesRef = deepcopy(acqData.traj[1].nodes)
signalRef = deepcopy(acqData.kdata[1])

## Make test simulation (neglecting T2 effects, etc...) using the tullio function 
@time referenceSim = weighted_EMulx_Tullio(I_mage, nodesRef, positions, get_weights(nodes_to_gradients(nodesRef)))

## Define Perfect Reconstruction and Plot
@time recon1 = weighted_EHMulx_Tullio(referenceSim, nodesRef, positions, get_weights(nodes_to_gradients(nodesRef)))
showReconstructedImage(recon1, imShape, true)

## Plot the actual nodes used for the perfect reconstruction
figure()
scatter(nodesRef[1, :], nodesRef[2, :])
title("K-Space Nodes")
xlabel("kx")
ylabel("ky")

## Generate ground truth perturbed k-space trajectory
perturbedNodes = apply_td_girf(nodesRef, ker)

## Simulate acquisition with perturbed nodes
@time perturbedSim = weighted_EMulx_Tullio(I_mage, perturbedNodes, positions, get_weights(nodes_to_gradients(perturbedNodes)))

## Plot the new nodes overlaid on the old nodes
scatter(perturbedNodes[1, :], perturbedNodes[2, :])

## Reconstruct with perturbed nodes
@time recon2 = weighted_EHMulx_Tullio(perturbedSim, perturbedNodes, positions, get_weights(nodes_to_gradients(perturbedNodes)))

## Plot
showReconstructedImage(recon2, imShape, true)
plotError(recon1, recon2, imShape)

## Clone reference data and use as input Data
reconRef = deepcopy(recon2)
positionsRef = deepcopy(collect(positions))
weights = get_weights(nodes_to_gradients(nodesRef))

## Naive Reconstruction (nominal recon, perturbed signal)
initialReconstruction = weighted_EHMulx_Tullio(perturbedSim, nodesRef, positionsRef, weights)
showReconstructedImage(initialReconstruction, imShape, true)
plotError(initialReconstruction, recon2, imShape)

########################################################################
## Optimization Routine!

## Gradient of the sensitivity matrix is sparse so we intuitively choose ADAM as our Optimizer
opt = ADAM(0.00015)

## Number of iterations until convergence
numiters = 1500

## Regularization Parameters
α = 0.0005 # Regularization parameter for L2
β = 10 # Regularization parameter for L1

## Get kernel length from g-t kernel and add additional length?
testKernLength = kernel_length + 3

## Initialize kernel with uniform values summing to 1
kernel = ones(2, testKernLength) ./ testKernLength

## Create logging vectors
lossTrack = Vector{Float64}(undef, numiters)
kernelLossTrack = Vector{Float64}(undef, numiters)
trajLossTrack = Vector{Float64}(undef, numiters)


## Calculate kernel size difference and pad if necessary
kernel_size_difference = size(kernel, 2) - size(ker, 2)
padded_ker = hcat(zeros(2, kernel_size_difference), ker)

## Add Noise to the perturbed Data!
perturbedSim = perturbedSim + randn(length(perturbedSim)) + 0.5 .* 1im .* randn(length(perturbedSim))

## Optimization Loop
for i = 1:numiters

    local weights = get_weights(nodes_to_gradients(real(apply_td_girf(nodesRef, kernel))))
    local training_loss

    # Set kernel as tracked
    ps = Params([kernel])

    # Get the gradients of the loss function w.r.t kernel and return loss
    gs = gradient(ps) do
        reconEst = weighted_EHMulx_Tullio(perturbedSim, real(apply_td_girf(nodesRef, kernel)), positionsRef, weights)
        training_loss = loss(reconEst, recon1) + α * norm(reconEst, 2) + β * norm(kernel, 1)
        return training_loss
    end

    # Log the losses and print
    lossTrack[i] = training_loss
    kernelLossTrack[i] = Flux.Losses.mse(padded_ker, kernel)
    trajLossTrack[i] = Flux.Losses.mse(real(apply_td_girf(nodesRef, kernel)), perturbedNodes)

    print("[ITERATION $i] Train  Loss: ", lossTrack[i], "\n")
    print("[ITERATION $i] Kernel Loss: ", kernelLossTrack[i], "\n")
    print("[ITERATION $i] Trajectory Loss: ", trajLossTrack[i], "\n")


    # Update the kernel!
    Flux.update!(opt, ps, gs)

end


## Calculate output trajectory based on estimated kernel and reconstruct with trajectory
outputTrajectory = real(apply_td_girf(nodesRef, kernel))
@time finalRecon = weighted_EHMulx_Tullio(perturbedSim, outputTrajectory, positionsRef, get_weights(nodes_to_gradients(outputTrajectory)))

## Final Plotting Functions

plotLoss(lossTrack, kernelLossTrack, trajLossTrack)
plotError(finalRecon, recon2, imShape)
showReconstructedImage(finalRecon, imShape, true)
compareReconstructedImages(initialReconstruction, finalRecon, imShape, true)
plotTrajectories(nodesRef, perturbedNodes, outputTrajectory)
plotKernels(ker, kernel)

