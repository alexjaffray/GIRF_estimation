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
function showReconstructedImage(x,sh,do_normalization)

    fig = figure("Reconstruction", figsize = (10, 4))

    x_max = maximum(abs.(x))

    reshapedMag = reshape(abs.(x), sh[1], sh[2])
    reshapedAngle = reshape(angle.(x), sh[1], sh[2])

    ## Normalize step:
    if do_normalization
        reshapedMag = reshapedMag./x_max
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
function compareReconstructedImages(x,y,sh,do_normalization)

    fig = figure("Reconstruction Comparison", figsize = (10, 8))

    x_max = maximum(abs.(x))
    y_max = maximum(abs.(y))

    reshapedMag_x = reshape(abs.(x), sh[1], sh[2])
    reshapedAngle_x = reshape(angle.(x), sh[1], sh[2])

    reshapedMag_y = reshape(abs.(y), sh[1], sh[2])
    reshapedAngle_y = reshape(angle.(y), sh[1], sh[2])


    ## Normalize step:
    if do_normalization
        reshapedMag_x = reshapedMag_x./x_max
        x_max = 1.0
        reshapedMag_y = reshapedMag_y./y_max
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
function plotTrajectories(t_nom,t_perturbed,t_solved)

    figure("Trajectory Comparison", figsize = (12,12))
    
    gt_error = sqrt.(abs2.(t_nom[1, :]) .+ abs2.(t_nom[2, :])) - sqrt.(abs2.(t_perturbed[1, :]) + abs2.(t_perturbed[2, :]))
    solved_error = sqrt.(abs2.(t_solved[1, :]) .+ abs2.(t_solved[2, :])) - sqrt.(abs2.(t_perturbed[1, :]) + abs2.(t_perturbed[2, :]))

    subplot(211)
    plot(t_nom',label="Nominal Trajectory")
    plot(t_perturbed',label = "GT Trajectory")
    plot(t_solved', label = "Solved Trajectory")
    legend(loc="upper left")

    subplot(212)
    plot(gt_error, label="Nominal w.r.t GT")
    plot(solved_error, label="Solved w.r.t GT")
    xlabel("Sample Index")
    ylabel("Euclidean Distance between Nominal and Actual Positions")
    legend(loc="upper left")

end

function plotKernels(k_gt,k_est)

    kernel_size_difference = size(k_est,2) - size(k_gt,2)
    k_gt_padded = hcat(zeros(2, kernel_size_difference), k_gt)

    figure("Kernel Comparison")
    plot(k_gt_padded',label="ground truth kernel")
    plot(k_est',label="estimate kernel")
    legend(loc="upper left")
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

function get_weights(gradients::Matrix)

    @tullio W[k] := sqrt <| $gradients[i,k]*$gradients[i,k] ## Define weights as magnitude of gradients
    W = W/maximum(W)
    return W

end

## Weighted Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
function weighted_EHMulx_Tullio(x, nodes::Matrix{Float64}, positions::Matrix{Float64}, weights::Vector{Float64})

    #DENSITY COMPENSATION FUNCTION AS DESCRIBED IN NOLL, FESSLER and SUTTON
    #@tullio W[k] := sqrt <| $gradients[i,k]*$gradients[i,k] ## Define weights as magnitude of gradients
    @tullio EH[n, k] := exp <| (1.0im * pi * 2.0 * $positions[i, n] * nodes[i, k])
    @tullio y[n] := EH[n, k] * (weights[k]*$x[k])

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

    gradients = diff(hcat([0;0],nodes), dims = 2)
    return gradients

end

## Pad gradients to prepare for Tullio
function pad_gradients(gradients::Matrix, kernelSize)

    padding = zeros(kernelSize[1], kernelSize[2] - 1)
    padded = hcat(padding,gradients)
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
    @tullio RE_E[k, n] := cos <| (- pi * 2.0 * nodes[i, k] * $positions[i, n])
    @tullio IM_E[k, n] := sin <| (- pi * 2.0 * nodes[i, k] * $positions[i, n])

    @tullio y_re[k] := RE_E[k, n]*x_re[n] - IM_E[k, n]*x_im[n]
    @tullio y_im[k] := IM_E[k, n]*x_re[n] + RE_E[k, n]*x_im[n]

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

    @tullio y_re[n] := RE_E[n,k]*w_re[k] - IM_E[n,k]*w_im[k]
    @tullio y_im[n] := IM_E[n,k]*w_re[k] + RE_E[n,k]*w_im[k]

    return (y_re,y_im)

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
    padded = pad_gradients(g,kernelSize)
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
    Flux.Losses.mae(x,y)

end

## Custom simulation function 
function groundtruth_sim(nodes::Matrix, image, kernel, positions)

    outputData = EMulx_Tullio(vec(image),apply_td_girf(nodes, kernel),positions)
    return outputData


end

## Generates ground truth gaussian kernel
# TODO: Add support for variable width
function getGaussianKernel(kernel_length)

    ## Generate Ground Truth Filtering Kernel
    ker = rand(2,kernel_length)
    ker[1,:] = exp.(.-(-kernel_length÷2:kernel_length÷2 ).^2 ./ (5))
    ker[2,:] = exp.(.-(-kernel_length÷2:kernel_length÷2 ).^2 ./ (20))
    ker = ker ./ sum(ker, dims=2)    

end

## Generates delay kernel
function deltaKernel(kernel_length, shift)

    x = zeros(2,kernel_length)
    x[:,kernel_length - shift] .= 1.0
    return x

end

## Define Kernel Length
kernel_length = 9

## Get ground truth kernel
ker = getGaussianKernel(kernel_length)
#ker = deltaKernel(kernel_length, 1)

## Test Setting Up Simulation (forward sim)
N = 32
M = 32
imShape = (N, M)

B = Float64.(TestImages.testimage("mri_stack"))[:, :, 14]

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
parameters[:numSamplingPerProfile] = imShape[1] * imShape[2]*2
parameters[:windings] =25
parameters[:AQ] = 3.0e-2

## Do simulation to get the trajectory to perturb!
acqData = simulation(I_mage, parameters)
positions = getPositions(imShape)
nodesRef = deepcopy(acqData.traj[1].nodes)
signalRef = deepcopy(acqData.kdata[1])

## Make test simulation (neglecting T2 effects, etc...) using the tullio function 
@time referenceSim = weighted_EMulx_Tullio(I_mage, nodesRef, positions, get_weights(nodes_to_gradients(nodesRef)))

## Define Perfect Reconstruction
@time recon1 = weighted_EHMulx_Tullio(referenceSim, nodesRef, positions, get_weights(nodes_to_gradients(nodesRef)))
#normalizeRecon!(recon1)
showReconstructedImage(recon1, imShape,true)


## Plot the actual nodes used for the perfect reconstruction
figure()
scatter(nodesRef[1, :], nodesRef[2, :])

perturbedNodes = apply_td_girf(nodesRef, ker)

@time perturbedSim = weighted_EMulx_Tullio(I_mage, perturbedNodes, positions, get_weights(nodes_to_gradients(perturbedNodes)))

## Plot the new nodes
scatter(perturbedNodes[1, :], perturbedNodes[2, :])

## Reconstruct with perturbed nodes
@time recon2 = weighted_EHMulx_Tullio(perturbedSim, perturbedNodes, positions,get_weights(nodes_to_gradients(perturbedNodes)))
showReconstructedImage(recon2, imShape,true)
#normalizeRecon!(recon2)

plotError(recon1, recon2, imShape)

## Input Data
reconRef = deepcopy(recon2)
positionsRef = deepcopy(collect(positions))
weights = get_weights(nodes_to_gradients(nodesRef))

initialReconstruction = weighted_EHMulx_Tullio(perturbedSim,nodesRef,positionsRef, weights)
showReconstructedImage(initialReconstruction,imShape,true)
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
opt = ADAM(0.001) # Add 0.00001 as learning rate for better performance.

sqnorm(x) = sum(abs2, x)

## Number of iterations until convergence
numiters = 1000

testKernLength = kernel_length

kernel = ones(2,testKernLength)./testKernLength

dat = Vector{Float64}(undef,numiters)
datK = Vector{Float64}(undef,numiters)

kernel_size_difference = size(kernel,2) - size(ker,2)
padded_ker = hcat(zeros(2, kernel_size_difference), ker)

# Test Adding Noise to the perturbed Data!
perturbedSim = perturbedSim + randn(length(perturbedSim)) + 0.5 .* 1im.*randn(length(perturbedSim))

for i = 1:numiters

    local weights = get_weights(nodes_to_gradients(real(apply_td_girf(nodesRef, kernel))))
    local training_loss

    ps = Params([kernel])
    gs = gradient(ps) do
        training_loss = loss(weighted_EHMulx_Tullio(perturbedSim,real(apply_td_girf(nodesRef,kernel)),positionsRef, weights),reconRef) + norm(kernel,1)
        return training_loss
    end

    dat[i] = training_loss
    datK[i] = Flux.Losses.mse(padded_ker,kernel)

    print("[ITERATION $i] Train  Loss: ",dat[i],"\n")
    print("[ITERATION $i] Kernel Loss: ", datK[i],"\n")

    
    Flux.update!(opt,ps,gs)

end

figure()
plot(dat)
plot(datK)

outputTrajectory = real(apply_td_girf(nodesRef,kernel))

@time finalRecon = weighted_EHMulx_Tullio(perturbedSim,outputTrajectory,positionsRef, get_weights(nodes_to_gradients(outputTrajectory)))

plotError(finalRecon,recon2, imShape)

showReconstructedImage(finalRecon,imShape,true)

compareReconstructedImages(initialReconstruction,finalRecon,imShape,true)
plotTrajectories(nodesRef, perturbedNodes,outputTrajectory)

plotKernels(ker,kernel)