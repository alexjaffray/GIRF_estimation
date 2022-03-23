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
    Tullio,
    ROMEO


#use pyplot backend with interactivity turned on
pygui(true)

## Have to set this to the real number of threads because MRIReco.jl seems to set this to 1 when it's loaded :(
BLAS.set_num_threads(32)

include("utils.jl")

## DEBUG
CUDA.allowscalar(false)

## Define Kernel Length
kernel_length = 3

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
parameters[:AQ] = 1e-6 * parameters[:numSamplingPerProfile]
parameters[:correctionMap] = 1im .* quadraticFieldmap(N,M,125.0*2*pi)

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

b0_map = CuVector(Float32.(vec(polarFieldmap(N,M,125.0*pi))))
times = CuVector(Float32.(acqData.traj[1].times))

cuSig = (CuVector(Float32.(real.(vec(acqData.kdata[1])))),CuVector(Float32.(imag.(vec(acqData.kdata[1])))))
@time simReconGT = weighted_EHMulx_Tullio_Sep(cuSig[1],cuSig[2], nodesRef, positionsRef, get_weights(nodes_to_gradients(nodesRef)))
showReconstructedImage(pull_from_gpu(simReconGT), imShape, true)

## Make test simulation (neglecting T2 effects, etc...) using the tullio function 
@time referenceSim = weighted_EMulx_Tullio_Sep(image_real, image_imag, nodesRef, positionsRef, get_weights(nodes_to_gradients(nodesRef)), b0_map, times)

## Define Perfect Reconstruction
@time recon1 = weighted_EHMulx_Tullio_Sep(referenceSim[1], referenceSim[2], nodesRef, positionsRef, get_weights(nodes_to_gradients(nodesRef)),b0_map , times)
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

plotError(pull_from_gpu(recon1), pull_from_gpu(recon2), imShape)

## Input Data
reconRef = deepcopy(recon2)
weights = get_weights(nodes_to_gradients(nodesRef))

naiveReconstruction = weighted_EHMulx_Tullio_Sep(perturbedSim[1],perturbedSim[2], nodesRef, positionsRef, get_weights(nodes_to_gradients(perturbedNodes)))
showReconstructedImage(pull_from_gpu(naiveReconstruction), imShape, true)


plotError(pull_from_gpu(naiveReconstruction), pull_from_gpu(recon2), imShape)

# BELOW IS COMMENTED OUT

# Syntax for gradients is entirely based on implicit anonymous functions in Flux, and the documentation of this syntax is implicit as well. What the Flux man!
# See example below: 

# (x...) -> loss(x..., arg1, arg2...)

# Do Training of Model for one iteration
# parameters = Flux.params(model)
# opt = ADAGrad()

# for i = 1:200
#     Flux.train!(loss, parameters, [(dataRef, reconRef, nodesRef, positions)], opt)
# end

# ## Gradient of the sensitivity matrix is sparse so we intuitively choose ADAM as our Optimizer
# opt = ADAM() # Add 0.00001 as learning rate for better performance.

# sqnorm(x) = sum(abs2, x)

# ## Number of iterations until convergence
# numiters = 100

# testKernLength = kernel_length

# # @info "\nUSING CPU: "
# # @benchmark weighted_EHMulx_Tullio(perturbedSim, perturbedNodes, positions,get_weights(nodes_to_gradients(perturbedNodes)))

# ## Prepare CUDA
# kernel = CuMatrix(ones(2, testKernLength) ./ testKernLength)
# dat = CuVector{Float64}(undef, numiters)
# datK = CuVector{Float64}(undef, numiters)

# positionsRef = CuMatrix(positionsRef)
# weights = CuVector(weights)
# perturbedSim = (CuVector(perturbedSim[1]), CuVector(perturbedSim[2]))
# reconRef = (CuVector(reconRef[1]), CuVector(reconRef[2]))
# nodesRef = CuMatrix(nodesRef)
# perturbedNodes = CuMatrix(perturbedNodes)

# # ## InTerp vars into benchmark
# # @info "\nUSING GPU: "
# # @benchmark weighted_EHMulx_Tullio(perturbedSim, perturbedNodes, positionsRef,get_weights(nodes_to_gradients(perturbedNodes)))

# # WORKS UP TO HERE

# for i = 1:numiters

#     local weights = get_weights(nodes_to_gradients(apply_td_girf(nodesRef, kernel)))
#     local training_loss

#     ps = Params([kernel])
#     @info "made it here"
#     gs = gradient(ps) do
#         training_loss = loss(weighted_EHMulx_Tullio_Sep(perturbedSim[1], perturbedSim[2],apply_td_girf(nodesRef,kernel),positionsRef, weights),reconRef) #+ 500*sqnorm(kernel)
#         return training_loss
#     end
#     @info "made it past the gradient calculation"
#     println(gs)
#     # CUDA.@allowscalar dat[i] = training_loss
#     # CUDA.@allowscalar datK[i] = Flux.Losses.mse(ker,kernel)

#     println("[ITERATION $i] Train  Loss: ", Flux.Losses.mse(ker,kernel))
#     # print("[ITERATION $i] Kernel Loss: ", datK[i],"\n")

#     Flux.update!(opt, ps, gs)

# end

# # figure()
# # plot(dat)
# # plot(datK)

# outputTrajectory = real(apply_td_girf(nodesRef, kernel))

# @time finalRecon = weighted_EHMulx_Tullio(perturbedSim, outputTrajectory, positionsRef, get_weights(nodes_to_gradients(outputTrajectory))) |> cpu

# plotError(finalRecon, recon2, imShape)

# #showReconstructedImage(finalRecon,imShape,true)

# plotError(finalRecon, recon1, imShape)
