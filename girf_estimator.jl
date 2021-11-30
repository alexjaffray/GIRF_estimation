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

pygui(true)

## Have to set this to the real number of threads because MRIReco.jl seems to set this to 1 when it's loaded :(
BLAS.set_num_threads(32)

## Plot the Euclidean error between the two trajectories

function plotTrajectoryError(x, y)

    figure("Pointwise Trajectory Error")
    plot(sqrt.(abs2.(y[1, :]) .+ abs2.(y[1, :]) - sqrt.(abs2.(x[1, :]) + abs2.(x[2, :]))))

    xlabel("Sample Index")
    ylabel("Euclidean Distance between Nominal and Actual Positions")

end

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
    imshow(reshapedAngle, vmin = 0.0, vmax = pi, cmap = "jet")
    colorbar()

end

function filterGradientWaveForms(G, theta)

    r = DSP.conv(G, theta)[1:length(G)]

    figure()
    plot(r)
    plot(G)

    return r

end

function getFilteredTrajectory(nodes::Matrix, theta)



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
function EHMulx_Tullio(x, nodes::Matrix{Float64}, positions::Matrix{Float64})

    @tullio EH[n, k] := exp <| (1.0im * pi * 2.0 * $positions[i, n] * nodes[i, k])
    @tullio y[n] := EH[n, k] * $x[k]

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

    gradients = diff([[0; 0] nodes], dims = 2)
    return gradients

end

## Pad gradients to prepare for Tullio
function pad_gradients(gradients::Matrix, kernelSize)

    padding = zeros(kernelSize[1], kernelSize[2] - 1)
    padded = [padding gradients]
    return padded

end

## Filter gradients using Tullio for efficient convolution
function filter_gradients(gradients::Matrix, kernel::Matrix)

    @tullio d[b, i+_] := gradients[b, i+a] * kernel[b, a]
    return d

end

## Convert gradients to trajectory nodes
function gradients_to_nodes(gradients::Matrix)

    nodes = cumsum(gradients, dims = 2)
    return nodes

end

## Efficient function to apply a time domain gradient impulse response function kernel to the trajectory (2D only now)
function apply_td_girf(nodes::Matrix, kernel::Matrix)

    gradients = nodes_to_gradients(nodes)
    padded = pad_gradients(gradients, size(kernel))
    filtered = filter_gradients(padded, kernel)
    filtered_nodes = gradients_to_nodes(filtered)
    return filtered_nodes

end

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

function getPositions(sh::Tuple)

    # set up positions according to strong voxel condition
    x = collect(1:sh[1]) .- sh[1] / 2 .- 1
    y = collect(1:sh[2]) .- sh[2] / 2 .- 1

    p = Iterators.product(x, y)

    positions = collect(Float64.(vecvec_to_matrix(vec(collect.(p))))')

    return positions

end

function reshapeNodes(x)

    s = size(x)
    reshape(x, 1, s[2], s[1], 1)

end

function undoReshape(x)

    r = size(x)
    reshape(x, r[3], r[2])

end

function loss(x, y)

    Flux.Losses.mse(real(x), real(y)) + Flux.Losses.mse(imag(x), imag(y))

end

kernel_length = 10

## Generate Ground Truth Filtering Kernel
ker = rand(2,kernel_length)
ker = ker ./ sum(ker, dims=2)

## Test Setting Up Simulation (forward sim)
N = 226
M = 186

imShape = (N, M)

B = Float64.(TestImages.testimage("mri_stack"))[:, :, 13]

img_small = ImageTransformations.restrict(B)
img_medium = ImageTransformations.restrict(img_small)

I_mage = img_medium

imShape = size(I_mage)

I_mage = circularShutterFreq!(I_mage, 1)

# Simulation parameters
parameters = Dict{Symbol,Any}()
parameters[:simulation] = "fast"
parameters[:trajName] = "Spiral"
parameters[:numProfiles] = 1
parameters[:numSamplingPerProfile] = imShape[1] * imShape[2]
parameters[:windings] = 18
parameters[:AQ] = 3.0e-2

## Do simulation
acqData = simulation(I_mage, parameters)
positions = getPositions(imShape)

## Define Perfect Reconstruction
@time recon1 = EHMulx_Tullio(acqData.kdata[1], acqData.traj[1].nodes, positions)
#normalizeRecon!(recon1)

nodesRef = deepcopy(acqData.traj[1].nodes)

## Plot the actual nodes used for the perfect reconstruction
figure()
scatter(nodesRef[1, :], nodesRef[2, :])

perturbedNodes = apply_td_girf(nodesRef, ker)

## Plot the new nodes
scatter(perturbedNodes[1, :], perturbedNodes[2, :])

## Reconstruct with perturbed nodes
@time recon2 = EHMulx_Tullio(acqData.kdata[1], perturbedNodes, positions)
#normalizeRecon!(recon2)

plotError(recon1, recon2, imShape)

## Input Data
trajRef = deepcopy(acqData.traj[1])
dataRef = deepcopy(vec(acqData.kdata[1]))
reconRef = deepcopy(recon2)
positionsRef = deepcopy(collect(positions))
gradientsRef = nodes_to_gradients(nodesRef)

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
opt = ADAM()

sqnorm(x) = sum(abs2, x)

## Number of iterations until convergence
numiters = 10

kernel = ones(2,kernel_length)./kernel_length

dat = Vector{Float64}(undef,numiters)
datK = Vector{Float64}(undef,numiters)

for i = 1:numiters

    local training_loss
    ps = Params([kernel])
    gs = gradient(ps) do
        training_loss = loss(EHMulx_Tullio(dataRef,real(apply_td_girf(nodesRef,kernel)),positionsRef),reconRef) + sqnorm(kernel)
        return training_loss
    end

    dat[i] = training_loss
    datK[i] = Flux.mse(ker,kernel)

    print("TrainingLoss: ",dat[i],"\n")
    print("Kernel Loss: ", datK[i],"\n")

    Flux.update!(opt,ps,gs)

end

figure()
plot(dat./maximum(dat))
plot(datK./maximum(datK))

outputTrajectory = real(apply_td_girf(nodesRef,kernel))

@time finalRecon = EHMulx_Tullio(dataRef,real(apply_td_girf(nodesRef,kernel)),positionsRef)

plotError(finalRecon,recon2, imShape)
#plotError(finalRecon,recon1, imShape)