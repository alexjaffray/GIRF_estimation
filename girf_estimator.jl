using MRIReco,
    DSP,
    NIfTI,
    FiniteDifferences,
    PyPlot,
    Waveforms,
    Distributions,
    ImageFiltering,
    Flux,
    CUDA,
    NFFT,
    Zygote,
    TestImages

pygui(true)

# figure()
# plot(testK[1,:],testK[2,:])

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

    imfilter(G, reflect(centered(theta)), Fill(0))

end

function prepareE(imShape)

    # construct E in place
    E = Array{ComplexF32}(undef, imShape[1] * imShape[2], imShape[1] * imShape[2])

    # set up positions according to strong voxel condition
    x = collect(1:imShape[1]) .- imShape[1] / 2 .- 1
    y = collect(1:imShape[2]) .- imShape[2] / 2 .- 1

    p = Iterators.product(x, y)

    positions = vecvec_to_matrix(vec(collect.(p)))

    return E, positions

end

function constructE!(E, positions::Matrix, nodes::Matrix)

    @time phi = nodes' * positions'

    @time Threads.@threads for i in eachindex(phi)
        E[i] = cispi(-2 * phi[i])
    end

end

function constructEAdjoint!(EAdj, positions::Matrix, nodes::Matrix)

    @time phi = positions * nodes

    @time Threads.@threads for i in eachindex(phi)
        EAdj[i] = cispi(2 * phi[i])
    end

end

function vecvec_to_matrix(vecvec)
    dim1 = length(vecvec)
    dim2 = length(vecvec[1])
    my_array = zeros(Int64, dim1, dim2)
    for i = 1:dim1
        for j = 1:dim2
            my_array[i, j] = vecvec[i][j]
        end
    end
    return my_array
end

function doOp(x, data)

    op = ExplicitOp((N, N), x, Array{ComplexF64}(undef, 0, 0))
    op \ data

end

Zygote.@adjoint function doOp(x, data)

    op = ExplicitOp((N, N), x, Array{ComplexF64}(undef, 0, 0))

    opResult = op \ data

    function back(ΔopResult)
        B̄ = op' \ ΔopResult

        return (-B̄ * opResult', B̄)

    end

    return opResult, back

end

function getPrediction(x, data)

    nodesX = reshapeNodes(x.nodes[1, :])
    x.nodes[1, :] = vec(model(nodesX))

    doOp(x, data)

end

function reshapeNodes(x)

    reshape(x, 1, length(x), 1, 1)

end

function loss(x, y)

    Flux.Losses.mae(getPrediction(x, dataRef), y)

end

## Generate Ground Truth Filtering Kernel
ker = rand(6)
ker = ker ./ sum(ker)

## Test Setting Up Simulation (forward sim)
N = 226
M = 186

imShape = (N, M)

I = Float64.(TestImages.testimage("mri_stack"))[:, :, 13]
#I = circularShutterFreq!(I, 1)

## Simulation parameters
parameters = Dict{Symbol,Any}()
parameters[:simulation] = "fast"
parameters[:trajName] = "Spiral"
parameters[:numProfiles] = 1
parameters[:numSamplingPerProfile] = N * M
parameters[:windings] = 128
parameters[:AQ] = 3.0e-2

## Do simulation
acqData = simulation(I, parameters)

## Define Perfect Reconstruction
EAdj₀, positions = prepareE(imShape)
constructEAdjoint!(EAdj₀, positions, acqData.traj[1].nodes)
recon1 = EAdj₀ * acqData.kdata[1]

## Plot the actual nodes used for the perfect reconstruction
figure()
scatter(acqData.traj[1].nodes[1, :], acqData.traj[1].nodes[2, :])

oldNodesX = acqData.traj[1].nodes[1, :]
oldNodesY = acqData.traj[1].nodes[2, :]

## Filter the nodes with some kernel and write them back into acqData
filteredWaveformX = filterGradientWaveForms(diff(acqData.traj[1].nodes[1, :]), ker)
filteredWaveformY = filterGradientWaveForms(diff(acqData.traj[1].nodes[2, :]), ker)

newNodesY = prepend!(vec(cumsum(filteredWaveformY)), [0.0])
newNodesX = prepend!(vec(cumsum(filteredWaveformX)), [0.0])

acqData.traj[1].nodes[1, :] = newNodesX
acqData.traj[1].nodes[2, :] = newNodesY

## Plot the new nodes
scatter(acqData.traj[1].nodes[1, :], acqData.traj[1].nodes[2, :])

## Reconstruct with perturbed nodes
EAdj, positions = prepareE(imShape)
constructEAdjoint!(EAdj, positions, acqData.traj[1].nodes)
recon2 = EAdj * acqData.kdata[1]

plotError(recon1, recon2, imShape)

## Define ML Model
layer = Conv((1, 200), 1 => 1, pad = SamePad())
model = Chain(layer)

# # Test The layer Idea
# layer = Conv((1,30),1=>30,identity; bias=true, pad=SamePad())
# testDat = reshape(oldNodesX,1,256*256,1,1)

trajRef = deepcopy(acqData.traj[1])
dataRef = deepcopy(vec(acqData.kdata[1]))
reconRef = recon1

## Do Training of Model for one iteration
parameters = Flux.params(model)
opt = Descent()
# Flux.train!(loss, parameters, [(trajRef, reconRef)], opt)





