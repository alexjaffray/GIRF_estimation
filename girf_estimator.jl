using 
    MRIReco,
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

    imfilter(G, reflect(centered(theta)), Fill(0))

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

    E = constructE(nodes,positions)
    y = E*x
    return y

end

## Single Threaded Explicit Passing Version for Autodiff compat.
function EHMulx(x, nodes::Matrix, positions::Matrix)

    EH = constructEH(nodes,positions)
    y = EH*x
    return y

end

## Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
function EMulx_Tullio(x, nodes::Matrix{Float64}, positions::Matrix{Float64})

    @tullio E[k,n] := exp <| (-1.0im * pi * 2.0 * nodes[i,k]*positions[i,n])
    @tullio y[k] := E[k,n]*x[n]

    return y

end

## Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
function EHMulx_Tullio(x, nodes::Matrix{Float64}, positions::Matrix{Float64})

    @tullio EH[n,k] := exp <| (1.0im * pi * 2.0 * positions[i,n]*nodes[i,k])
    @tullio y[n] := EH[n,k]*x[k]

    return y

end

## Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
function EHMulx_Tullio(x, nodes::Array{Float64,4}, positions::Matrix{Float64})

    nodes2 = undoReshape(nodes)

    @tullio EH[n,k] := exp <| (1.0im * pi * 2.0 * positions[i,n]*nodes2[i,k])
    @tullio y[n] := EH[n,k]*x[k]

    return y

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

# function doOp(x, data)

#     op = ExplicitOp((N, N), x, Array{ComplexF64}(undef, 0, 0))
#     op \ data

# end

# Zygote.@adjoint function doOp(x, data)

#     op = ExplicitOp((N, N), x, Array{ComplexF64}(undef, 0, 0))

#     opResult = op \ data

#     function back(ΔopResult)
#         B̄ = op' \ ΔopResult

#         return (-B̄ * opResult', B̄)

#     end

#     return opResult, back

# end

# function getPrediction(x, data)

#     nodesX = reshapeNodes(x.nodes[1, :])
#     x.nodes[1, :] = vec(model(nodesX))

#     doOp(x, data)

# end

function reshapeNodes(x)

    s = size(x)
    reshape(x,1, s[2], s[1], 1)

end

function undoReshape(x)

    r = size(x)
    reshape(x,r[3],r[2])

end

function loss(x, y)

    Flux.Losses.mse(angle.(x),angle.(y))

end

## Generate Ground Truth Filtering Kernel
ker = rand(6)
ker = ker ./ sum(ker)

## Test Setting Up Simulation (forward sim)
N = 226
M = 186

imShape = (N, M)

I = Float64.(TestImages.testimage("mri_stack"))
#I = circularShutterFreq!(I, 1)

## Simulation parameters
parameters = Dict{Symbol,Any}()
parameters[:simulation] = "fast"
parameters[:trajName] = "Spiral"
parameters[:numProfiles] = 1
parameters[:numSamplingPerProfile] = N * M
parameters[:windings] = 8
parameters[:AQ] = 3.0e-2

## Do simulation
acqData = simulation(I, parameters)
positions = getPositions(imShape)

## Define Perfect Reconstruction

@time recon1 = EHMulx_Tullio(acqData.kdata[1],acqData.traj[1].nodes,positions)
nodesRef = Float64.(deepcopy(acqData.traj[1].nodes))

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

perturbedNodes = deepcopy(acqData.traj[1].nodes)

## Plot the new nodes
scatter(perturbedNodes[1, :], perturbedNodes[2, :])

## Reconstruct with perturbed nodes
@time recon2 = EHMulx_Tullio(acqData.kdata[1],perturbedNodes,positions)
plotError(recon1, recon2, imShape)

## Define ML Model
layer = Conv((1, 3), 2 => 2, pad = SamePad())
model = Chain(layer)


# # Test The layer Idea
# layer = Conv((1,30),1=>30,identity; bias=true, pad=SamePad())
# testDat = reshape(oldNodesX,1,256*256,1,1)

# These should All be Float32 types
trajRef = deepcopy(acqData.traj[1])
dataRef = deepcopy(vec(acqData.kdata[1]))
reconRef = deepcopy(recon2)
positionsRef = deepcopy(collect(positions))

pN_r = reshapeNodes(nodesRef)

# Syntax for gradients is entirely based on implicit anonymous functions in Flux, and the documentation of this syntax is implicit as well. What the Flux man!
# See example below: 

# (x...) -> loss(x..., arg1, arg2...)

# ## Do Training of Model for one iteration
# parameters = Flux.params(model)
# opt = ADAGrad()

# for i = 1:200
#     Flux.train!(loss, parameters, [(dataRef, reconRef, nodesRef, positions)], opt)
# end

f = 0
learningRate = 0.0001
opt = ADAM()
figure()

sqnorm(x) = sum(abs2, x)

numiters = 1000

dat = Vector{Float64}(undef,numiters)

for i = 1:numiters
    
    local training_loss
    ps = Params(Flux.params(model))
    gs = gradient(ps) do
        training_loss = loss(EHMulx_Tullio(dataRef,model(pN_r),positionsRef),reconRef) + sum(sqnorm,pN_r)
        return training_loss
    end

    dat[i] = training_loss

    print(training_loss,"\n")

    Flux.update!(opt,ps,gs)

end

figure()
plot(dat)