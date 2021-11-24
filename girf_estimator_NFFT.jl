using MRIReco, DSP, NIfTI, FiniteDifferences, PyPlot, Waveforms, Distributions, ImageFiltering, Flux, CUDA, NFFT, Zygote

pygui(true)


# figure()
# plot(testK[1,:],testK[2,:])


function filterGradientWaveForms(G, theta)

    imfilter(G, reflect(centered(theta)), Fill(0))

end


function simulatePerfectRecon(EH, b₀)

    EH * b₀

end

ker = rand(30)
ker = ker ./ sum(ker)

#ker = ImageFiltering.Kernel.gaussian(30) 

## Test Setting Up Simulation (forward sim)

N = 128
I = shepp_logan(N)
I = circularShutterFreq!(I, 1)

# simulation parameters
parameters = Dict{Symbol,Any}()
parameters[:simulation] = "explicit"
parameters[:trajName] = "Spiral"
parameters[:numProfiles] = 1
parameters[:numSamplingPerProfile] = N * N
parameters[:windings] = 32
parameters[:AQ] = 3.0e-2

# do simulation
acqData = simulation(I, parameters)

# Define Perfect Reconstruction Operation 

testOp = NFFTOp((N, N), acqData.traj[1])
testOp2 = adjoint(testOp)

recon1 = simulatePerfectRecon(testOp2, vec(acqData.kdata[1]))

figure()
scatter(acqData.traj[1].nodes[1, :], acqData.traj[1].nodes[2, :])

oldNodesX = acqData.traj[1].nodes[1, :]
oldNodesY = acqData.traj[1].nodes[2, :]

filteredWaveformX = filterGradientWaveForms(diff(acqData.traj[1].nodes[1, :]), ker)
filteredWaveformY = filterGradientWaveForms(diff(acqData.traj[1].nodes[2, :]), ker)

newNodesY = prepend!(vec(cumsum(filteredWaveformY)), [0.0])
newNodesX = prepend!(vec(cumsum(filteredWaveformX)), [0.0])

acqData.traj[1].nodes[1, :] = newNodesX
acqData.traj[1].nodes[2, :] = newNodesY

scatter(acqData.traj[1].nodes[1, :], acqData.traj[1].nodes[2, :])

testOp3 = NFFTOp((N, N), acqData.traj[1])
testOp4 = adjoint(testOp3)

recon2 = testOp4 * vec(acqData.kdata[1])

error = mse(recon1, recon2)

figure()
plot(error)

figure()
plot(sqrt.(abs2.(oldNodesX) + abs2.(oldNodesY)) - sqrt.(abs2.(newNodesX) .+ abs2.(newNodesY)))

## Test the layer idea
layer = Conv((1, 200), 1 => 1, pad = SamePad())
model = Chain(layer)


trajRef = deepcopy(acqData.traj[1])
dataRef = deepcopy(vec(acqData.kdata[1]))
reconRef = testOp2 * vec(acqData.kdata[1])

function doOp(x,nodes, data)

    y = deepcopy(x)
    y.nodes[1,:] = nodes[1,:]

    op = NFFTOp((N, N), x)
    op \ data

end

Zygote.@adjoint function doOp(x₂, nodes2, data)

    x = deepcopy(x₂)
    x.nodes[1,:] = nodes2[1,:]

    op = NFFTOp((N, N), x)

    y = op \ data

    function back(Δy)
        B̄ = op' \ Δy

        return (-B̄ * y', B̄)

    end

    return y, back 

end

function getPrediction(x, data)

    nodesX = reshapeNodes(x.nodes[1, :])
    newNodes = vec(model(nodesX))
    
    doOp(x,data)

    # Change two lines -> 
    # op = NFFTOp((N, N), x)
    # op \ data

end

function reshapeNodes(x)

    reshape(x, 1, length(x), 1, 1)

end

function loss(x, y)

    Flux.Losses.mae(getPrediction(x, dataRef), y)

end

parameters = Flux.params(model)

opt = Descent()
Flux.train!(loss, parameters, [(trajRef, reconRef)], opt)

# NOTE: Current implementation of the NFFTOp relies on FFTW calls and so is inherently not autodifferentiable without adding custom Adjoint. Need to ask Jon about this tomorrow. 
# Can also do the operation explicitly and it just takes a very long time...

imShape = (N,N)

function prepareE(imShape)

    # construct E in place
    E = Array{ComplexF32}(undef, imShape[1] * imShape[2], imShape[1] * imShape[2])

    # set up positions according to strong voxel condition
    x = collect(1:imShape[1]) .- imShape[1]/2 .- 1
    y = collect(1:imShape[2]) .- imShape[2]/2 .- 1

    p = Iterators.product(x,y)

    positions = vecvec_to_matrix(vec(collect.(p)))

    return E, positions

end

function constructE!(E, positions::Matrix, nodes::Matrix)

    phi = nodes'*positions'

    E .= exp.(-2*1im*pi*phi)

end

function E_adjoint(E)

    transpose(conj.(E))

end

function vecvec_to_matrix(vecvec)
    dim1 = length(vecvec)
    dim2 = length(vecvec[1])
    my_array = zeros(Int64, dim1, dim2)
    for i in 1:dim1
        for j in 1:dim2
            my_array[i,j] = vecvec[i][j]
        end
    end
    return my_array
end

