using MRIReco, DSP, NIfTI, FiniteDifferences, PyPlot, Waveforms, Distributions, ImageFiltering, Flux, CUDA

pygui(true)


# figure()
# plot(testK[1,:],testK[2,:])


function filterGradientWaveForms(G,theta)

    imfilter(G,reflect(centered(theta)),Fill(0))

end


function simulatePerfectRecon(EH,b₀)

    EH * b₀

end

ker = rand(30)
ker = ker ./ sum(ker)

#ker = ImageFiltering.Kernel.gaussian(30) 

## Test Setting Up Simulation (forward sim)

N = 256
I = shepp_logan(N)
I = circularShutterFreq!(I,1)

# simulation parameters
parameters = Dict{Symbol, Any}()
parameters[:simulation] = "explicit"
parameters[:trajName] = "Spiral"
parameters[:numProfiles] = 1
parameters[:numSamplingPerProfile] = N*N
parameters[:windings] = 128
parameters[:AQ] = 3.0e-2

# do simulation
acqData = simulation(I, parameters)

# Define Perfect Reconstruction Operation 

testOp = NFFTOp((256,256), acqData.traj[1])
testOp2 = adjoint(testOp)

recon1 = simulatePerfectRecon(testOp2, vec(acqData.kdata[1]))

figure()
scatter(acqData.traj[1].nodes[1,:], acqData.traj[1].nodes[2,:])

oldNodesX = acqData.traj[1].nodes[1,:]
oldNodesY = acqData.traj[1].nodes[2,:]

filteredWaveformX = filterGradientWaveForms(diff(acqData.traj[1].nodes[1,:]),ker)
filteredWaveformY = filterGradientWaveForms(diff(acqData.traj[1].nodes[2,:]),ker)

newNodesY = prepend!(vec(cumsum(filteredWaveformY)),[0.0])
newNodesX = prepend!(vec(cumsum(filteredWaveformX)),[0.0])

acqData.traj[1].nodes[1,:] = newNodesX
acqData.traj[1].nodes[2,:] = newNodesY

scatter(acqData.traj[1].nodes[1,:], acqData.traj[1].nodes[2,:])

testOp3 = NFFTOp((256,256), acqData.traj[1])
testOp4 = adjoint(testOp3)

recon2 = testOp4*vec(acqData.kdata[1])

error = mse(recon1,recon2)

figure()
plot(error)

figure()
plot(sqrt.(abs2.(oldNodesX) + abs2.(oldNodesY)) - sqrt.(abs2.(newNodesX) .+ abs2.(newNodesY)))

layer = Conv((1,30),1=>30,pad=SamePad())
layer2 = ConvTranspose((1,5),30=>7, pad=SamePad())
layer3 = ConvTranspose((1,6),7=>1, pad = SamePad())

model = Chain(layer,layer2,layer3)

# # Test The layer Idea
# layer = Conv((1,30),1=>30,identity; bias=true, pad=SamePad())
# testDat = reshape(oldNodesX,1,256*256,1,1)

trajRef = deepcopy(acqData.traj[1])
dataRef = deepcopy(vec(acqData.kdata[1]))
reconRef = testOp2 * vec(acqData.kdata[1])

function getPrediction(x)

    nodesX = reshapeNodes(x.nodes[1,:])
    x.nodes[1,:] = vec(model(nodesX))
    return x

end

function reshapeNodes(x)

    reshape(x, 1,length(x),1,1)

end

loss(x,y) = Flux.Losses.mae(adjoint(NFFTOp((256,256),getPrediction(x)))*dataRef, y)

