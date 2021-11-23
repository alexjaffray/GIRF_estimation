using MRIReco, DSP, NIfTI, FiniteDifferences, PyPlot, Waveforms, Distributions, ImageFiltering

pygui(true)

testTrajectory = SpiralTrajectoryVarDens(4,128*128)
testK = testTrajectory.nodes

figure()
plot(testK[1,:],testK[2,:])

gradTrainX = diff(testK[1,:])
gradTrainY = diff(testK[2,:])

function filterGradientWaveForms(G,theta)

    conv(G, theta)

end

ker = rand()
ker = ker ./ sum(ker)
#ker = ImageFiltering.Kernel.gaussian((30,))

filteredWaveformX = filterGradientWaveForms(gradTrainX,ker)
filteredWaveformY = filterGradientWaveForms(gradTrainY,ker)

trajectoryX = cumsum(filteredWaveformX)
trajectoryY = cumsum(filteredWaveformY)

figure()
scatter(trajectoryX, trajectoryY)
scatter(testK[1,:], testK[2,:])

## Test Setting Up Simulation (forward sim)

N = 256
I = shepp_logan(N)
I = circularShutterFreq!(I,1)

# simulation parameters
params = Dict{Symbol, Any}()
params[:simulation] = "explicit"
params[:trajName] = "Spiral"
params[:numProfiles] = 1
params[:numSamplingPerProfile] = N*N
params[:windings] = 128
params[:AQ] = 3.0e-2

# do simulation
acqData = simulation(I, params)

testOp = NFFTOp((256,256), acqData.traj[1])
testOp2 = adjoint(testOp)

recon1 = testOp2*vec(acqData.kdata[1])

figure()
scatter(acqData.traj[1].nodes[1,:], acqData.traj[1].nodes[2,:])

filteredWaveformX = filterGradientWaveForms(diff(acqData.traj[1].nodes[1,:]),ker)
filteredWaveformY = filterGradientWaveForms(diff(acqData.traj[1].nodes[2,:]),ker)

acqData.traj[1].nodes[1,:] = cumsum(filteredWaveformX)[1:65536]
acqData.traj[1].nodes[2,:] = cumsum(filteredWaveformY)[1:65536]

scatter(acqData.traj[1].nodes[1,:], acqData.traj[1].nodes[2,:])

testOp3 = NFFTOp((256,256), acqData.traj[1])
testOp4 = adjoint(testOp3)

recon2 = testOp4*vec(acqData.kdata[1])

error = sum(vec(abs.(recon2)-abs.(recon1)))./sum(vec(abs.(recon1)))

figure()
plot(error)




