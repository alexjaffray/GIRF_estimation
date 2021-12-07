# ## Custom simulation function 
# function groundtruth_sim(nodes::Matrix, image, kernel, positions)

#     outputData = EMulx_Tullio(vec(image),apply_td_girf(nodes, kernel),positions)
#     return outputData


# end

# ## Single Threaded Explicit Passing Version for Autodiff compat.
# function constructE(nodes::Matrix, positions::Matrix)

#     phi = nodes' * positions'

#     # Multithreaded
#     # Threads.@threads for i in eachindex(phi)
#     #     E[i] = cispi(-2 * phi[i])
#     # end

#     E = cispi.(-2 * phi[i])

#     return E

# end

# ## Single Threaded Explicit Passing Version for Autodiff compat.
# function constructEH(nodes::Matrix, positions::Matrix)

#     phi = positions * nodes

#     # Multithreaded
#     # Threads.@threads for i in eachindex(phi)
#     #     EH[i] = cispi(2 * phi[i])
#     # end

#     EH = cispi.(2 * phi)

#     return EH

# end

# ## Efficient function to apply a time domain gradient impulse response function kernel to the trajectory (2D only now)
# function apply_td_girf(nodes::Matrix, kernel::Matrix)

#     gradients = nodes_to_gradients(nodes)
#     padded = pad_gradients(gradients, size(kernel))
#     filtered = filter_gradients(padded, kernel)
#     filtered_nodes = gradients_to_nodes(filtered)
#     return filtered_nodes

# end

# ## Convert gradients to trajectory nodes
# function gradients_to_nodes(gradients::Matrix)

#     nodes = cumsum(gradients, dims = 2)
#     return nodes

# end

# ## Filter gradients using Tullio for efficient convolution
# function filter_gradients(gradients::Matrix, kernel::Matrix)

#     @tullio d[b, i] := gradients[b, i+a-1] * kernel[b, a]
#     return d

# end

# function nodes_to_gradients(nodes::Matrix)
#     newNodes = hcat([0;0],nodes)
#     gradients = diff(newNodes, dims = 2)
#     return gradients

# end

# ## Pad gradients to prepare for Tullio
# function pad_gradients(gradients::Matrix, kernelSize)

#     padding = zeros(Float32,kernelSize[1], kernelSize[2] - 1)
#     padded = hcat(padding, gradients)
#     return padded

# end

# ## Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
# function EHMulx_Tullio(x, nodes::Array{Float64,4}, positions::Matrix{Float64})

#     nodes2 = undoReshape(nodes)

#     @tullio EH[n, k] := exp <| (1.0im * pi * 2.0 * $positions[i, n] * nodes2[i, k])
#     @tullio y[n] := EH[n, k] * $x[k]

#     return y

# end

# ## Weighted Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
# function weighted_EHMulx_Tullio(x, nodes::Matrix{Float64}, positions::Matrix{Float64}, weights::Vector{Float64})

#     # TODO: ADD DENSITY COMPENSATION FUNCTION AS DESCRIBED IN NOLL, FESSLER and SUTTON
#     #@tullio W[k] := sqrt <| $gradients[i,k]*$gradients[i,k] ## Define weights as magnitude of gradients
#     @tullio EH[n, k] := exp <| (1.0im * pi * 2.0 * $positions[i, n] * nodes[i, k])
#     @tullio y[n] := EH[n, k] * (weights[k]*$x[k])

#     return y

# end

# ## Weighted Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
# function weighted_EHMulx_Tullio(x, nodes, positions, weights)

#     # TODO: ADD DENSITY COMPENSATION FUNCTION AS DESCRIBED IN NOLL, FESSLER and SUTTON
#     #@tullio W[k] := sqrt <| $gradients[i,k]*$gradients[i,k] ## Define weights as magnitude of gradients
#     @tullio EH[n, k] := exp <| (1.0im * pi * 2.0 * $positions[i, n] * nodes[i, k])
#     @tullio y[n] := EH[n, k] * (weights[k]*$x[k])

#     return y

# end

# ## Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
# function EHMulx_Tullio(x, nodes::Matrix{Float64}, positions::Matrix{Float64})

#     @tullio EH[n, k] := exp <| (1.0im * pi * 2.0 * $positions[i, n] * nodes[i, k])
#     @tullio y[n] := EH[n, k] * $x[k]

#     return y

# end

# ## Generic Allocator for the E system matrix
# function prepareE(imShape)

#     # construct E in place
#     E = Array{ComplexF32}(undef, imShape[1] * imShape[2], imShape[1] * imShape[2])

#     positions = getPositions(imShape)

#     return E, positions

# end

# ## Memory Efficient Multi-threaded in-place E constructor
# function constructE!(E, nodes::Matrix, positions::Matrix)

#     phi = nodes' * positions

#     Threads.@threads for i in eachindex(phi)
#         E[i] = cispi(-2 * phi[i])
#     end

# end

# ## Memory Efficient Multi-threaded in-place EH constructor
# function constructEH!(EH, nodes::Matrix, positions::Matrix)

#     phi = positions * nodes'

#     Threads.@threads for i in eachindex(phi)
#         EH[i] = cispi(2 * phi[i])
#     end

# end

# ## Single Threaded Explicit Passing Version for Autodiff compat. 
# function EMulx(x, nodes::Matrix, positions::Matrix)

#     E = constructE(nodes, positions)
#     y = E * x
#     return y

# end

# ## Single Threaded Explicit Passing Version for Autodiff compat.
# function EHMulx(x, nodes::Matrix, positions::Matrix)

#     EH = constructEH(nodes, positions)
#     y = EH * x
#     return y

# end

# ## Version of Matrix-Vector Multiplication using Tullio.jl. Supposedly very fast and flexible.
# function EMulx_Tullio(x, nodes::Matrix{Float64}, positions::Matrix{Float64})

#     @tullio E[k, n] := exp <| (-1.0im * pi * 2.0 * nodes[i, k] * $positions[i, n])
#     @tullio y[k] := E[k, n] * $x[n]

#     return y

# end