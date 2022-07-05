# GIRF_estimation

Estimation of the gradient system impulse response function (the perturbing function) by differentiation through the image reconstruction operation.

Code used for JuliaCon 2022 is in girf_estimator.jl. 

Flux requires version 12.0. If you've updated Flux, please type `] add Flux@0.12.0`

It also includes fast explicit reconstruction operators that might be memory inefficient but are trivial to differentiate through. 

Will update as I go along...
