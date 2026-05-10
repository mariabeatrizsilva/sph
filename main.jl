# main.jl
include("sph_core.jl")
using .SPHCore

SPHCore.run(default_params(), poly6_kernel())