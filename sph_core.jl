module SPHCore

#  sph_core.jl – NYU CS 421 SPH
#  Original code by Dr. Gizem Kayar in April 2026
#  Refactored into a module so test files can reuse functions
# Assignment completed by Matias Ortiz and Maria BSeatriz Silva

using LinearAlgebra
export Params, Kernel
export default_params
export poly6_kernel, cubic_spline_kernel, wendland_c2_kernel
export init_particles, build_grid, find_neighbors
export compute_density!, compute_pressure!, compute_forces
export integrate_cromer!, integrate_euler!
export xsph!, save_csv, run

# Parameter struct we can pass in for every experiment
struct Params
    w          ::Float64   # block width   (TODO 6/8)
    l          ::Float64   # block height  (TODO 6/8)
    dx         ::Float64   # initial particle spacing
    h          ::Float64   # smoothing length
    mass       ::Float64
    rho0       ::Float64
    k          ::Float64   # stiffness
    mu         ::Float64   # dynamic viscosity
    g          ::Vector{Float64}
    dt         ::Float64
    damp_wall  ::Float64
    damp_floor ::Float64
    steps      ::Int       # total timesteps to run
    save_every ::Int       # write CSV every N steps
end


# Returns default parameters
function default_params(;
        w          = 0.4,
        l          = 0.6,
        h          = 0.04,
        mass       = 1.06,
        rho0       = 1000.0,
        k          = 3000.0,
        mu         = 0.1,
        g          = [0.0, -9.8],
        dt         = 2.0e-5,
        damp_wall  = 0.9,
        damp_floor = 0.85,
        dx         = nothing,   # defaults to h * 0.8; use something() to resolve
        steps      = 100000,
        save_every = 10,
    )
    Params(w, l, something(dx, h * 0.8), h, mass, rho0, k, mu, g, dt, damp_wall, damp_floor, steps, save_every)
end

# Our 3 kernel options for testing
struct Kernel
    name  ::String
    W     ::Function   # W(r, h)          → scalar
    gradW ::Function   # gradW(rvec, h)   → 2-vector  (used for pressure)
    lapW  ::Function   # lapW(r, h)       → scalar    (used for viscosity)
end

_poly6(r, h)    = (0 ≤ r ≤ h) ? 4/(pi*h^8)*(h^2 - r^2)^3 : 0.0

function _grad_spiky(rvec, h)
    r = norm(rvec)
    (0 < r ≤ h) ? -30/(pi*h^5)*(h - r)^2*(rvec/r) : zeros(2)
end

_lap_visc(r, h) = (0 ≤ r ≤ h) ? 20/(3*pi*h^5)*(h - r) : 0.0

poly6_kernel() = Kernel("Poly6 + Spiky (original)", _poly6, _grad_spiky, _lap_visc)

function _cubic_spline(r, h)
    q = r / h
    σ = 10.0 / (7.0*pi*h^2)
    if 0 ≤ q < 1
        return σ * (1 - 1.5*q^2 + 0.75*q^3)
    elseif 1 ≤ q ≤ 2
        return σ * 0.25*(2 - q)^3
    else
        return 0.0
    end
end

function _grad_cubic_spline(rvec, h)
    r = norm(rvec)
    r < 1e-12 && return zeros(2)
    q  = r / h
    σ  = 10.0 / (7.0*pi*h^2)
    dW = if 0 ≤ q < 1
        σ * (-3q + 2.25*q^2)
    elseif 1 ≤ q ≤ 2
        σ * (-0.75*(2 - q)^2)
    else
        0.0
    end
    return (dW / h) * (rvec / r)
end

function _lap_cubic_spline(r, h)
    q = r / h
    σ = 10.0 / (7.0*pi*h^2)
    if 0 ≤ q < 1
        return σ / h^2 * (-3 + 4.5*q)
    elseif 1 ≤ q ≤ 2
        return σ / h^2 * 1.5*(2 - q)
    else
        return 0.0
    end
end

cubic_spline_kernel() = Kernel("Cubic Spline (Monaghan 1992)", _cubic_spline, _grad_cubic_spline, _lap_cubic_spline)

function _wendland_c2(r, h)
    q = r / h
    q > 1.0 && return 0.0
    α = 7.0 / (pi*h^2)
    return α * (1 - q)^4 * (1 + 4q)
end

function _grad_wendland_c2(rvec, h)
    r = norm(rvec)
    r < 1e-12 && return zeros(2)
    q = r / h
    q > 1.0 && return zeros(2)
    α    = 7.0 / (pi*h^2)
    dWdr = α / h * (4*(1-q)^3*(1+4q)*(-1) + (1-q)^4*4)
    return dWdr * (rvec / r)
end

_lap_wendland_c2(r, h) = _lap_visc(r, h)  # visc Laplacian is a good stand-in for comparison

wendland_c2_kernel() = Kernel("Wendland C2", _wendland_c2, _grad_wendland_c2, _lap_wendland_c2)

# -----------------------------
# Initialization (corner dam)
# -----------------------------
function init_particles(p::Params)
    pos = Vector{Vector{Float64}}()
    vel = Vector{Vector{Float64}}()

    # Dense block in lower-left corner
    # TO DO 1 and 8

    nx  = floor(Int, p.w / p.dx)
    ny  = floor(Int, p.l / p.dx)

    for i in 1:nx
        for j in 1:ny
            push!(pos, [i*p.dx, j*p.dx])
            push!(vel, [0.0, 0.0])
        end
    end

    Np  = length(pos)
    rho = zeros(Np)
    P   = zeros(Np)

    return pos, vel, rho, P
end

# -----------------------------
# Grid (spatial hashing)
# -----------------------------
function build_grid(pos, p::Params)
    grid = Dict{Tuple{Int,Int}, Vector{Int}}()
    for i in eachindex(pos)
        cx = clamp(Int(floor(pos[i][1] / p.h)), 0, grid_res(p))
        cy = clamp(Int(floor(pos[i][2] / p.h)), 0, grid_res(p))
        key = (cx, cy)
        if haskey(grid, key)
            push!(grid[key], i)
        else
            grid[key] = [i]
        end
    end
    return grid
end

grid_res(p::Params) = Int(ceil(1.0 / p.h))

# -----------------------------
# Neighbor search
# -----------------------------
function find_neighbors(pos, grid, p::Params)
    neighbors = [Int[] for _ in eachindex(pos)]
    for i in eachindex(pos)
        cx = clamp(Int(floor(pos[i][1] / p.h)), 0, grid_res(p)) # get key for current particle
        cy = clamp(Int(floor(pos[i][2] / p.h)), 0, grid_res(p)) # get key for current particle
        neighbor_cells = [(cx-1, cy-1), (cx, cy-1), (cx+1, cy-1), # define neighbor cells
                          (cx-1, cy),   (cx, cy),   (cx+1, cy),
                          (cx-1, cy+1), (cx, cy+1), (cx+1, cy+1)]
        for cell in neighbor_cells # check all neighbor cells
            if haskey(grid, cell) # if cell is in grid
                append!(neighbors[i], grid[cell]) # add neighbors to current particle
            end
        end
    end
    return neighbors
end

# -----------------------------
# Density
# -----------------------------
function compute_density!(pos, rho, neighbors, p::Params, kern::Kernel)
    for i in eachindex(pos)
        ρ = p.mass * kern.W(0.0, p.h) # initialize density with self contribution
        for j in neighbors[i]
            if i != j
                rij = pos[i] - pos[j]
                r = norm(rij)
                ρ += p.mass * kern.W(r, p.h)
            end
        end
        rho[i] = max(ρ, 1e-6) # prevent division issues
    end
end

# -----------------------------
# Pressure
# -----------------------------
function compute_pressure!(rho, P, p::Params)
    for i in eachindex(rho)
        P[i] = p.k * ((rho[i] / p.rho0)^7 - 1.0)
    end
end

# -----------------------------
# Forces
# -----------------------------
function compute_forces(pos, vel, rho, P, neighbors, p::Params, kern::Kernel)
    forces = [zeros(2) for _ in eachindex(pos)]
    for i in eachindex(pos)
        f_p = zeros(2)
        f_v = zeros(2)
        for j in neighbors[i]
            if i != j
                rij = pos[i] - pos[j]
                r = norm(rij)
                f_p -= p.mass * (P[j] / rho[j]^2 + P[i] / rho[i]^2) * kern.gradW(rij, p.h)
                f_v += p.mu * p.mass * (vel[j] - vel[i]) / rho[j] * kern.lapW(r, p.h)
            end
        end
        forces[i] = f_p + f_v + p.g  # gravity also added
    end
    return forces
end

# -----------------------------
# Integration (Euler-Cromer)
# -----------------------------
function integrate_cromer!(pos, vel, accel, p::Params)
    for i in eachindex(pos)
        vel[i] = vel[i] + p.dt * accel[i]   # update velocity first
        pos[i] = pos[i] + p.dt * vel[i]     # then position with NEW velocity
        if(pos[i][1] < 0.0) # beyond left boundary
            pos[i] = [0.0, pos[i][2]] # reflect
            vel[i] = [-vel[i][1]*p.damp_wall, vel[i][2]] # reflect
        elseif(pos[i][1] > 1.0) # beyond right boundary
            pos[i] = [1.0, pos[i][2]] # reflect
            vel[i] = [-vel[i][1]*p.damp_wall, vel[i][2]] # reflect
        end
        if(pos[i][2] < 0.0) # beyond bottom boundary
            pos[i] = [pos[i][1], 0.0] # reflect
            vel[i] = [vel[i][1], -vel[i][2]*p.damp_floor] # reflect
        end
    end
end

# -----------------------------
# Integration (Euler) -- added for TODO 10
# -----------------------------
function integrate_euler!(pos, vel, accel, p::Params)
    for i in eachindex(pos)
        pos[i] = pos[i] + p.dt * vel[i]     # position uses OLD velocity
        vel[i] = vel[i] + p.dt * accel[i]   # then velocity updated AFTER
        if(pos[i][1] < 0.0) # beyond left boundary
            pos[i] = [0.0, pos[i][2]] # reflect
            vel[i] = [-vel[i][1]*p.damp_wall, vel[i][2]] # reflect
        elseif(pos[i][1] > 1.0) # beyond right boundary
            pos[i] = [1.0, pos[i][2]] # reflect
            vel[i] = [-vel[i][1]*p.damp_wall, vel[i][2]] # reflect
        end
        if(pos[i][2] < 0.0) # beyond bottom boundary
            pos[i] = [pos[i][1], 0.0] # reflect
            vel[i] = [vel[i][1], -vel[i][2]*p.damp_floor] # reflect
        end
    end
end

# -----------------------------
# CSV output
# -----------------------------
function save_csv(pos, vel, rho, P, step)
    outdir = "output"
    isdir(outdir) || mkdir(outdir)
    filename = joinpath(outdir, "sph_$(lpad(step,6,'0')).csv")
    open(filename, "w") do io
        for p in pos
            println(io, "$(p[1]),$(p[2])")
        end
    end
end

# -----------------------------
# XSPH velocity smoothing
# -----------------------------
function xsph!(vel, pos, rho, neighbors, p::Params, kern::Kernel)
    ε = 0.5
    newvel = deepcopy(vel)
    for i in eachindex(pos)
        corr = zeros(2)
        for j in neighbors[i]
            corr += p.mass * (vel[j] - vel[i]) / rho[j] *
                    kern.W(norm(pos[i] - pos[j]), p.h)
        end
        newvel[i] += ε * corr
    end
    for i in eachindex(vel)
        vel[i] = newvel[i]
    end
end

# -----------------------------
# Main loop
# -----------------------------
function run(p::Params, kern::Kernel)

    pos, vel, rho, P = init_particles(p)

    for step in 1:p.steps
        grid     = build_grid(pos, p)
        neighbors = find_neighbors(pos, grid, p)

        compute_density!(pos, rho, neighbors, p, kern)
        compute_pressure!(rho, P, p)
        accel    = compute_forces(pos, vel, rho, P, neighbors, p, kern)
        integrate_cromer!(pos, vel, accel, p)

        if step % p.save_every == 0
            println("Step $step / $(p.steps)")
            save_csv(pos, vel, rho, P, step)
        end
    end

    println("Done (corner breaking dam).")
end

end # module SPHCore
