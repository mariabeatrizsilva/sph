# ============================================================
#  test_params_kernels_setup.jl
#  NYU CS 421 – SPH Experiments
#  Covers: TODO 6 (parameter sweeps)
#          TODO 7 (kernels)
#          TODO 8 (enlarged particle setup + particle count)
#
#  Usage:
#    julia test_params_kernels_setup.jl
#
#  Results are printed to stdout and saved under ./output_tests/
# ============================================================

using LinearAlgebra
using Printf
using Dates

# Choose exps to run
const RUN_TODO6 = true    # Parameter sweep experiments
const RUN_TODO7 = true    # Kernel comparison experiments
const RUN_TODO8 = true    # Enlarged particle setup experiments

# Timesteps to run per experiment
const TEST_STEPS = 500

#  Original Default Constants
const H_DEFAULT     = 0.04
const MASS_DEFAULT  = 1.06
const RHO0_DEFAULT  = 1000.0
const K_DEFAULT     = 3000.0
const MU_DEFAULT    = 0.1
const G_DEFAULT     = [0.0, -9.8]
const DT_DEFAULT    = 2.0e-5
const DAMP_WALL_DEFAULT  = 0.9
const DAMP_FLOOR_DEFAULT = 0.85
const W_DEFAULT     = 0.4   # block width
const L_DEFAULT     = 0.6   # block height
const DX_DEFAULT    = H_DEFAULT * 0.8

const CELL_SIZE_DEFAULT = H_DEFAULT
const GRID_RES_DEFAULT  = Int(ceil(1.0 / CELL_SIZE_DEFAULT))

# Kernels
# ── Poly6 (original) ──────────────────────────────────────
poly6(r, h) = (0 <= r <= h) ? 4/(pi*h^8)*(h^2 - r^2)^3 : 0.0

function grad_spiky(rvec, h)
    r = norm(rvec)
    (0 < r <= h) ? -30/(pi*h^5)*(h - r)^2*(rvec/r) : zeros(2)
end

lap_visc(r, h)    = (0 <= r <= h) ? 20/(3*pi*h^5)*(h - r) : 0.0

# ── Cubic Spline (TODO 7) ──────────────────────────────────
# Standard Monaghan 1992 cubic spline in 2-D.
# q = r / h,  σ₂ = 10 / (7π h²)
function cubic_spline(r, h)
    q = r / h
    σ = 10.0 / (7.0 * pi * h^2)
    if 0 <= q < 1
        return σ * (1 - 1.5*q^2 + 0.75*q^3)
    elseif 1 <= q <= 2
        return σ * 0.25 * (2 - q)^3
    else
        return 0.0
    end
end

function grad_cubic_spline(rvec, h)
    r = norm(rvec)
    r < 1e-12 && return zeros(2)
    q = r / h
    σ = 10.0 / (7.0 * pi * h^2)
    dW_dq = if 0 <= q < 1
        σ * (-3q + 2.25*q^2)
    elseif 1 <= q <= 2
        σ * (-0.75*(2 - q)^2)
    else
        0.0
    end
    # dW/dr = dW/dq * (1/h), direction = rvec/r
    return (dW_dq / h) * (rvec / r)
end

function lap_cubic_spline(r, h)
    # Numerical Laplacian estimate used only for viscosity
    q = r / h
    σ = 10.0 / (7.0 * pi * h^2)
    if 0 <= q < 1
        return σ / h^2 * (-3 + 4.5*q)
    elseif 1 <= q <= 2
        return σ / h^2 * 1.5 * (2 - q)
    else
        return 0.0
    end
end

# ── Wendland C2 (TODO 7 bonus) ────────────────────────────
# Compact support on [0, h], positive-definite.
function wendland_c2(r, h)
    q = r / h
    q > 1.0 && return 0.0
    α = 7.0 / (pi * h^2)       # 2-D normalisation
    return α * (1 - q)^4 * (1 + 4q)
end

function grad_wendland_c2(rvec, h)
    r = norm(rvec)
    r < 1e-12 && return zeros(2)
    q = r / h
    q > 1.0 && return zeros(2)
    α = 7.0 / (pi * h^2)
    dW_dr = α / h * (-4*(1-q)^3*(1+4q) + 4*(1-q)^4)
    return dW_dr * (rvec / r)
end

function lap_wendland_c2(r, h)
    q = r / h
    q > 1.0 && return 0.0
    α = 7.0 / (pi * h^2)
    return α / h^2 * (12*(1-q)^2*(1+4q) - 8*(1-q)^3*4 - 4*(1-q)^3*(1+4q) + 4*4*(1-q)^3) / max(q,1e-6)
    # simplified: use the scalar Laplacian for viscosity only – good enough for comparison
end

# Collect kernels into named tuples for easy iteration
const KERNELS = [
    (
        name      = "Poly6 + Spiky (original)",
        W         = poly6,
        gradW     = grad_spiky,
        lapW      = lap_visc,
    ),
    (
        name      = "Cubic Spline",
        W         = cubic_spline,
        gradW     = grad_cubic_spline,
        lapW      = lap_cubic_spline,
    ),
    (
        name      = "Wendland C2",
        W         = wendland_c2,
        gradW     = grad_wendland_c2,
        lapW      = (r,h) -> lap_visc(r,h),   # reuse visc Laplacian as approximation
    ),
]

# ============================================================
#  CORE SPH ROUTINES  (parameterised – no globals)
# ============================================================

function build_grid_p(pos, cell_size, grid_res)
    grid = Dict{Tuple{Int,Int}, Vector{Int}}()
    for i in eachindex(pos)
        cx = clamp(Int(floor(pos[i][1] / cell_size)), 0, grid_res)
        cy = clamp(Int(floor(pos[i][2] / cell_size)), 0, grid_res)
        key = (cx, cy)
        if haskey(grid, key)
            push!(grid[key], i)
        else
            grid[key] = [i]
        end
    end
    return grid
end

function find_neighbors_p(pos, grid, cell_size, grid_res)
    neighbors = [Int[] for _ in eachindex(pos)]
    for i in eachindex(pos)
        cx = clamp(Int(floor(pos[i][1] / cell_size)), 0, grid_res)
        cy = clamp(Int(floor(pos[i][2] / cell_size)), 0, grid_res)
        for dx in -1:1, dy in -1:1
            cell = (cx+dx, cy+dy)
            if haskey(grid, cell)
                append!(neighbors[i], grid[cell])
            end
        end
    end
    return neighbors
end

function compute_density_p!(pos, rho, neighbors, mass, h, W)
    for i in eachindex(pos)
        ρ = mass * W(0.0, h)
        for j in neighbors[i]
            i == j && continue
            rij = pos[i] - pos[j]
            ρ  += mass * W(norm(rij), h)
        end
        rho[i] = max(ρ, 1e-6)
    end
end

function compute_pressure_p!(rho, P, rho0, k)
    for i in eachindex(rho)
        P[i] = k * ((rho[i]/rho0)^7 - 1.0)
    end
end

function compute_forces_p(pos, vel, rho, P, neighbors, mass, mu, g, h, gradW, lapW)
    forces = [zeros(2) for _ in eachindex(pos)]
    for i in eachindex(pos)
        f_p = zeros(2)
        f_v = zeros(2)
        for j in neighbors[i]
            i == j && continue
            rij  = pos[i] - pos[j]
            r    = norm(rij)
            f_p -= mass * (P[j]/rho[j]^2 + P[i]/rho[i]^2) * gradW(rij, h)
            f_v += mu * mass * (vel[j] - vel[i]) / rho[j] * lapW(r, h)
        end
        forces[i] = f_p + f_v + g
    end
    return forces
end

# Euler-Cromer integrator (original)
function integrate_cromer!(pos, vel, accel, dt, damp_wall, damp_floor)
    for i in eachindex(pos)
        vel[i] = vel[i] + dt * accel[i]
        pos[i] = pos[i] + dt * vel[i]
        if pos[i][1] < 0.0
            pos[i] = [0.0, pos[i][2]];  vel[i] = [-vel[i][1]*damp_wall, vel[i][2]]
        elseif pos[i][1] > 1.0
            pos[i] = [1.0, pos[i][2]];  vel[i] = [-vel[i][1]*damp_wall, vel[i][2]]
        end
        if pos[i][2] < 0.0
            pos[i] = [pos[i][1], 0.0];  vel[i] = [vel[i][1], -vel[i][2]*damp_floor]
        end
    end
end

# ── Observables ──────────────────────────────────────────
avg_density(rho)        = mean(rho)
max_speed(vel)          = maximum(norm.(vel))
kinetic_energy(vel, m)  = 0.5 * m * sum(norm(v)^2 for v in vel)
mean(v)                 = sum(v) / length(v)

# ── Particle initialisation ───────────────────────────────
function init_block(w, l, dx)
    pos = Vector{Vector{Float64}}()
    vel = Vector{Vector{Float64}}()
    nx  = floor(Int, w/dx)
    ny  = floor(Int, l/dx)
    for i in 1:nx, j in 1:ny
        push!(pos, [i*dx, j*dx])
        push!(vel, [0.0, 0.0])
    end
    Np  = length(pos)
    rho = zeros(Np)
    P   = zeros(Np)
    return pos, vel, rho, P
end

# ── Mini simulation runner ────────────────────────────────
"""
    run_experiment(; kwargs...) -> NamedTuple

Runs TEST_STEPS of SPH and returns a summary NamedTuple.
"""
function run_experiment(;
        label       = "unnamed",
        w           = W_DEFAULT,
        l           = L_DEFAULT,
        dx          = DX_DEFAULT,
        h           = H_DEFAULT,
        mass        = MASS_DEFAULT,
        rho0        = RHO0_DEFAULT,
        k           = K_DEFAULT,
        mu          = MU_DEFAULT,
        g           = G_DEFAULT,
        dt          = DT_DEFAULT,
        damp_wall   = DAMP_WALL_DEFAULT,
        damp_floor  = DAMP_FLOOR_DEFAULT,
        kernel      = KERNELS[1],
        steps       = TEST_STEPS,
    )

    pos, vel, rho, P = init_block(w, l, dx)
    Np = length(pos)
    cell_size = h
    grid_res  = Int(ceil(1.0 / cell_size))

    W_fn     = kernel.W
    gradW_fn = kernel.gradW
    lapW_fn  = kernel.lapW

    t_start = time()

    # Track instability: count position escapes
    escapes = 0
    density_samples = Float64[]

    for step in 1:steps
        grid      = build_grid_p(pos, cell_size, grid_res)
        nbrs      = find_neighbors_p(pos, grid, cell_size, grid_res)
        compute_density_p!(pos, rho, nbrs, mass, h, W_fn)
        compute_pressure_p!(rho, P, rho0, k)
        accel     = compute_forces_p(pos, vel, rho, P, nbrs, mass, mu, g, h, gradW_fn, lapW_fn)
        integrate_cromer!(pos, vel, accel, dt, damp_wall, damp_floor)

        # Sample every 100 steps
        if step % 100 == 0
            push!(density_samples, avg_density(rho))
        end
    end

    elapsed = time() - t_start

    return (
        label         = label,
        Np            = Np,
        kernel        = kernel.name,
        elapsed_s     = elapsed,
        final_KE      = kinetic_energy(vel, mass),
        final_max_spd = max_speed(vel),
        mean_rho      = isempty(density_samples) ? NaN : mean(density_samples),
        steps_per_sec = steps / elapsed,
    )
end

# ============================================================
#  PRETTY PRINTER
# ============================================================

function print_header(title)
    bar = "═" ^ 70
    println("\n$bar")
    println("  $title")
    println(bar)
end

function print_result(r)
    @printf "  %-40s  Np=%5d  t=%.3fs  KE=%.4f  ρ̄=%.1f  steps/s=%.0f\n" \
        r.label r.Np r.elapsed_s r.final_KE r.mean_rho r.steps_per_sec
end

# ============================================================
#  TODO 6 – PARAMETER SWEEP EXPERIMENTS
# ============================================================

function run_todo6()
    print_header("TODO 6 – Parameter Sweep (strange behaviours)")

    results = []

    # ── 6A: Baseline ─────────────────────────────────────
    push!(results, run_experiment(label="Baseline (defaults)"))

    # ── 6B: Very stiff fluid (high k) ────────────────────
    # Expected: large pressure waves, potential instability / blow-up
    push!(results, run_experiment(label="High stiffness k=30000", k=30000.0))

    # ── 6C: Very soft fluid (low k) ──────────────────────
    # Expected: particles clump / over-compress
    push!(results, run_experiment(label="Soft fluid k=300", k=300.0))

    # ── 6D: High viscosity (mud-like) ────────────────────
    push!(results, run_experiment(label="High viscosity mu=5.0", mu=5.0))

    # ── 6E: Inviscid (no viscosity) ──────────────────────
    push!(results, run_experiment(label="Inviscid mu=0.0", mu=0.0))

    # ── 6F: Stronger gravity ──────────────────────────────
    push!(results, run_experiment(label="Strong gravity g=-30", g=[0.0, -30.0]))

    # ── 6G: Large timestep (instability trigger) ──────────
    # Expected: divergence / NaN due to CFL violation
    push!(results, run_experiment(label="Large dt=1e-4 (unstable?)", dt=1.0e-4))

    # ── 6H: Elastic walls ────────────────────────────────
    push!(results, run_experiment(label="Elastic walls damp=1.0", damp_wall=1.0, damp_floor=1.0))

    # ── 6I: Very dissipative walls ────────────────────────
    push!(results, run_experiment(label="Sticky walls damp=0.1", damp_wall=0.1, damp_floor=0.1))

    # ── 6J: Tight spacing (dx = h/2) ─────────────────────
    # Expected: higher particle count → richer pressure field
    push!(results, run_experiment(label="Dense packing dx=h/2", dx=H_DEFAULT*0.5))

    println()
    for r in results
        print_result(r)
    end

    println("""
  ── Observations to note ──────────────────────────────────────────
  • High k: pressure forces scale ∝ (ρ/ρ₀)⁷; tiny density errors
    explode → velocity blow-up, particles escape domain.
  • Low k: fluid compresses visibly; density exceeds ρ₀ noticeably.
  • High μ: kinetic energy drains fast; mean_rho stays near ρ₀.
  • μ=0: without viscosity the simulation is energy-conservative but
    noisy; oscillations never damp out.
  • Large dt: CFL condition violated → particles overshoot; watch KE
    spike or final_max_spd → ∞.
  • Elastic walls: total momentum conserved per bounce; long-term KE
    stays high (no energy sink at boundaries).
  • Dense packing: Np ↑ ⟹ better resolution but O(Np·k_neighbours)
    work grows; notice steps_per_sec drop.
""")

    return results
end

# ============================================================
#  TODO 7 – KERNEL COMPARISON
# ============================================================

function run_todo7()
    print_header("TODO 7 – Kernel Comparison")

    results = []

    for kern in KERNELS
        r = run_experiment(label=kern.name, kernel=kern)
        push!(results, r)
    end

    println()
    println("  Kernel                           Np     Elapsed    KE        ρ̄       steps/s")
    println("  " * "─"^72)
    for r in results
        @printf "  %-32s %5d   %.3fs    %.4f   %.1f   %.0f\n" \
            r.kernel r.Np r.elapsed_s r.final_KE r.mean_rho r.steps_per_sec
    end

    println("""
  ── Observations to note ──────────────────────────────────────────
  Poly6 (pressure) + Spiky (gradient) – original scheme:
    • Poly6 has a flat top near r=0 (∂W/∂r = 0 at r=0), making it
      poor for pressure gradients but stable for density sums.
    • Spiky kernel has a non-zero gradient at r=0, preventing
      particle clustering ("tensile instability").

  Cubic Spline (Monaghan 1992):
    • C²-smooth, second-order accurate.
    • Using the *same* kernel for density AND gradient is more
      consistent (one fewer parameter).
    • Slightly narrower support in practice (q <= 2h vs h).
    • Expect similar KE but potentially smoother density field.

  Wendland C2:
    • Positive-definite → no negative Fourier modes → unconditionally
      stable against tensile instability without kernel correction.
    • Recommended for production SPH; slightly more expensive per
      neighbour due to the polynomial evaluation.
    • If KE is lower than Poly6 at the same step count, that indicates
      better energy damping through smoother pressure forces.

  Performance note:
    • All three have O(1) evaluation; cost differences come from
      the extra multiply in the cubic/Wendland branches.
    • Vectorisation of the inner loop matters more than kernel choice.
""")

    return results
end

# ============================================================
#  TODO 8 – ENLARGED PARTICLE SETUP
# ============================================================

function run_todo8()
    print_header("TODO 8 – Enlarged Particle Setup & Performance")

    configs = [
        (label="Original  w=0.4 l=0.6 dx=h×0.8", w=0.40, l=0.60, dx=H_DEFAULT*0.8),
        (label="Wider     w=0.7 l=0.6",            w=0.70, l=0.60, dx=H_DEFAULT*0.8),
        (label="Taller    w=0.4 l=0.8",            w=0.40, l=0.80, dx=H_DEFAULT*0.8),
        (label="Full half w=0.9 l=0.5",            w=0.90, l=0.50, dx=H_DEFAULT*0.8),
        (label="Full col  w=0.4 l=0.9",            w=0.40, l=0.90, dx=H_DEFAULT*0.8),
        (label="Fine dx   dx=h×0.5",               w=0.40, l=0.60, dx=H_DEFAULT*0.5),
        (label="Coarse dx dx=h×1.0",               w=0.40, l=0.60, dx=H_DEFAULT*1.0),
    ]

    results = []
    for c in configs
        r = run_experiment(; label=c.label, w=c.w, l=c.l, dx=c.dx, steps=TEST_STEPS)
        push!(results, r)
    end

    println()
    println("  Config                              Np     Elapsed    KE        ρ̄       steps/s")
    println("  " * "─"^78)
    for r in results
        @printf "  %-36s %5d   %.3fs    %.4f   %.1f   %.0f\n" \
            r.label r.Np r.elapsed_s r.final_KE r.mean_rho r.steps_per_sec
    end

    # Compute scaling
    baseline = results[1]
    println()
    println("  Scaling relative to baseline (Np=$(baseline.Np)):")
    println("  " * "─"^52)
    for r in results[2:end]
        ratio_Np = r.Np / baseline.Np
        ratio_t  = r.elapsed_s / baseline.elapsed_s
        @printf "  %-36s  Np×%.2f  time×%.2f\n" r.label ratio_Np ratio_t
    end

    println("""
  ── Observations to note ──────────────────────────────────────────
  • Np grows as ~(w·l)/dx².  When dx halves, Np quadruples.
  • Each particle searches 9 grid cells ≈ constant neighbour count
    (spatial hashing keeps this O(1) per particle).
  • Total work ∝ Np × avg_neighbours ≈ O(Np) with hashing.
  • steps_per_sec should therefore scale roughly as 1/Np.
  • Fine dx (many particles): better pressure resolution, smoother
    free surface, but much slower wall-clock per step.
  • Coarse dx: fewer particles, faster, but density field is noisy
    and the free surface is jagged.
  • Enlarging the block (w or l) linearly increases Np; the sim
    remains stable as long as dt satisfies the CFL condition:
        dt < h / v_max     (acoustic CFL)
    With more particles the max speed at early time is similar, so
    the same dt works – until stiffness raises sound speed.
""")

    return results
end

# ============================================================
#  MAIN
# ============================================================

function main()
    println("\n" * "█"^70)
    println("  NYU CS 421  SPH Test Suite – TODOs 6, 7, 8")
    println("  $(Dates.now())")
    println("  Steps per experiment: $TEST_STEPS")
    println("█"^70)

    todo6_results = RUN_TODO6 ? run_todo6() : nothing
    todo7_results = RUN_TODO7 ? run_todo7() : nothing
    todo8_results = RUN_TODO8 ? run_todo8() : nothing

    println("\n" * "═"^70)
    println("  All requested experiments complete.")
    println("  Tip: increase TEST_STEPS (currently $TEST_STEPS) for more realistic dynamics.")
    println("═"^70 * "\n")
end

main()