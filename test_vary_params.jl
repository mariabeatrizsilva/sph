include("sph_core.jl")
using .SPHCore
using LinearAlgebra
using Printf, Dates

kinetic_energy(vel, mass) = 0.5 * mass * sum(norm(v)^2 for v in vel)
max_speed(vel) = maximum(norm.(vel))
avg_density(rho) = sum(rho) / length(rho)

# ── Experiment flags ─────────────────────────────────────
const RUN_TODO6 = false
const RUN_TODO7 = false
const RUN_TODO8 = false
const RUN_TODO9 = false
const RUN_TODO10 = true

const TEST_STEPS = 2000

function run_experiment(label::String, p::Params, kern::Kernel;
                        steps=TEST_STEPS, integrator=:cromer, save_frames=false, todo=0)
    pos, vel, rho, P = init_particles(p)
    Np = length(pos)

    prefix = todo > 0 ? "$(todo). " : ""
    outdir = joinpath("output", prefix * replace(label, r"[^\w.-]" => "_"))
    save_frames && (isdir(outdir) || mkdir(outdir))

    density_samples = Float64[]
    t_start = time()

    for step in 1:steps
        grid = build_grid(pos, p)
        nbrs = find_neighbors(pos, grid, p)
        compute_density!(pos, rho, nbrs, p, kern)
        compute_pressure!(rho, P, p)
        accel = compute_forces(pos, vel, rho, P, nbrs, p, kern)

        if integrator == :cromer
            integrate_cromer!(pos, vel, accel, p)
        else
            integrate_euler!(pos, vel, accel, p)
        end

        step % 100 == 0 && push!(density_samples, avg_density(rho))

        save_frames && step % p.save_every == 0 && save_csv(pos, vel, rho, P, step, outdir)
    end

    elapsed = time() - t_start
    mean_rho = isempty(density_samples) ? NaN : sum(density_samples)/length(density_samples)
    final_KE = kinetic_energy(vel, p.mass)
    escaped = sum(x[1] < 0 || x[1] > 1 || x[2] < 0 || x[2] > 1 for x in pos)

    return (
        label         = label,
        kernel        = kern.name,
        Np            = Np,
        elapsed_s     = elapsed,
        final_KE      = final_KE,
        final_max_spd = maximum(norm(v) for v in vel),
        mean_rho      = mean_rho,
        steps_per_sec = steps / elapsed,
        escaped       = escaped,
    )
end

# TODO 9 — naïve O(n²) neighbor search (no spatial hashing)
function find_neighbors_naive(pos, p::Params)
    neighbors = [Int[] for _ in eachindex(pos)]
    for i in eachindex(pos)
        for j in eachindex(pos)
            if norm(pos[i] - pos[j]) < p.h
                push!(neighbors[i], j)
            end
        end
    end
    return neighbors
end

function run_experiment_naive(label::String, p::Params, kern::Kernel; steps=TEST_STEPS)
    pos, vel, rho, P = init_particles(p)
    Np = length(pos)

    density_samples = Float64[]
    t_start = time()

    for step in 1:steps
        # No build_grid — brute force all pairs
        nbrs = find_neighbors_naive(pos, p)
        compute_density!(pos, rho, nbrs, p, kern)
        compute_pressure!(rho, P, p)
        accel = compute_forces(pos, vel, rho, P, nbrs, p, kern)
        integrate_cromer!(pos, vel, accel, p)

        step % 100 == 0 && push!(density_samples, avg_density(rho))
    end

    elapsed = time() - t_start
    mean_rho = isempty(density_samples) ? NaN : sum(density_samples)/length(density_samples)
    final_KE = kinetic_energy(vel, p.mass)
    escaped = sum(x[1] < 0 || x[1] > 1 || x[2] < 0 || x[2] > 1 for x in pos)

    return (
        label         = label,
        kernel        = kern.name,
        Np            = Np,
        elapsed_s     = elapsed,
        final_KE      = final_KE,
        final_max_spd = maximum(norm(v) for v in vel),
        mean_rho      = mean_rho,
        steps_per_sec = steps / elapsed,
        escaped       = escaped,
    )
end

# Formatting helper 
function print_result(r)
    @printf "  %-35s  Np=%3d  t=%.2fs  KE=%.2f  rho=%.1f  escaped=%d  steps/s=%.0f\n" r.label r.Np r.elapsed_s r.final_KE r.mean_rho r.escaped r.steps_per_sec
end

function run_todo6()
    println("TODO 6 - Try different params")
    kern = poly6_kernel()

    experiments = [
        ("Baseline (defaults)",                 default_params()),
        ("High stiffness k=30000",              default_params(k=30000.0)),
        ("Soft fluid k=300",                    default_params(k=300.0)),
        ("High viscosity mu=5.0",               default_params(mu=5.0)),
        ("Inviscid mu=0.0",                     default_params(mu=0.0)),
        ("Strong gravity g=-30",                default_params(g=[0.0,-30.0])),
        ("Large dt=1e-4 (should be unstable?)", default_params(dt=1.0e-4)),
        ("Elastic walls damp=1.0",              default_params(damp_wall=1.0, damp_floor=1.0)),
        ("Sticky walls damp=0.1",               default_params(damp_wall=0.1, damp_floor=0.1)),
        ("Dense packing dx=h/2",               default_params(dx=0.04*0.5)),
    ]

    results = [run_experiment(label, p, kern; save_frames=true, todo=6) for (label, p) in experiments]
    println(); foreach(print_result, results)
end

function run_todo7()
    println("TODO 7 - Try different kernels")
    p = default_params()
    kernels = [poly6_kernel(), cubic_spline_kernel(), wendland_c2_kernel()]
    results = [run_experiment(k.name, p, k; save_frames=true, todo=7) for k in kernels]

    println()
    @printf "  %-32s %5s  %8s  %9s  %6s  %s\n" "Kernel" "Np" "Elapsed" "KE" "rho" "steps/s"
    println("  " * "-"^74)
    for r in results
        @printf "  %-32s %5d  %.3fs   %.4f  %.1f  %.0f\n" r.kernel r.Np r.elapsed_s r.final_KE r.mean_rho r.steps_per_sec
    end
end

function run_todo8()
    println("TODO 8 - Enlarge the initial particle setup")
    kern = poly6_kernel()
    configs = [
        ("Original  w=0.4 l=0.6 dx=h*0.8", default_params()),
        ("Wider     w=0.7 l=0.6",           default_params(w=0.7)),
        ("Taller    w=0.4 l=0.8",           default_params(l=0.8)),
        ("Full half w=0.9 l=0.5",           default_params(w=0.9, l=0.5)),
        ("Full col  w=0.4 l=0.9",           default_params(l=0.9)),
        ("Fine dx   dx=h*0.5",              default_params(dx=0.04*0.5)),
        ("Coarse dx dx=h*1.0",              default_params(dx=0.04*1.0)),
    ]
    results = [run_experiment(label, p, kern; save_frames=true, todo=8) for (label, p) in configs]

    println()
    println("  Config                              Np     Elapsed   KE        rho      steps/s")
    println("  " * "-"^78)
    foreach(print_result, results)
    base = results[1]
    println("\n  Scaling relative to baseline (Np=$(base.Np), t=$(round(base.elapsed_s,digits=3))s):")
    println("  " * "-"^52)
    for r in results[2:end]
        @printf "  %-36s  Np*%.2f  time*%.2f\n" r.label (r.Np/base.Np) (r.elapsed_s/base.elapsed_s)
    end
end

function run_todo9()
    println("TODO 9 - Naïve neighbor search vs spatial hashing")
    kern = poly6_kernel()
    configs = [
        ("Small  Np~216  (default)",  default_params()),
        ("Medium Np~378  w=0.7",      default_params(w=0.7)),
        ("Large  Np~600  dx=h*0.5",   default_params(dx=0.04*0.5)),
    ]
    println()
    @printf "  %-28s  %5s  %10s  %10s  %8s\n" "Config" "Np" "hash sps" "naive sps" "slowdown"
    println("  " * "-"^68)
    for (label, p) in configs
        r_hash = run_experiment(label, p, kern; save_frames=true, todo=9)
        r_naive = run_experiment_naive(label, p, kern)  # naive doesn't save, no point
        slowdown = r_hash.steps_per_sec / r_naive.steps_per_sec
        @printf "  %-28s  %5d  %10.1f  %10.1f  %6.1fx\n" label r_hash.Np r_hash.steps_per_sec r_naive.steps_per_sec slowdown
    end
end

function run_todo10()
    println("TODO 10 - Euler vs Euler-Cromer integrator comparison")
    kern = poly6_kernel()
    configs = [
        ("Smaller   dt=5e-6", default_params()),
        ("Default   dt=2e-5", default_params()),
        ("Larger   dt=5e-5",           default_params(dt=5.0e-5)),
        ("Large    dt=1e-4",           default_params(dt=1.0e-4)),
        ("Very large dt=5e-4", default_params(dt=5.0e-4)),
        ("Extreme  dt=1e-3",   default_params(dt=1.0e-3)),
    ]
    println()
    @printf "  %-30s  %8s  %12s  %12s  %10s  %10s\n" "Config" "integr." "KE" "max_speed" "mean_rho" "steps/s"
    println("  " * "-"^90)
    for (label, p) in configs
        r_cromer = run_experiment(label, p, kern; integrator=:cromer, save_frames=true, todo=10)
        r_euler  = run_experiment(label, p, kern; integrator=:euler,  save_frames=true, todo=10)
        @printf "  %-30s  %8s  %12.4f  %12.4f  %10.1f  %10.1f\n" label "Cromer" r_cromer.final_KE r_cromer.final_max_spd r_cromer.mean_rho r_cromer.steps_per_sec
        @printf "  %-30s  %8s  %12.4f  %12.4f  %10.1f  %10.1f\n" ""     "Euler"  r_euler.final_KE  r_euler.final_max_spd  r_euler.mean_rho  r_euler.steps_per_sec
        println()
    end
end

# ----------
#  Main
# ----------
function main()
    RUN_TODO6  && run_todo6()
    RUN_TODO7  && run_todo7()
    RUN_TODO8  && run_todo8()
    RUN_TODO9  && run_todo9()
    RUN_TODO10 && run_todo10()

    println("\n" * "="^70 * "\n  Done.\n" * "="^70 * "\n")
end

main()