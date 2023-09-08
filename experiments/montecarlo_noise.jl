using Revise

const project_root_dir = realpath(joinpath(@__DIR__, ".."))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "test/utils"))))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "experiments/utils"))))

import Ipopt
import TestDynamics

using CollisionAvoidanceGame: CollisionAvoidanceGame
using CSV, DataFrames
using Ipopt: Ipopt
using JuMP: JuMP, @variable, @constraint, @NLconstraint, @objective, set_start_value
using UnPack: @unpack
using PartiallyObservedInverseGames.TrajectoryVisualization: visualize_rotating_hyperplanes, animate_trajectory
using PartiallyObservedInverseGames.CostUtils
using PartiallyObservedInverseGames.JuMPUtils
using PartiallyObservedInverseGames.InverseGames: solve_inverse_game, InverseHyperplaneSolver, InverseWeightSolver
using PartiallyObservedInverseGames.ForwardGame: KKTGameSolver, KKTGameSolverBarrier, solve_game
using Random 
# using Plots
using LinearAlgebra: norm
using Statistics: median
using GLMakie

include("../experiments/utils/misc.jl")

function setup()
    # ---- Parameters ---- 
    
    # Game 
    ΔT = 0.1
    n_players = 2
    n_states_per_player = 4
    
    # Initial state
    v_init = 0.0
    as = [0.0, pi/2]
    # as = [0.0, 2*pi/3, 4*pi/3]
    os = deg2rad(0)
    
    # Hyperplane stuff
    adjacency_matrix = [false true;
                        false false]
    ωs = 0.3
    αs = 0.0
    ρs = 10.0

    # adjacency_matrix = [false true true; 
    #                     false false true;
    #                     false false false]
    # ωs = [0.3, 0.3, 0.3]
    # αs = [0.0, 0.0, 0.0]
    # ρs = [10.0, 10.0, 10.0] 

    # Costs function parameters
    scale = 100
    t_real = 7.0
    t_real_activate_goalcost = t_real
    weights = repeat([0.9 0.1], outer = n_players)
    
    # Solver  
    # μs = [0.1, 0.01, 0.005, 0.001]
    # μs = [1.0, 0.1, 0.01, 0.005, 0.001]
    # μs = [100.0, 10.0, 1.0, 0.1]
    μs = [25.0, 10.0, 1.0, 0.1]
    ρmin = 2.0
    max_wall_time = 15.0

    # noise
    rng = MersenneTwister(0)

    # ---- Setup system and costs ----
    control_system = TestDynamics.ProductSystem([TestDynamics.DoubleIntegrator(ΔT) for _ in 1:n_players])

    T = Int(t_real / ΔT)
    T_activate_goalcost = Int(t_real_activate_goalcost / ΔT)
    player_cost_models = map(enumerate(as)) do (ii, a)
        cost_model_p1 = CollisionAvoidanceGame.generate_hyperintegrator_cost(;
            player_idx = ii,
            control_system,
            T,
            goal_position = scale*unitvector(a),
            weights = (; 
                state_goal      = weights[ii, 1],
                control_Δv      = weights[ii, 2]),
            T_activate_goalcost,
        )
    end

    x0 = vcat(
    [
        vcat(-scale*unitvector(a), [v_init*cos(a - os), v_init*sin(a - os)]) for
        a in as
    ]...,
    )

    return (;
        ΔT,
        n_players,
        n_states_per_player,
        v_init,
        as,
        os,
        adjacency_matrix,
        ωs,
        αs,
        ρs,
        scale,
        t_real,
        t_real_activate_goalcost,
        T,
        weights,
        μs,
        ρmin,
        max_wall_time,
        rng,
        control_system,
        player_cost_models,
        x0
    )
end

function forward(game_setup)

    # Unpack 
    @unpack ΔT,
    n_players,
    n_states_per_player,
    v_init,
    as,
    os,
    adjacency_matrix,
    ωs,
    αs,
    ρs,
    scale,
    t_real,
    t_real_activate_goalcost,
    T, 
    weights,
    μs,
    ρmin,
    max_wall_time,
    rng,
    control_system, 
    player_cost_models,
    x0 = game_setup
    
    # ---- Solve FG ---- 

    # Solve forward game with no hyperplane constraints (just kkt solver)
    kkt_converged, solution_kkt, kkt_model = solve_game(
        KKTGameSolver(),
        control_system,
        player_cost_models,
        x0,
        T;
        solver = Ipopt.Optimizer,
        solver_attributes = (; max_wall_time, print_level = 5),
    )

    # Solve forward game with hyperplane constraints
    constraint_parameters = (;adjacency_matrix, ωs = ωs, αs = αs, ρs = ρs) 
    converged_forward, _, solution_forward, model_forward = solve_game(
        KKTGameSolverBarrier(),
        control_system,
        player_cost_models,
        x0,
        constraint_parameters,
        T;
        # init = (; s = 1.5*scale, x = solution_kkt.x, u = solution_kkt.u),
        init = (; s = 1.5*scale, x = solution_kkt.x, u = solution_kkt.u),
        solver = Ipopt.Optimizer,
        solver_attributes = (; max_wall_time, print_level = 2),
        μ = μs[1],
    )

    if converged_forward == false
        println("Forward game did not converge")
        visualize_rotating_hyperplanes(     
            solution_kkt.x,
            (;
                ΔT = 0.1,
                adjacency_matrix = zeros(Bool, n_players, n_players),
                ωs = constraint_parameters.ωs,
                αs = constraint_parameters.αs,
                ρs = constraint_parameters.ρs,
                n_players = 2,
                n_states_per_player = 4,
                goals = [player_cost_models[i].goal_position for i in 1:n_players],
            );
            title = "forward_kkt",
            koz = true,
            fps = 10.0,
        )
        visualize_rotating_hyperplanes(     
            solution_forward.x,
            (;
                ΔT = 0.1,
                adjacency_matrix = adjacency_matrix,
                ωs = constraint_parameters.ωs,
                αs = constraint_parameters.αs,
                ρs = constraint_parameters.ρs,
                n_players = 2,
                n_states_per_player = 4,
                goals = [player_cost_models[i].goal_position for i in 1:n_players],
            );
            title = "forward_nonconverged",
            koz = true,
            fps = 10.0,
        )
        return nothing
    else
        println("Converged at μ = ", μs[1])
    end

    solution_new = solution_forward
    model_new = model_forward
    for μ in μs[2:end]
        converged_forward, _, solution_forward, model_forward = solve_game(
            KKTGameSolverBarrier(),
            control_system,
            player_cost_models,
            x0,
            constraint_parameters,
            T;
            init = (;model = model_new, solution_new...),
            solver = Ipopt.Optimizer,
            solver_attributes = (; max_wall_time, print_level = 1),
            μ,
        )

        if !converged_forward
            println("Forward game did not converge at μ = ", μ)
            solution_forward = solution_new
            model_forward = model_new
            break
        else
            println("Converged at μ = ", μ)
            solution_new = solution_forward
            model_new = model_forward
        end
    end
    solution_forward = solution_new
    model_forward = model_new

    @assert minimum(solution_forward.s.data) ≥ 0 "Negative slacks"

    # Visualize 
    visualize_rotating_hyperplanes(     
            solution_kkt.x,
            (;
                ΔT = 0.1,
                adjacency_matrix = zeros(Bool, n_players, n_players),
                ωs = constraint_parameters.ωs,
                αs = constraint_parameters.αs,
                ρs = constraint_parameters.ρs,
                n_players = 2,
                n_states_per_player = 4,
                goals = [player_cost_models[i].goal_position for i in 1:n_players],
            );
            title = "forward_kkt",
            koz = true,
            fps = 10.0,
        )
    visualize_rotating_hyperplanes(
            solution_forward.x,
            (;
                ΔT = 0.1,
                adjacency_matrix = adjacency_matrix,
                ωs = constraint_parameters.ωs,
                αs = constraint_parameters.αs,
                ρs = constraint_parameters.ρs,
                n_players = 2,
                n_states_per_player = 4,
                goals = [player_cost_models[i].goal_position for i in 1:n_players],
            );
            title = "forward",
            koz = true,
            fps = 10.0,
        )

    return solution_forward

end

function inverse(observation, game_setup)

    # Unpack parameters
    @unpack ΔT,
    n_players,
    n_states_per_player,
    v_init,
    as,
    os,
    adjacency_matrix,
    ωs,
    αs,
    ρs,
    scale,
    t_real,
    t_real_activate_goalcost,
    T,
    weights,
    μs,
    ρmin,
    max_wall_time,
    rng,
    control_system,
    player_cost_models,
    x0 = game_setup

    # ---- Inverse w/ noise ----

    # Solve forward game to initialize 
    kkt_converged, solution_kkt, _ = solve_game(
        KKTGameSolver(),
        control_system,
        player_cost_models,
        x0,
        T;
        solver = Ipopt.Optimizer,
        solver_attributes = (; max_wall_time, print_level = 1),
    )

    # Solve inverse game
    converged_inverse, solution_inverse, model_inverse = solve_inverse_game(
            InverseHyperplaneSolver(),
            observation, 
            adjacency_matrix;
            control_system,
            player_cost_models,
            init = (;s = 1.5*scale, x = solution_kkt.x, u = solution_kkt.u, λ_e = solution_kkt.λ),
            solver = Ipopt.Optimizer,
            solver_attributes = (; max_wall_time, print_level = 1),
            ρmin,
            μ = μs[1],
        )
    if !converged_inverse
        println("       Inverse game did not converge at μ[1] = ", μs[1])
 
        # ---- Animation of trajectories ----
        constraint_parameters = (;adjacency_matrix, ωs = ωs, αs = αs, ρs = ρs) 
        # Forward noisy
        visualize_rotating_hyperplanes(
                observation.x,
                (;
                    ΔT = 0.1,
                    # adjacency_matrix = zeros(Bool, n_players, n_players),
                    adjacency_matrix = adjacency_matrix,
                    ωs = constraint_parameters.ωs,
                    αs = constraint_parameters.αs,
                    ρs = constraint_parameters.ρs,
                    n_players = 2,
                    n_states_per_player = 4,
                    goals = [player_cost_models[i].goal_position for i in 1:n_players],
                );
                title = "forward_noisy",
                koz = true,
                fps = 10.0,
            )
        # Inverse 
        visualize_rotating_hyperplanes(
                solution_inverse.x,
                (;
                    ΔT = 0.1,
                    adjacency_matrix = adjacency_matrix,
                    ωs = solution_inverse.ωs,
                    αs = solution_inverse.αs,
                    ρs = solution_inverse.ρs,
                    n_players = 2,
                    n_states_per_player = 4,
                    goals = [player_cost_models[i].goal_position for i in 1:n_players],
                );
                title = "inferred",
                koz = true,
                fps = 10.0,
            )

        return converged_inverse, solution_inverse
    else
        # println(
        #     "   Converged at μ = ",
        #     μs[1],
        #     " ω ",
        #     solution_inverse.ωs[1],
        #     " α ",
        #     solution_inverse.αs[1],
        #     " ρ ",
        #     solution_inverse.ρs[1],
        # )
    end

    converged_new = converged_inverse
    solution_new = solution_inverse
    model_new = model_inverse
    for μ in μs[2:end]
        converged_inverse, solution_inverse, model_inverse = solve_inverse_game(
            InverseHyperplaneSolver(),
            observation, 
            adjacency_matrix;
            control_system,
            player_cost_models,
            init = (;model = model_new, solution_new...),
            solver = Ipopt.Optimizer,
            solver_attributes = (; max_wall_time, print_level = 1),
            ρmin,
            μ,
        )

        if !converged_inverse
            println("       Inverse game did not converge at μ = ", μ)
            return converged_new, solution_new
        else
            # println(
            #     "   Converged at μ = ",
            #     μ,
            #     " ω ",
            #     solution_inverse.ωs[1],
            #     " α ",
            #     solution_inverse.αs[1],
            #     " ρ ",
            #     solution_inverse.ρs[1],
            # )
            solution_new = solution_inverse
            model_new = model_inverse
        end
    end
    # converged_inverse = converged_new
    # solution_inverse = solution_new
    # model_inverse = model_new

    # @assert minimum(solution_inverse.s.data) ≥ 0 "Negative slacks"

    # ---- Compare inferred and true parameters ----
    # if length(findall(adjacency_matrix)) > 0
    #     println("True parameters:     ", ωs, " ", αs, " ", ρs)
    #     println(
    #         "Inferred parameters: ",
    #         round(solution_inverse.ωs[1], digits = 2),
    #         " ",
    #         round(solution_inverse.αs[1], digits = 2),
    #         " ",
    #         round(solution_inverse.ρs[1], digits = 2),
    #     )
    # end

    # # ---- Animation of trajectories ----
    # constraint_parameters = (;adjacency_matrix, ωs = ωs, αs = αs, ρs = ρs) 
    # # Forward noisy
    # visualize_rotating_hyperplanes(
    #         observation.x,
    #         (;
    #             ΔT = 0.1,
    #             # adjacency_matrix = zeros(Bool, n_players, n_players),
    #             adjacency_matrix = adjacency_matrix,
    #             ωs = constraint_parameters.ωs,
    #             αs = constraint_parameters.αs,
    #             ρs = constraint_parameters.ρs,
    #             n_players = 2,
    #             n_states_per_player = 4,
    #             goals = [player_cost_models[i].goal_position for i in 1:n_players],
    #         );
    #         title = "forward_noisy",
    #         koz = true,
    #         fps = 10.0,
    #     )
    # # Inverse 
    # visualize_rotating_hyperplanes(
    #         solution_inverse.x,
    #         (;
    #             ΔT = 0.1,
    #             adjacency_matrix = adjacency_matrix,
    #             ωs = solution_inverse.ωs,
    #             αs = solution_inverse.αs,
    #             ρs = solution_inverse.ρs,
    #             n_players = 2,
    #             n_states_per_player = 4,
    #             goals = [player_cost_models[i].goal_position for i in 1:n_players],
    #         );
    #         title = "inferred",
    #         koz = true,
    #         fps = 10.0,
    #     )

    # Check slack 
    if minimum(solution_inverse.s.data') < -0.1
        converged_inverse = false
        println("       negative slacks: ", minimum(solution_inverse.s.data'))
    end

    return converged_inverse, solution_inverse
end

function mc(trials, game_setup, solution_forward)

    @unpack rng = game_setup

    # ---- Noise levels ----
    # noise_levels = unique([0:0.001:0.01; 0.01:0.005:0.03; 0.03:0.01:0.1])
    noise_levels = 0.0:0.1:2.5
    # noise_levels = [2.4, 2.5]
    # noise_levels = [5.0,6.0]
    # noise_levels = 0.0

    # ---- Monte Carlo ----
    println("Starting Monte Carlo for ", length(noise_levels), " noise levels and ", trials, " trials each.")
    println("True parameters: ", game_setup.ωs, " ", game_setup.αs, " ", game_setup.ρs)
    
    # Initialize results array 1x7 empty array of floats
    results = zeros(Float64, 1, 7)

    for noise_level in noise_levels
        println("Noise level: ", noise_level)

        for i in 1:trials
            # Assemble noisy observation
            observation = (;
                x = solution_forward.x .+ noise_level * randn(rng, size(solution_forward.x)),
                u = solution_forward.u .+ noise_level * randn(rng, size(solution_forward.u)),
            )

            # Initialize result vector with NaN
            result = ones(Float64, 1, 7) * NaN

            # Solve inverse game
            converged_inverse, solution_inverse = inverse(observation, game_setup)

            # Compute trajectory reconstruction error
            reconstruction_error = compute_rec_error(solution_forward, solution_inverse, game_setup)

            # Assemble result matrix
            result = [
                noise_level,
                converged_inverse ? 1.0 : 0.0,
                solution_inverse.ωs[1],
                solution_inverse.αs[1],
                solution_inverse.ρs[1],
                reconstruction_error,
                solution_inverse.time,
            ]

            # Append result to results array
            results = vcat(results, result')
            
            # Print progress 
            println("   ",
                lpad(string(noise_level), 6, "0"),
                " Trial: ",
                i,
                "/",
                trials,
                " Converged: ",
                round(result[2], digits = 1),
                " ω: ",
                round(result[3], digits = 3),
                " α: ",
                round(result[4], digits = 3),
                " ρ: ",
                round(result[5], digits = 3),
                " Error: ",
                round(result[6], digits = 5),
                " Time: ",
                round(result[7], digits = 3),
            )

            # Stop if first noise level 
            if noise_level == noise_levels[1] && i == 2
                break
            end
        end

        # Dirty hack 
        if noise_level == noise_levels[1]
            results = results[2:end, :]
        end

        idx_current = results[:, 1] .== noise_level
        idx_converged = (results[:, 1] .== noise_level ) .* (results[:, 2] .== 1.0)
        num_converged = sum(idx_converged)

        println(
            "Convergence rate: ",
            num_converged / trials,
            " error = ",
            sum(results[idx_converged, 6]) / trials,
            " time = ",
            sum(results[idx_converged, 7]) / trials,
        )
    end 

    

    # ---- Save results ----
    df = DataFrame(
        noise_level = results[:, 1],
        converged = results[:, 2],
        ω = results[:, 3],
        α = results[:, 4],
        ρ = results[:, 5],
        reconstruction_error = results[:, 6],
        time = results[:, 7],
    )
    CSV.write("mcresults.csv", df, header = false)

    try
        plotmc(results, noise_levels, game_setup)
    catch
        println("Plotting failed")
    end
    
    return results, noise_levels
end

"Plot convergence rate, average reconstruction error, and parameter error vs noise level"
function plotmc(results, noise_levels, game_setup)

    # Parameters
    color_iqr = :dodgerblue
    set_theme!()

    # Create makie screens 

    # Calculation
    idx_converged = results[:,2] .== 1.0
    trials = sum(results[:, 1] .== noise_levels[1])

    # Plot convergence rate  
    fig_convergence = Figure()
    ax_convergence = Axis(
        fig_convergence[1, 1],
        xlabel = "Noise level [m]",
        ylabel = "Convergence rate [%]",
        # title = "Convergence rate vs noise level",
        limits = (nothing, (0, 110)),
    )
    Makie.barplot!(
        ax_convergence,
        noise_levels,
        [sum(results[results[:, 1] .== noise_level, 2]) / trials * 100 for noise_level in noise_levels],
    )
    # display(fig_convergence)

    # Plot bands for ω
    ω_median = [median(results[(results[:, 1] .== noise_level) .* idx_converged, 3]) for noise_level in noise_levels]
    ρ_median = [median(results[(results[:, 1] .== noise_level) .* idx_converged, 5]) for noise_level in noise_levels]

    ω_iqr = [iqr(results[(results[:, 1] .== noise_level) .* idx_converged, 3]) for noise_level in noise_levels]
    ρ_iqr = [iqr(results[(results[:, 1] .== noise_level) .* idx_converged, 5]) for noise_level in noise_levels]

    fig_bands = Figure()
    ax_ω = Axis(
        fig_bands[1, 1],
        xlabel = "Noise level [m]",
        ylabel = "ω [rad/s]",
        limits = ((noise_levels[1], noise_levels[end]), (0, 2*game_setup.ωs[1])),
    )
    Makie.scatter!(ax_ω, noise_levels, ω_median, color = color_iqr)
    Makie.band!(ax_ω, noise_levels, ω_median .- ω_iqr/2, ω_median .+ ω_iqr/2, color = (color_iqr, 0.2))
    Makie.hlines!(ax_ω, game_setup.ωs[1], color = color_iqr, linewidth = 2, linestyle = :dot)

    ax_ρ = Axis(
        fig_bands[1, 2],
        xlabel = "Noise level [m]",
        ylabel = "ρ [m]",
        limits = ((noise_levels[1], noise_levels[end]), (0, 2*game_setup.ρs[1])),
    )
    Makie.scatter!(ax_ρ, noise_levels, ρ_median, color = color_iqr)
    Makie.band!(ax_ρ, noise_levels, clamp.(ρ_median .- ρ_iqr/2, game_setup.ρmin, Inf), ρ_median .+ ρ_iqr/2, color = (color_iqr, 0.2))
    Makie.hlines!(ax_ρ, game_setup.ρs[1], color = color_iqr, linewidth = 2, linestyle = :dot)
    display(fig_bands)

    # Plot reconstruction error
    fig_error = Figure()
    ax_error = Axis(
        fig_error[1, 1],
        xlabel = "Noise level [m]",
        ylabel = "Reconstruction error [m]",
        # title = "Reconstruction error vs noise level",
    )
    Makie.scatter!(
        ax_error,
        results[idx_converged, 1],
        results[idx_converged, 6],
        markersize = 15,
    )
    # display(fig_error)

    # # Save figures 
    save("figures/mc_noise_convergence.jpg", fig_convergence)
    save("figures/mc_noise_bands.jpg", fig_bands)
    save("figures/mc_noise_error.jpg", fig_error)

    return nothing
end

"Compute trajectory reconstruction error. See Peters et al. 2021 experimental section"
function compute_rec_error(solution_forward, solution_inverse, game_setup)
    @unpack n_players, n_states_per_player, T = game_setup

    # Position indices 
    position_indices = vcat(
            [[1 2] .+ (player - 1) * n_states_per_player for player in 1:(n_players)]...,
        )

    # Compute sum of norms
    reconstruction_error = 0
    for player in 1:n_players
        for t in 1:T
            reconstruction_error += norm(
                solution_forward.x[position_indices[player, :], t] -
                solution_inverse.x[position_indices[player, :], t],
            )
        end
    end

    1/(n_players * T) * reconstruction_error
end

"Compute the interquartile range of a sample. 
Taken from https://turreta.com/blog/2020/03/28/find-interquartile-range-in-julia/"
function iqr(samples)
    samples = sort(samples)

    # Get the size of the samples
    samples_len = length(samples)

    # Divide the size by 2
    sub_samples_len = div(samples_len, 2)

    # Know the indexes
    start_index_of_q1 = 1
    end_index_of_q1 = sub_samples_len
    start_index_of_q3 = samples_len - sub_samples_len + 1
    end_index_of_q3 = samples_len

    # Q1 median value
    median_value_of_q1 = median(view(samples, start_index_of_q1:end_index_of_q1))

    # Q2 median value
    median_value_of_q3 = median(view(samples, start_index_of_q3:end_index_of_q3))

    # Find the IQR value
    iqr_result = median_value_of_q3 - median_value_of_q1
    return iqr_result
end