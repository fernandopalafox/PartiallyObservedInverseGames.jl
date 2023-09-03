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
using Plots
using LinearAlgebra: norm

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
    os = deg2rad(0)
    
    # Hyperplane stuff
    adjacency_matrix = [false true;
                        false false]
    ω = 0.3
    α = 0.0
    ρ = 10.0

    # Costs function parameters
    scale = 100
    t_real = 7.0
    t_real_activate_goalcost = t_real
    weights = repeat([0.9 0.1], outer = n_players)
    
    # Solver  
    μs = [0.1]
    ρmin = 2.0
    max_wall_time = 10.0

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
        ω,
        α,
        ρ,
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
    ω,
    α,
    ρ,
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
    x0 = setup()
    
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
    constraint_parameters = (;adjacency_matrix, ωs = ω, αs = α, ρs = ρ) 
    converged_forward, _, solution_forward, _ = solve_game(
        KKTGameSolverBarrier(),
        control_system,
        player_cost_models,
        x0,
        constraint_parameters,
        T;
        init = (; s = 1.5*scale, x = solution_kkt.x, u = solution_kkt.u),
        solver = Ipopt.Optimizer,
        solver_attributes = (; max_wall_time, print_level = 5),
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
    end

    for μ in μs[2:end]
        converged_forward, _, solution_forward, _ = solve_game(
            KKTGameSolverBarrier(),
            control_system,
            player_cost_models,
            x0,
            constraint_parameters,
            T;
            init = solution_forward,
            solver = Ipopt.Optimizer,
            solver_attributes = (; max_wall_time, print_level = 2),
            μ,
        )

        if !converged_forward
            println("Forward game did not converge at μ = ", μ)
            break
        end
    end

    # Visualize 
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
    ω,
    α,
    ρ,
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
            init = (;s = 1.5*scale, x = solution_kkt.x, u = solution_kkt.u),
            solver = Ipopt.Optimizer,
            solver_attributes = (; max_wall_time, print_level = 5),
            cmin = 1e-5,
            ρmin,
            μ = μs[1],
        )
    for μ in μs[2:end]
        converged_inverse, solution_inverse, model_inverse = solve_inverse_game(
            InverseHyperplaneSolver(),
            y, 
            adjacency_matrix;
            control_system,
            player_cost_models,
            init = solution_inverse,
            solver = Ipopt.Optimizer,
            solver_attributes = (; max_wall_time, print_level = 5),
            cmin = 1e-5,
            ρmin,
            μ,
        )

        if !converged_inverse
            println("Inverse game did not converge at μ = ", μ)
            break
        end
    end

    # ---- Compare inferred and true parameters ----
    if length(findall(adjacency_matrix)) > 0
        println("True parameters:     ", ω, " ", α, " ", ρ)
        println(
            "Inferred parameters: ",
            round(solution_inverse.ωs[1], digits = 2),
            " ",
            round(solution_inverse.αs[1], digits = 2),
            " ",
            round(solution_inverse.ρs[1], digits = 2),
        )
    end

    # ---- Animation trajectories ----
    constraint_parameters = (;adjacency_matrix, ωs = ω, αs = α, ρs = ρ) 

    # Forward
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
end

function mc(trials, game_setup, solution_forward)

    @unpack rng = game_setup

    # ---- Noise levels ----
    noise_levels = unique([0:0.001:0.01; 0.01:0.005:0.03; 0.03:0.01:0.1])
    noise_levels = [0.0]

    # ---- Monte Carlo ----
    println("Starting Monte Carlo for ", length(noise_levels), " noise levels and ", trials, " trials")
    
    # Initialize results array 1x7 empty array of floats
    results = zeros(Float64, 1, 8)

    for noise_level in noise_levels
        println("Noise level: ", noise_level)

        # Assemble noisy observation
        observation = (;
            x = solution_forward.x .+ noise_level * randn(rng, size(solution_forward.x)),
            u = solution_forward.u .+ noise_level * randn(rng, size(solution_forward.u)),
        )

        for i in 1:trials
            # Initialize result vector with NaN
            result = ones(Float64, 1, 8) * NaN

            # Solve inverse game
            converged_inverse, solution_inverse = inverse(observation, game_setup)

            # Compute trajectory reconstruction error
            reconstruction_error = compute_rec_error(solution_forward, solution_inverse, game_setup)

            # Compute cosine similarity between inferred and true parameters
            dissimilarity = compute_dissimilarity(solution_inverse, game_setup)

            # Assemble result matrix
            result = [
                noise_level,
                converged_inverse ? 1.0 : 0.0,
                solution_inverse.ωs[1],
                solution_inverse.αs[1],
                solution_inverse.ρs[1],
                dissimilarity,
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
                " Dissimilarity: ",
                round(result[6], digits = 5),
                " Error: ",
                round(result[7], digits = 5),
                " Time: ",
                round(result[8], digits = 3),
            )
        end
        println("Convergence rate: ", 
        sum(results[results[:, 1] .== noise_level, 2]) / trials,
         " avg dis. for converged: ", sum(results[(results[:, 2] .== 1.0).*(results[:, 1] .== noise_level), 6]) / trials,
         " avg error for converged: ", sum(results[(results[:, 2] .== 1.0).*(results[:, 1] .== noise_level), 7]) / trials,
         " avg time for converged: ",  sum(results[(results[:, 2] .== 1.0).*(results[:, 1] .== noise_level), 8]) / trials)
    end 

    # Remove first row
    results = results[2:end, :]

    # ---- Save results ----
    df = DataFrame(
        noise_level = results[:, 1],
        converged = results[:, 2],
        ω = results[:, 3],
        α = results[:, 4],
        ρ = results[:, 5],
        dissimiliarity = results[:, 6],
        reconstruction_error = results[:, 7],
        time = results[:, 8],
    )
    CSV.write("mcresults.csv", df, header = false)

    plotmc(results, noise_levels, game_setup)

    return results, noise_levels
end

"Plot convergence rate, average reconstruction error, and parameter error vs noise level"
function plotmc(results, noise_levels, game_setup)

    trials = sum(results[:, 1] .== noise_levels[1])

    # Plot results 
    plt1 = plot(
        noise_levels,
        [sum(results[results[:, 1] .== noise_level, 2]) / trials for noise_level in noise_levels],
        xlabel = "Noise level",
        ylabel = "Convergence rate",
        title = "Convergence rate vs noise level",
        legend = false,
    )
    display(plt1)

    # Plot average dissimilarity
    plt3 = plot(
        noise_levels,
        [sum(results[(results[:, 1] .== noise_level) .* (results[:, 2] .== 1.0), 6]) / trials for noise_level in noise_levels],
        xlabel = "Noise level [m]",
        ylabel = "Average dissimilarity error [m]",
        title = "Average dissimilarity error vs noise level",
        legend = false,
    )
    display(plt3)

    # Plot average noise level 
    plt3 = plot(
        noise_levels,
        [sum(results[(results[:, 1] .== noise_level) .* (results[:, 2] .== 1.0), 7]) / trials for noise_level in noise_levels],
        xlabel = "Noise level [m]",
        ylabel = "Average reconstruction error [m]",
        title = "Average reconstruction error vs noise level",
        legend = false,
    )
    display(plt3)
    
    # Cosine similarity for parameters


    return nothing
end

"Compute trajectory reconstruction error. See Peters et al. 2021 experimental section"
function compute_rec_error(solution_forward, solution_inverse, game_setup)
    @unpack n_players, n_states_per_player, T = game_setup

    # Position indicies 
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

"Compute cosine similarity between inferred and true parameters"
function compute_dissimilarity(solution_inverse, game_setup)

    # Parameter vectors 
    truth = [game_setup.ω, game_setup.α, game_setup.ρ]
    estimated = [solution_inverse.ωs[1], solution_inverse.αs[1], solution_inverse.ρs[1]]

    # Compute cosine similarity
    1 -  truth' * estimated / (norm(truth) * norm(estimated))
end