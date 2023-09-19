# Imports
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
using PartiallyObservedInverseGames.TrajectoryVisualization: visualize_rotating_hyperplanes, display_3D_trajectory, trajectory_comparison
using PartiallyObservedInverseGames.CostUtils
using PartiallyObservedInverseGames.JuMPUtils
using PartiallyObservedInverseGames.InverseGames: solve_inverse_game, InverseHyperplaneSolver
using PartiallyObservedInverseGames.ForwardGame: KKTGameSolver, KKTGameSolverBarrier, solve_game
using Random 
using Plots
using LinearAlgebra: norm

include("utils/misc.jl")

"Setup for experiment where we infer hyperplane parameters from expert data and then use them to solve the forward game"
function setup_forward_hyperplanes(;rng = MersenneTwister(0))

    # ---- User settings ----

    # Game settings
    n_players = 3
    n_states_per_player = 4 # in 2D
    scale = 100

    # Load data 
    data_states   = Matrix(CSV.read("data/f_2d_" * string(n_players) * "p_s.csv", DataFrame, header = false))
    data_inputs   = Matrix(CSV.read("data/f_2d_" * string(n_players) * "p_c.csv", DataFrame, header = false))
    observation = (;x = data_states, u = data_inputs)

    # Dynamics settings
    ΔT = 5.0
    m   = 100.0 # kg
    r₀ = (400 + 6378.137) # km
    grav_param  = 398600.4418 # km^3/s^2
    n = sqrt(grav_param/(r₀^3)) # rad/s
    u_max = 1.0
    control_system = TestDynamics.ProductSystem([TestDynamics.Satellite2D(ΔT, n, m, u_max) for _ in 1:n_players])

    # Cost settings 
    weights = repeat([10.0 0.0001], outer = n_players) # from forward game 
    os_init = pi/4 # init. angle offset
    as = [-pi/2 + os_init*(i - 1) for i in 1:n_players] # angles
    T = size(data_states, 2)
    T_activate_goalcost = T
    player_cost_models = map(enumerate(as)) do (ii, a)
        cost_model = CollisionAvoidanceGame.generate_hyperintegrator_cost(;
            player_idx = ii,
            control_system,
            T,
            goal_position = scale*[cos(a), sin(a)],
            weights = (; 
                state_goal      = weights[ii, 1],
                control_Δv      = weights[ii, 2]),
            T_activate_goalcost,
        )
    end

    # Inverse game parameters
    μs = [1000.0, 500.0, 250.0, 100.0, 50.0, 10.0, 5.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.001, 0.0001]
    μs = [500.0, 400.0, 300.0, 200.0, 100.0, 50.0, 10.0, 5.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.001, 0.0001]
    μs = [500.0]
    # μs = [1000.0]
    parameter_bounds = (; ω = (-0.03, 0.0), α = (0,0), ρ = (10.0, scale/2.0))
    regularization_weights = (; α = 0.001, ρ = 0.001)
    adjacency_matrix = zeros(Bool, n_players, n_players)
    for i in 1:n_players
        for j in 1:n_players
            if i < j
                adjacency_matrix[i, j] = true
            end
        end
    end

    # Solver settings 
    max_wall_time = 60.0

    # Monte Carlo settings 
    v_init = 0.0
    max_samples = 50
    mc_scale = 1.0*scale
    mc_n_states_per_player = 6 # in 3D
    mc_control_system = TestDynamics.ProductSystem([TestDynamics.Satellite3D(ΔT, n, m, u_max) for _ in 1:n_players])
    mc_T = Int(1.0*T)
    mc_T_activate_goalcost = mc_T
    mc_μs = [500.0, 100.0, 10.0, 5.0, 1.0]
    mc_μs = [500.0]
    mc_max_wall_time = 5.0
    scale_z = 1.0

    (;
        n_players,
        n_states_per_player,
        scale,
        ΔT,
        T,
        T_activate_goalcost,
        control_system,
        weights,
        player_cost_models,
        adjacency_matrix,
        μs,
        parameter_bounds,
        regularization_weights,
        max_wall_time,
        observation,
        rng,
        v_init, 
        max_samples,
        mc_scale,
        mc_n_states_per_player,
        mc_control_system,
        mc_T,
        mc_T_activate_goalcost,
        mc_μs,
        mc_max_wall_time,
        scale_z
    )
end

"Infer hyperplane parameters from expert data "
function inverse(game_setup)

    # ---- Extract setup ----
    @unpack n_players,
    n_states_per_player,
    scale,
    ΔT,
    T,
    control_system,
    player_cost_models,
    adjacency_matrix,
    μs,
    parameter_bounds,
    regularization_weights,
    max_wall_time,
    observation = game_setup

    # ---- Solve ----

    # Solve forward game to initialize 
    _, solution_kkt, _ = solve_game(
        KKTGameSolver(),
        control_system,
        player_cost_models,
        observation.x[:, 1],
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
        init = (;
            s_hyperplanes = 1.5 * scale,
            x = observation.x,
            u = observation.u,
            λ_dynamics = solution_kkt.λ,
            s_thrust_limits = 1.0,
            λ_thrust_limits = μs[1],
        ),
        solver = Ipopt.Optimizer,
        solver_attributes = (; max_wall_time, print_level = 5),
        μ = μs[1],
        parameter_bounds, 
        regularization_weights,
    )

    # Annealing
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
            solver_attributes = (; max_wall_time, print_level = 5),
            μ,
            parameter_bounds, 
            regularization_weights,
        )

        if !converged_inverse
            println("       Inverse game did not converge at μ = ", μ)
            solution_inverse = solution_new
            model_inverse = model_new
            break
        else
            println(
                "   Converged at μ = ",
                μ,
                " ω ",
                solution_inverse.ωs,
                " α ",
                solution_inverse.αs,
                " ρ ",
                solution_inverse.ρs,
            )
            solution_new = solution_inverse
            model_new = model_inverse
        end
    end

    # Plot 2d
    plot_parameters = 
        (;
            n_players,
            n_states_per_player,
            goals = [player_cost_models[player].goal_position for player in 1:n_players],
            adjacency_matrix, 
            couples = findall(adjacency_matrix),
            ωs = solution_inverse.ωs,
            αs = solution_inverse.αs,
            ρs = solution_inverse.ρs,
            ΔT = ΔT
        )
    constraint_parameters = (;adjacency_matrix, ωs = solution_inverse.ωs, αs = solution_inverse.αs, ρs = solution_inverse.ρs) 
    # Forward noisy
    visualize_rotating_hyperplanes(
            observation.x,
            (;
                ΔT,
                adjacency_matrix = zeros(Bool, n_players, n_players),
                # adjacency_matrix = adjacency_matrix,
                ωs = constraint_parameters.ωs,
                αs = constraint_parameters.αs,
                ρs = constraint_parameters.ρs,
                n_players,
                n_states_per_player,
                goals = [player_cost_models[i].goal_position for i in 1:n_players],
            );
            filename = "expert_pull",
            koz = true,
            fps = 10.0,
            noisy = true,
            save_frame = 25,
        )
    visualize_rotating_hyperplanes(
        solution_inverse.x,
        plot_parameters;
        # title = string(n_players)*"p",
        koz = true,
        fps = 10.0,
        filename = "inverse_pull",
        save_frame = 25,
    )

    solution_inverse
end

# Monte Carlo simulation
function mc(trials, solution_inverse, game_setup; visualize = false)

    # ---- Extract setup ----
    @unpack n_players,
    mc_n_states_per_player,
    mc_scale,
    ΔT,
    weights,
    mc_control_system,
    adjacency_matrix,
    parameter_bounds,
    regularization_weights,
    observation,
    rng, 
    v_init,
    max_samples,
    mc_scale,
    mc_T,
    mc_T_activate_goalcost,
    mc_μs,
    mc_max_wall_time,
    scale_z = game_setup

    n_states_per_player = mc_n_states_per_player
    control_system = mc_control_system
    scale = mc_scale
    T = mc_T
    T_activate_goalcost = mc_T_activate_goalcost
    μs = mc_μs
    max_wall_time = mc_max_wall_time

    # Monte Carlo simulation
    results_x0 = []
    results_adjacency_matrix = []
    results_goals = []
    results_converged = []
    results_solution = []
    results_time = []
    trial_counter = 1
    sample_counter = 0
    while trial_counter <= trials
        
        # Generate initial position and velocity for each player
        as = rand(rng, n_players) .* 2*pi
        zs_0 = (rand(rng, n_players) .* 2*scale .- scale)*scale_z
        x0 = vcat(
        [
            vcat(-scale*unitvector(a), zs_0[idx], [v_init*cos(a), v_init*sin(a), 0]) for
            (idx, a) in enumerate(as)
        ]...,
        )

        # Setup sampled costs and control system 
        as_goals = as
        zs_goals = (rand(rng, n_players) .* 2*scale .- scale)*scale_z # random goal for each player

        # Costs
        player_cost_models = map(enumerate(as_goals)) do (ii, a)
            cost_model_p1 = CollisionAvoidanceGame.generate_3dintegrator_cost(;
                player_idx = ii,
                control_system,
                T,
                goal_position = [scale * cos(a), scale * sin(a), zs_goals[ii]],
                weights = (;  
                    state_goal      = weights[ii, 1],
                    control_Δv      = weights[ii, 2]),
                T_activate_goalcost,
            )
        end

        skip = false
        sample_counter = sample_counter + 1
        if sample_counter > max_samples
            break
        end
        # println("   Sample counter = $sample_counter")

        # Skip sample if initial positions are too close or goals are too close
        couples = findall(adjacency_matrix)
        for (couple_counter, couple) in enumerate(couples)
            p1 = couple[1]
            p2 = couple[2]

            p1_xy_indices = (1 + (p1 - 1) * n_states_per_player):(2 + (p1 - 1) * n_states_per_player)
            p2_xy_indices = (1 + (p2 - 1) * n_states_per_player):(2 + (p2 - 1) * n_states_per_player)

            norm_initial = norm(x0[p1_xy_indices] - x0[p2_xy_indices])

            norm_goal =
                norm(player_cost_models[p1].goal_position - player_cost_models[p2].goal_position)

            # Check initial positions are not closer than corresponding KoZ radius
            if norm_initial < 1.0 * solution_inverse.ρs[couple_counter]
                skip = true
                # println("   Skipping sample: ($p1,$p2) init too close. norm_initial = $norm_initial, ρ = $(solution_inverse.ρs[couple_counter])")
            end

            # Check goals are not too close 
            if norm_goal < 1.0 * solution_inverse.ρs[couple_counter]
                skip = true
                # println("   Skipping sample: ($p1,$p2) goals too close. norm_goal = $norm_goal, ρ = $(solution_inverse.ρs[couple_counter])")
            end
        end

        if skip 
            continue
        else
            sample_counter = 0
        end        

        # Solve forward game with KKT solver
        _, solution_kkt, _ = 
        solve_game(KKTGameSolver(), control_system, player_cost_models, x0, T; 
        solver = Ipopt.Optimizer, 
        solver_attributes = (; max_wall_time, print_level = 1))    

        # Solve forward game with inferred hyperplane
        constraint_parameters = (;adjacency_matrix, ωs = solution_inverse.ωs, αs = solution_inverse.αs, ρs = solution_inverse.ρs)
        converged_forward, time_forward, solution_forward, model_forward = 
            solve_game(KKTGameSolverBarrier(), control_system, player_cost_models, x0, constraint_parameters, T;
            init = (;
                s_hyperplanes = 1.5 * scale,
                x = solution_kkt.x,
                u = solution_kkt.u,
                λ_dynamics = solution_kkt.λ,
                s_thrust_limits = 1.0,
                λ_thrust_limits = μs[1], 
            ),
            solver = Ipopt.Optimizer,
            solver_attributes = (; max_wall_time, print_level = 1),
            μ = μs[1]
            )
        if converged_forward
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
                    println("   Did not converge at μ = ", μ)
                    solution_forward = solution_new
                    model_forward = model_new
                    converged_forward = true
                    break
                else
                    println("   Converged at μ = ", μ)
                    solution_new = solution_forward
                    model_new = model_forward
                end
            end
            solution_forward = solution_new
            model_forward = model_new
        end

        # Visualization 
        if visualize || !converged_forward
            plot_parameters = 
                (;
                    n_players,
                    n_states_per_player,
                    goals = [player_cost_models[player].goal_position for player in 1:n_players],
                    adjacency_matrix,
                    couples = findall(adjacency_matrix),
                    ωs = constraint_parameters.ωs,
                    αs = constraint_parameters.αs,
                    ρs = constraint_parameters.ρs,
                    ΔT = ΔT
                )
            display_3D_trajectory(
                solution_forward.x,
                plot_parameters
                ;
                title = "Collision avoidance w/ rotating hyperplane",
                filename = "3D_debug_"*string(trial_counter)*".gif",
                hyperplane = false
            )
            trajectory_comparison(
                solution_kkt.x,
                solution_forward.x,
                plot_parameters
            )
            visualize_rotating_hyperplanes( 
                vcat(solution_forward.x[1:4, :], solution_forward.x[7:10, :], solution_forward.x[13:16, :]), # dirty hack 
                (;
                    n_states_per_player = 4,
                    n_players,
                    ΔT = ΔT,
                    plot_parameters.goals,
                    adjacency_matrix = plot_parameters.adjacency_matrix,
                    ωs = plot_parameters.ωs,
                    αs = plot_parameters.αs,
                    ρs = plot_parameters.ρs,
                );
                title = "debug_"*string(trial_counter),
                koz = true,
                fps = 10.0,
            )
        end

        # Print trial number, whether it converged or not, and the time it took
        println("Trial $trial_counter: Converged: $converged_forward, Time: $time_forward")

        # push! trial info into results tuple
        # x0, goals, adjacency_matrix, converged_forward
        push!(results_x0, x0)
        push!(results_solution, solution_forward)
        push!(results_goals, [player_cost_models[i].goal_position for i in 1:n_players])
        push!(results_adjacency_matrix, adjacency_matrix)
        push!(results_converged, converged_forward)
        push!(results_time, time_forward)

        # Plot segments
        if !converged_forward 
            plt = Plots.plot(title = "Trial $trial_counter, Converged = $converged_forward",
                        xlabel = "x",
                        ylabel = "y",
                        aspect_ratio = :equal,
                        legend = :none,
                        )
            colors = palette(:default)[1:(n_players)]
            for i in 1:n_players
                for j in 1:n_players
                    if i != j
                        
                        # Line segment 
                        Plots.plot!(
                            plt,
                            [x0[1 + (i - 1)*n_states_per_player], player_cost_models[i].goal_position[1]],
                            [x0[2 + (i - 1)*n_states_per_player], player_cost_models[i].goal_position[2]],
                            color = colors[i],
                            label = "$i",
                            )

                        # Player position  
                        Plots.scatter!(
                            plt,
                            [x0[1 + (i - 1)*n_states_per_player]],
                            [x0[2 + (i - 1)*n_states_per_player]],
                            color = colors[i],
                            marker = :circle,
                            )
                        Plots.annotate!([x0[1 + (i - 1)*n_states_per_player] + 0.2*mc_scale],
                                [x0[2 + (i - 1)*n_states_per_player]],"$i")

                        # goal position
                        Plots.scatter!(
                            plt,
                            [player_cost_models[i].goal_position[1]],
                            [player_cost_models[i].goal_position[2]],
                            color = colors[i],
                            label = "$i",
                            markershape = :xcross,
                            markerstrokewidth = 5,
                            ) 
                    end
                end
            end
            display(plt)
        end

        trial_counter = trial_counter + 1
    end

    results = (;
        x0 = results_x0,
        solutions = results_solution,
        goals = results_goals,
        adjacency_matrix = results_adjacency_matrix,
        converged = results_converged,
        time = results_time,
    )

    # Display avg time across all trials and convergence rate
    println("DONE")
    println("Average solver time: $(sum(results.time)/trials)")
    println("Convergence rate: $(sum(results.converged)/trials)")

    return results
end