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
using PartiallyObservedInverseGames.TrajectoryVisualization: visualize_rotating_hyperplanes
using PartiallyObservedInverseGames.CostUtils
using PartiallyObservedInverseGames.JuMPUtils
using PartiallyObservedInverseGames.InverseGames: solve_inverse_game, InverseHyperplaneSolver, InverseWeightSolver
using PartiallyObservedInverseGames.ForwardGame: KKTGameSolverBarrier, solve_game
using Random 
using Plots
using LinearAlgebra: norm

include("utils/misc.jl")

# Infer hyperplane parameters from expert data 
function infer()

    # User input
    m   = 100.0 # kg
    r₀ = (400 + 6378.137) # km
    μ  = 398600.4418 # km^3/s^2
    n = sqrt(μ/(r₀^3)) # rad/s

    ΔT = 0.1
    n_states_per_player = 4 
    scale = 1
    ρmin = 0.2

    adjacency_matrix = [false true;
                        false false]
    μs = [1e-5]
    μs = [1e-5*(1/10)^(i - 1) for i in 1:5]
    
    # Load data 
    data_states   = Matrix(CSV.read("data/f_2d_s.csv", DataFrame, header = false))
    data_inputs   = Matrix(CSV.read("data/f_2d_c.csv", DataFrame, header = false))
    y = (;x = data_states, u = data_inputs)

    T = size(data_states,2)
    T_activate_goalcost = T

    # Setup control system 
    n_players = size(adjacency_matrix, 2)
    control_system = TestDynamics.ProductSystem([TestDynamics.Satellite2D(ΔT, n, m) for _ in 1:n_players])

    # Presumed cost system with dummy variables
    as = [2*pi/n_players * (i-1) for i in 1:n_players]
    as = [a > pi ? a - 2*pi : a for a in as]
    player_cost_models = map(enumerate(as)) do (ii, a)
        cost_model = CollisionAvoidanceGame.generate_hyperintegrator_cost(;
            player_idx = ii,
            control_system,
            T,
            goal_position = scale*[cos(a), sin(a)],
            weights = (; 
                state_goal      = -1,
                control_Δv      = -1),
            T_activate_goalcost,
        )
    end

    # Solve
    (converged, solution_inverse) = solve_inverse_game(
            InverseHyperplaneSolver(),
            y, 
            adjacency_matrix;
            control_system,
            player_cost_models,
            init = (;s = 1.5),
            solver = Ipopt.Optimizer,
            solver_attributes = (; max_wall_time = 60.0, print_level = 5),
            cmin = 1e-5,
            ρmin,
            μ = μs[1],
        )

    # solution_previous = solution_inverse
    # for μ in μs
    #     (converged, solution_intermediate) = solve_inverse_game(
    #         InverseHyperplaneSolver(),
    #         y, 
    #         adjacency_matrix;
    #         control_system,
    #         player_cost_models,
    #         init = (;s = solution_previous.s),
    #         solver = Ipopt.Optimizer,
    #         solver_attributes = (; max_wall_time = 20.0, print_level = 5),
    #         cmin = 1e-5,
    #         ρmin,
    #         μ,
    #     )

    #     if converged
    #         solution_previous = solution_intermediate
    #         println("μ = $μ converged, ωs = $(solution_inverse.ωs), αs = $(solution_inverse.αs), ρs = $(solution_inverse.ρs)")
    #         if μ == μs[end]
    #             solution_inverse = solution_intermediate
    #         end
    #     else
    #         println("μ = $μ did not converge")
    #         break
    #     end
    # end

    # Plot inverse game
    visualize_rotating_hyperplanes(
        solution_inverse.x,
        (;
            ΔT,
            adjacency_matrix,
            ωs = solution_inverse.ωs,
            αs = solution_inverse.αs,
            ρs = solution_inverse.ρs,
            title = "Inverse Sats",
            n_players,
            n_states_per_player,
            goals = [player_cost_models[i].goal_position for i in 1:n_players],
        );
        koz = true,
        fps = 10.0,
    )

    solution_inverse
end

# Monte Carlo simulation
function mc(solution_inverse, trials; rng = MersenneTwister(0), μ = 1.0, max_wall_time = 10.0)

    # Tuneable parameters
    scale = 1.0
    v_init = 0.1
    ΔT = 0.1
    T_activate_goalcost = 1
    n_states_per_player = 4
    os = deg2rad(0) # init. angle offset

    # User input
    m   = 100.0 # kg
    r₀ = (400 + 6378.137) # km
    μ  = 398600.4418 # km^3/s^2
    n = sqrt(μ/(r₀^3)) # rad/s

    # Useful numbers 
    n_players = length(solution_inverse.player_weights)
    T = 100
    T_activate_goalcost = 100

    # System 
    control_system = TestDynamics.ProductSystem([TestDynamics.Satellite2D(ΔT, n, m) for _ in 1:n_players])

    # Monte Carlo simulation
    results_x0 = []
    results_adjacency_matrix = []
    results_goals = []
    results_converged = []
    results_solution = []
    results_time = []
    i = 1
    while i <= trials

        # Generate initial position and velocity for each player
        as = rand(rng, n_players) .* 2*pi
        x0 = vcat(
        [
            vcat(-scale*unitvector(a), [v_init*cos(a - os), v_init*sin(a - os)]) for
            a in as
        ]...,
        )

        # Skip if initial positions closer than smallest KoZ radius
        if norm(x0[1:2,:] - x0[5:6,:]) < 1.2*minimum(solution_inverse.ρs)
            continue
        end

        # Setup costs and control system 
        as_goals = rand(rng, n_players) .* 2*pi # random goal for each player
        player_cost_models = map(enumerate(as_goals)) do (ii, a)
            cost_model = CollisionAvoidanceGame.generate_hyperintegrator_cost(;
                player_idx = ii,
                control_system,
                T,
                goal_position = scale*[cos(a), sin(a)],
                weights = (; 
                    state_goal      = solution_inverse.player_weights[ii][:state_goal],
                    control_Δv      = solution_inverse.player_weights[ii][:control_Δv]),
                T_activate_goalcost,
            )
        end

        # Check for possible collisions
        adjacency_matrix = [false for _ in 1:n_players, _ in 1:n_players]
        for i in 1:n_players
            for j in 1:n_players
                if i != j
                    idx_p1 = (1:2) .+ (i - 1)*n_states_per_player
                    idx_p2 = (1:2) .+ (j - 1)*n_states_per_player
                    if collision_likely(
                        x0[idx_p1],
                        player_cost_models[i].goal_position,
                        x0[idx_p2],
                        player_cost_models[j].goal_position,
                    )
                        if !adjacency_matrix[j, i]
                            # println("   Collision likely for players $i and $j")
                            adjacency_matrix[i, j] = true
                        end
                    end
                end 
            end
        end
        # Only run forward simulation if there is a possible collision
        if !any(adjacency_matrix)
            continue
        end

        # Initialize with regular kkt solution
        _, solution_kkt, _ = 
        solve_game(KKTGameSolver(), control_system, player_cost_models, x0, T; 
        solver = Ipopt.Optimizer, 
        solver_attributes = (; max_wall_time, print_level = 5))

        constraint_parameters = (;adjacency_matrix, ωs = solution_inverse.ωs, αs = solution_inverse.αs, ρs = solution_inverse.ρs)
        converged_forward, time_forward, solution_forward = 
            solve_game(KKTGameSolverBarrier(), control_system, player_cost_models, x0, constraint_parameters, T;
            init = (;x = solution_kkt.x, u = solution_kkt.u, s = 1.5),
            solver = Ipopt.Optimizer, 
            solver_attributes = (; max_wall_time = 20.0, print_level = 5),
            μ
            )

        

        # RUN THIS TO PLOT A SPECIFIC RESULT
        # visualize_rotating_hyperplanes(
        #     results.solutions[2].x,
        #                 (; ΔT = 0.1, adjacency_matrix = results.adjacency_matrix[2], ωs = solution_inverse.ωs, αs = solution_inverse.αs, ρs = solution_inverse.ρs , title = "New FWD", n_players = 2, n_states_per_player = 4, goals = results[2].goals);
        #                             koz = true, fps = 10.0
        # )
        visualize_rotating_hyperplanes(
            solution_kkt.x,
            (;
                ΔT = 0.1,
                adjacency_matrix = zeros(Bool, 2, 2),
                ωs = solution_inverse.ωs,
                αs = solution_inverse.αs,
                ρs = solution_inverse.ρs,
                title = "FWD KKT",
                n_players = 2,
                n_states_per_player = 4,
                goals = [player_cost_models[i].goal_position for i in 1:n_players],
            );
            koz = true,
            fps = 10.0,
        )
        visualize_rotating_hyperplanes(
            solution_forward.x,
            (;
                ΔT = 0.1,
                adjacency_matrix = adjacency_matrix,
                ωs = -solution_inverse.ωs,
                αs = solution_inverse.αs,
                ρs = solution_inverse.ρs,
                title = "FWD Barrier",
                n_players = 2,
                n_states_per_player = 4,
                goals = [player_cost_models[i].goal_position for i in 1:n_players],
            );
            koz = true,
            fps = 10.0,
        )

        # Print trial number, whether it converged or not, and the time it took
        println("Trial $i: Converged: $converged_forward, Time: $time_forward")

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
            plt = plot(title = "Trial $i, Converged = $converged_forward",
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
                        plot!(
                            plt,
                            [x0[1 + (i - 1)*n_states_per_player], player_cost_models[i].goal_position[1]],
                            [x0[2 + (i - 1)*n_states_per_player], player_cost_models[i].goal_position[2]],
                            color = colors[i],
                            label = "$i",
                            )

                        # Player position  
                        scatter!(
                            plt,
                            [x0[1 + (i - 1)*n_states_per_player]],
                            [x0[2 + (i - 1)*n_states_per_player]],
                            color = colors[i],
                            marker = :circle,
                            )
                        annotate!([x0[1 + (i - 1)*n_states_per_player] + 0.2*scale],
                                [x0[2 + (i - 1)*n_states_per_player]],"$i")

                        # goal position
                        scatter!(
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
            # display(plt)
        end

        i = i + 1
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

# Helper functions
function collision_likely(x1, g1, x2, g2)
    # Doesn't work for parallel lines

    # Check orientation of line segments
    o1 = orientation(x1, g1, x2)
    o2 = orientation(x1, g1, g2)
    o3 = orientation(x2, g2, x1)
    o4 = orientation(x2, g2, g1)

    # General case 
    if o1 != o2 && o3 != o4
        return true
    else 
        return false
    end

end

function orientation(p, q, r)
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/
    # for details of below formula.
    val = (q[2] - p[2]) * (r[1] - q[1]) -
          (q[1] - p[1]) * (r[2] - q[2])
    if val > 0
        return 1
    elseif val < 0
        return 2
    else
        return 0
    end

end

function test_intersection()
    p1 = [1, 1]
    q1 = [10, 1]
    p2 = [1, 2]
    q2 = [10, 2]
    
    if collision_likely(p1, q1, p2, q2)
        println("Yes")
    else
        println("No")
    end

    p1 = [10, 0]
    q1 = [0, 10]
    p2 = [0, 0]
    q2 = [10, 10]

    if collision_likely(p1, q1, p2, q2)
        println("Yes")
    else
        println("No")
    end

    p1 = [-5, -5]
    q1 = [0, 0]
    p2 = [1, 1]
    q2 = [10, 10]

    if collision_likely(p1, q1, p2, q2)
        println("Yes")
    else
        println("No")
    end
end