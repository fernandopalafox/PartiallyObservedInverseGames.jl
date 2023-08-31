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
using PartiallyObservedInverseGames.InverseGames: solve_inverse_game, InverseHyperplaneSolver, InverseWeightSolver
using PartiallyObservedInverseGames.ForwardGame: KKTGameSolver, KKTGameSolverBarrier, solve_game
using Random 
using Plots
using LinearAlgebra: norm

include("utils/misc.jl")

# Infer hyperplane parameters from expert data 
function infer()

    # User input
    m   = 100.0 # kg
    r₀ = (400 + 6378.137) # km
    grav_param  = 398600.4418 # km^3/s^2
    n = sqrt(grav_param/(r₀^3)) # rad/s

    ΔT = 0.1
    n_states_per_player = 4
    scale = 1
    ρmin = 0.1

    adjacency_matrix = [false true  true;
                        false false true;
                        false false false]
    # adjacency_matrix = [false true  true  true;
    #                     false false true  true;
    #                     false false false true;
    #                     false false false false]
    μs = [1e-6]
    
    # Load data 
    data_states   = Matrix(CSV.read("data/f_2d_3p_s.csv", DataFrame, header = false))
    data_inputs   = Matrix(CSV.read("data/f_2d_3p_c.csv", DataFrame, header = false))
    y = (;x = data_states, u = data_inputs)

    T = size(data_states, 2)
    T_activate_goalcost = T

    # Setup control system 
    n_players = size(adjacency_matrix, 2)
    control_system = TestDynamics.ProductSystem([TestDynamics.Satellite2D(ΔT, n, m) for _ in 1:n_players])

    # Presumed cost system with dummy variables
    as = [2*pi/n_players * (i-1) for i in 1:n_players]
    as = [a > pi ? a - 2*pi : a for a in as]
    # zs = [1.0, 1.0, 1.0]
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
            init = (;s = 1.5, x = data_states, u = data_inputs),
            solver = Ipopt.Optimizer,
            solver_attributes = (; max_wall_time = 60.0, print_level = 5),
            cmin = 1e-5,
            ρmin,
            μ = μs[1],
        )

    # Display 3d trajectory
    plot_parameters = 
        (;
            n_players = n_players,
            n_states_per_player = 4,
            goals = [player_cost_models[player].goal_position for player in 1:n_players],
            adjacency_matrix, 
            couples = findall(adjacency_matrix),
            ωs = solution_inverse.ωs,
            αs = solution_inverse.αs,
            ρs = solution_inverse.ρs,
            ΔT = ΔT
        )
    # Plot 2d
    visualize_rotating_hyperplanes(
        # vcat(solution_inverse.x[1:4, :], solution_inverse.x[7:10, :], solution_inverse.x[13:16, :]), # dirt hack 
        solution_inverse.x,
        plot_parameters;
        koz = true,
        fps = 10.0,
    )
    # # Plot 23d
    # display_3D_trajectory(
    #     solution_inverse.x,
    #     plot_parameters
    #     ;
    #     title = "3D_hyperplane",
    #     hyperplane = true
    # )

    solution_inverse, plot_parameters
end

# Monte Carlo simulation
function mc(solution_inverse, trials; rng = MersenneTwister(0), μ = 1.0, max_wall_time = 10.0, visualize = false)

    # Tuneable parameters
    scale = 1.0
    v_init = 0.1
    ΔT = 0.1
    T_activate_goalcost = 1
    n_states_per_player = 6
    os = deg2rad(0) # init. angle offset

    # User input
    m   = 100.0 # kg
    r₀ = (400 + 6378.137) # km
    grav_param  = 398600.4418 # km^3/s^2

    n = sqrt(grav_param/(r₀^3)) # rad/s

    # Useful numbers 
    n_players = length(solution_inverse.player_weights)
    T = 100
    T_activate_goalcost = T

    # System 
    control_system = TestDynamics.ProductSystem([TestDynamics.Satellite3D(ΔT, n, m) for _ in 1:n_players])

    # Monte Carlo simulation
    results_x0 = []
    results_adjacency_matrix = []
    results_goals = []
    results_converged = []
    results_solution = []
    results_time = []
    i = 1
    i_sample = 1
    while i <= trials
        

        # Generate initial position and velocity for each player
        as = rand(rng, n_players) .* 2*pi
        # zs_0 = rand(rng, n_players)
        zs_0 = zeros(n_players)
        x0 = vcat(
        [
            vcat(-scale*unitvector(a), zs_0[idx], [v_init*cos(a - os), v_init*sin(a - os), 0]) for
            (idx, a) in enumerate(as)
        ]...,
        )

        # Setup costs and control system 
        # as_goals = rand(rng, n_players) .* 2*pi # random goal for each player
        as_goals = as
        # zs_goals = rand(rng, n_players) # random goal for each player
        zs_goals = ones(n_players)

        # Costs
        weights = [10.0 0.00001;
                   10.0 0.00001; 
                   10.0 0.00001];   

        player_cost_models = map(enumerate(as_goals)) do (ii, a)
            cost_model_p1 = CollisionAvoidanceGame.generate_3dintegrator_cost(;
                player_idx = ii,
                control_system,
                T,
                goal_position = scale*[cos(a), sin(a), zs_goals[ii]],
                weights = (;  
                    state_goal      = weights[ii, 1],
                    control_Δv      = weights[ii, 2]),
                T_activate_goalcost,
            )
        end

        skip = false
        i_sample = i_sample + 1
        println("   Sample counter = $i_sample")
        if i_sample > 50
            break
        end

        # Skip if any two initial positions are closer than maximum KoZ radius
        for i in 1:n_players
            for j in 1:n_players
                if i != j
                    if norm(x0[(1:2) .+ (i - 1)*n_states_per_player] - x0[(1:2) .+ (j - 1)*n_states_per_player]) < 1.2*maximum(solution_inverse.ρs)
                        skip = true
                    end
                end
            end
        end
        # Skip if any two goal positions are closer than maximum KoZ radius
        for i in 1:n_players
            for j in 1:n_players
                if i != j
                    if norm(player_cost_models[i].goal_position[1:2] - player_cost_models[j].goal_position[1:2]) < 1.2*maximum(solution_inverse.ρs)
                        skip = true
                    end
                end
            end
        end

        if skip 
            continue
        end

        adjacency_matrix = [false true  true;
                            false false true;
                            false false false]
        

        # Solve forward game with KKT solver
        _, solution_kkt, _ = 
        solve_game(KKTGameSolver(), control_system, player_cost_models, x0, T; 
        solver = Ipopt.Optimizer, 
        solver_attributes = (; max_wall_time, print_level = 1))

        # Check closest approach and skip if closest approach is greater than smallest KoZ radius
        if minimum(min_distance(solution_kkt.x, (;n_players, n_states_per_player))) > 1.0*minimum(solution_inverse.ρs)
            continue
        end      

        # Solve forward game with inferred hyperplane
        constraint_parameters = (;adjacency_matrix, ωs = solution_inverse.ωs, αs = solution_inverse.αs, ρs = solution_inverse.ρs)
        converged_forward, time_forward, solution_hyp, _ = 
            solve_game(KKTGameSolverBarrier(), control_system, player_cost_models, x0, constraint_parameters, T;
            # init = (;x = solution_kkt.x, u = solution_kkt.u, s = 1.5),
            init = (;s = 1.5),
            solver = Ipopt.Optimizer, 
            solver_attributes = (; max_wall_time = 20.0, print_level = 5),
            μ
            )

        # Visualization 
        if visualize || !converged_forward
            plot_parameters = 
                (;
                    n_players = 3,
                    n_states_per_player = 6,
                    goals = [player_cost_models[player].goal_position for player in 1:n_players],
                    adjacency_matrix,
                    couples = findall(adjacency_matrix),
                    ωs = solution_inverse.ωs,
                    αs = solution_inverse.αs,
                    ρs = solution_inverse.ρs,
                    ΔT = ΔT
                )
            display_3D_trajectory(
                solution_hyp.x,
                plot_parameters
                ;
                title = "Collision avoidance w/ rotating hyperplane",
                hyperplane = false
            )
            trajectory_comparison(
                solution_kkt.x,
                solution_hyp.x,
                plot_parameters
            )
            visualize_rotating_hyperplanes( 
                vcat(solution_hyp.x[1:4, :], solution_hyp.x[7:10, :], solution_hyp.x[13:16, :]), # dirty hack 
                (;
                    n_states_per_player = 4,
                    n_players = 3,
                    ΔT = ΔT,
                    goals = [player_cost_models[player].goal_position for player in 1:n_players],
                    adjacency_matrix,
                    ωs = solution_inverse.ωs,
                    αs = solution_inverse.αs,
                    ρs = solution_inverse.ρs,
                );
                title = "debug",
                koz = true,
                fps = 10.0,
            )
        end

        # Print trial number, whether it converged or not, and the time it took
        println("Trial $i: Converged: $converged_forward, Time: $time_forward")

        # push! trial info into results tuple
        # x0, goals, adjacency_matrix, converged_forward
        push!(results_x0, x0)
        push!(results_solution, solution_hyp)
        push!(results_goals, [player_cost_models[i].goal_position for i in 1:n_players])
        push!(results_adjacency_matrix, adjacency_matrix)
        push!(results_converged, converged_forward)
        push!(results_time, time_forward)

        # Plot segments
        if !converged_forward || true #temporary
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
            display(plt)
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

function min_distance(states, parameters)
    position_indices = vcat(
            [[1 2 3] .+ (player - 1) * parameters.n_states_per_player for player in 1:(parameters.n_players)]...,
        )

    # Create an empty vector of vectors to store the colors
    min_distances = zeros(parameters.n_players)

    for idx_ego in 1:parameters.n_players
        min_distance_to_other = ones(size(states,2)) .* Inf
        pos_ego = states[position_indices[idx_ego, 1:2], :]
        for idx_other in setdiff(1:parameters.n_players, idx_ego)
            pos_other = states[position_indices[idx_other, 1:2], :]
            distance_to_other = map(norm, eachcol(pos_ego .- pos_other))
            min_distance_to_other = vec(minimum(hcat(min_distance_to_other, distance_to_other), dims = 2)) # In Julia 1.9 norm has dim arg 
        end
        # Print player i minimum distance to others
        println("   Player $idx_ego min distance to others: ", minimum(min_distance_to_other))
        min_distances[idx_ego] = minimum(min_distance_to_other)
    end
    return min_distances
end