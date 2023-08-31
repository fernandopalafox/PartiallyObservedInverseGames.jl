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

function generate_forward_game() 

    ΔT = 0.1
    n_players = 2
    scale = 1
    t_real = 4.0
    T_activate_goalcost = 1

    v_init = 0.5
    os = deg2rad(90) # init. angle offset
    max_wall_time = 60.0

    # Setup system
    control_system = TestDynamics.ProductSystem([TestDynamics.DoubleIntegrator(ΔT) for _ in 1:n_players])
    as = [2*pi/n_players * (i-1) for i in 1:n_players] # angles
    as = [a > pi ? a - 2*pi : a for a in as]

    x0 = vcat(
    [
        vcat(-scale*unitvector(a), [v_init*cos(a - os), v_init*sin(a - os)]) for
        a in as
    ]...,
    )

    # Costs
    weights = [0.05 100.0 10.0;
               0.05 100.0 10.0];       

    T = Int(t_real / ΔT)
    player_cost_models = map(enumerate(as)) do (ii, a)
    cost_model_p1 = CollisionAvoidanceGame.generate_integrator_cost(;
        player_idx = ii,
        control_system,
        T,
        goal_position = scale*unitvector(a),
        weights = (; 
            state_proximity = weights[ii, 1], 
            state_goal      = weights[ii, 2],
            control_Δv      = weights[ii, 3]),
        T_activate_goalcost,
        prox_min_regularization = 0.1
    )
    end

    # ---- Solve FG ---- 
    kkt_converged, kkt_solution, kkt_model = 
        solve_game(KKTGameSolver(), control_system, player_cost_models, x0, T; 
        solver = Ipopt.Optimizer, 
        solver_attributes = (;max_wall_time, print_level = 5))

    # ---- Animation trajectories ----
    animate_trajectory(
        kkt_solution.x, 
        (;
            ΔT = ΔT,
            title = "double_integrators", 
            n_players, 
            n_states_per_player = 4
        );
        fps = 10
    )


    kkt_solution.x, kkt_solution.u
end

# function initialize_duals(opt_model_old, opt_model_new)
#     Main.@infiltrate
    
#     constraint_solution = Dict()
#     for (F, S) in JuMP.list_of_constraint_types(opt_model_old)
#         # We add a try-catch here because some constraint types might not
#         # support getting the primal or dual solution.
#         try
#             for (ref_old, ref_new) in zip(
#                 JuMP.all_constraints(opt_model_old, F, S),
#                 JuMP.all_constraints(opt_model_new, F, S),
#             )
#                 constraint_solution[ref_new] = (JuMP.value(ref_old), JuMP.dual(ref_old))
#             end
#         catch
#             @info("Something went wrong getting $F-in-$S. Skipping")
#         end
#     end
#     # Now we can loop through our cached solutions and set the starting values.
#     # for (x, primal_start) in variable_primal
#     #     set_start_value(x, primal_start)
#     # end

#     for (ci, (primal_start, dual_start)) in constraint_solution
#         # JuMP.set_start_value(ci, primal_start)
#         JuMP.set_dual_start_value(ci, dual_start)
#     end
#     return

# end


function forward_then_inverse()
    # Solve a forward game with known weights and hyperplane parameters, and then see if inverse solver can correctly recover them 

    # ---- Parameters ---- 
    
    # Game 
    ΔT = 0.1
    n_players = 2
    scale = 1
    t_real = 4.0
    T_activate_goalcost = 1
    v_init = 0.5
    os = deg2rad(90) # init. angle offset
    
    # Cost and hyperplane
    adjacency_matrix = [false true;
                        false false]
    ω = 0.7
    α = 0.0
    ρ = 0.1
    weights = [0.9 0.1;
               0.9 0.1];    
    

    # Solver  
    μ = 1e-7
    ρmin = 0.1
    
    # ---- Solve FG ---- 

    # System and initial conditions
    control_system = TestDynamics.ProductSystem([TestDynamics.DoubleIntegrator(ΔT) for _ in 1:n_players])
    as = [2*pi/n_players * (i-1) for i in 1:n_players] # angles
    as = [a > pi ? a - 2*pi : a for a in as]

    x0 = vcat(
    [
        vcat(-scale*unitvector(a), [v_init*cos(a - os), v_init*sin(a - os)]) for
        a in as
    ]...,
    )

    # Costs
    T = Int(t_real / ΔT)
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

    constraint_parameters = (;adjacency_matrix, ωs = ω, αs = α, ρs = ρ) # These parameters work 
    converged_forward, time_forward, solution_forward, model_forward = 
        solve_game(KKTGameSolverBarrier(), control_system, player_cost_models, x0, constraint_parameters, T;
        init = (; s = 1.5),
        solver = Ipopt.Optimizer, 
        solver_attributes = (; max_wall_time = 20.0, print_level = 5),
        μ
        )
    player_weights = [
            (; state_goal = weights[1, 1], control_Δv = weights[1, 2]),
            (; state_goal = weights[2, 1], control_Δv = weights[2, 2]),
        ]

    # ---- Inverse ----

    # New cost model w/ dummy weights
    player_cost_models = map(enumerate(as)) do (ii, a)
        cost_model_p1 = CollisionAvoidanceGame.generate_hyperintegrator_cost(;
            player_idx = ii,
            control_system,
            T,
            goal_position = scale*unitvector(a),
            weights = (; 
                state_goal      = -1,
                control_Δv      = -1),
            T_activate_goalcost,
        )
    end

    y = (;x = solution_forward.x, u = solution_forward.u)
    converged, solution_inverse, model_inverse = solve_inverse_game(
            InverseHyperplaneSolver(),
            y, 
            adjacency_matrix;
            control_system,
            player_cost_models,
            init = (;ωs = ω, αs = α, ρs = ρ, player_weights, solution_forward...),
            solver = Ipopt.Optimizer,
            solver_attributes = (; max_wall_time = 60.0, print_level = 5),
            cmin = 1e-5,
            ρmin,
            μ,
        )

    # ---- Compare inferred and true parameters ----
    println("True weights: ", weights)
    println(
        "Inferred weights: ",
        round.(
            hcat(
                [solution_inverse.player_weights[i].state_goal for i in 1:n_players],
                [solution_inverse.player_weights[i].control_Δv for i in 1:n_players],
            ),
            digits = 2,
        ),
    )
    if length(findall(adjacency_matrix)) > 0
        println("True parameters: ", ω, " ", α, " ", ρ)
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

    # Forward
    visualize_rotating_hyperplanes(
            solution_forward.x,
            (;
                ΔT = 0.1,
                adjacency_matrix = adjacency_matrix,
                ωs = constraint_parameters.ωs,
                αs = constraint_parameters.αs,
                ρs = constraint_parameters.ρs,
                title = "forward_hyperplane",
                n_players = 2,
                n_states_per_player = 4,
                goals = [player_cost_models[i].goal_position for i in 1:n_players],
            );
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
                title = "inverse_hyperplane",
                n_players = 2,
                n_states_per_player = 4,
                goals = [player_cost_models[i].goal_position for i in 1:n_players],
            );
            koz = true,
            fps = 10.0,
        )

end

function infer_and_check(data_states, data_inputs)

    # User input
    ΔT = 0.1
    n_states_per_player = 4 
    scale = 1
    ρmin = 0.1

    adjacency_matrix = [false true;
                        false false]
    μs = [1e-1]
    
    # Load data 
    y = (;x = data_states, u = data_inputs)

    T = size(data_states,2)
    T_activate_goalcost = T

    # Setup control system 
    n_players = size(adjacency_matrix, 2)
    control_system = TestDynamics.ProductSystem([TestDynamics.DoubleIntegrator(ΔT) for _ in 1:n_players])

    # Presumed cost system with dummy variables
    T = size(data_states,2)
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

    # First solve 
    # adjacency_matrix = zeros(Bool, n_players, n_players)
    converged_1, solution_1, opt_model_1 = solve_inverse_game(
            InverseHyperplaneSolver(),
            y, 
            adjacency_matrix;
            control_system,
            player_cost_models,
            # init = (;s = 1.5),
            solver = Ipopt.Optimizer,
            solver_attributes = (; max_wall_time = 60.0, print_level = 5),
            cmin = 1e-5,
            ρmin,
            μ = μs[1],
        )

    # Warm start solver with previous soln at solve agian. Should converge almost immediately
    converged_2, solution_2, opt_model_2 = solve_inverse_game(
            InverseHyperplaneSolver(),
            y, 
            adjacency_matrix;
            control_system,
            player_cost_models,
            init = solution_1, 
            solver = Ipopt.Optimizer,
            solver_attributes = (; max_wall_time = 20.0, print_level = 5, warm_start_init_point = :yes),
            cmin = 1e-5,
            ρmin,
            μ = μs[1],
        )

    # visualize_rotating_hyperplanes(
    #         data_states,
    #         (;
    #             ΔT = 0.1,
    #             adjacency_matrix = adjacency_matrix,
    #             ωs = solution_inverse.ωs,
    #             αs = solution_inverse.αs,
    #             ρs = solution_inverse.ρs,
    #             title = "inverse_solution",
    #             n_players = 2,
    #             n_states_per_player = 4,
    #             goals = [player_cost_models[i].goal_position for i in 1:n_players],
    #         );
    #         koz = true,
    #         fps = 10.0,
    #     )
    # animate_trajectory(
    #     solution_inverse.x, 
    #     (;
    #         ΔT = ΔT,
    #         title = "traj_1", 
    #         n_players, 
    #         n_states_per_player = 4
    #     );
    #     fps = 10
    # )

end

function main()
    data_states, data_inputs = generate_forward_game()

    infer_and_check(data_states, data_inputs)

end