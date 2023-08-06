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

include("utils/misc.jl")

function main()

    # User input
    ΔT = 0.1
    n_states_per_player = 4 
    scale = 1
    T_activate_goalcost = 1
    ρmin = 0.2

    adjacency_matrix = [false true true;
                        false false true;
                        false false false]
    adjacency_matrix = [false true;
                        false false]
    μs = [1*(1/10)^(i - 1) for i in 1:5]
    μs = [1e-5]
    
    # Load data 
    data_states   = Matrix(CSV.read("data/f_di_s.csv", DataFrame, header = false))
    data_inputs   = Matrix(CSV.read("data/f_di_c.csv", DataFrame, header = false))
    y = (;x = data_states, u = data_inputs)

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

    # Solve
    (converged, solution_inverse) = solve_inverse_game(
            InverseHyperplaneSolver(),
            y, 
            adjacency_matrix;
            control_system,
            player_cost_models,
            init = (; x = data_states, u = data_inputs, s = 1.5),
            solver = Ipopt.Optimizer,
            solver_attributes = (; max_wall_time = 60.0, print_level = 5),
            cmin = 1e-5,
            ρmin,
            μ = μs[1],
        )

    # Plot inverse game
    visualize_rotating_hyperplanes(
        solution_inverse.x,
        (; ΔT, adjacency_matrix, ωs = solution_inverse.ωs, αs = solution_inverse.αs, ρs = solution_inverse.ρs , title = "Inverse", n_players, n_states_per_player);
        koz = true, fps = 10.0
    )

    # Use inverse game to solve a different forward game

    # Parameters
    T = size(data_states,2)

    # New initial condition 
    x0 = data_states[:,1]

    os_v = deg2rad(90) # init. angle offset
    os_p = deg2rad(45)
    v_init = 0.5
    as = [2*pi/n_players * (i-1) for i in 1:n_players] 
    as = [a > pi ? a - 2*pi : a for a in as]
    x0 = vcat(
        [
            vcat(-scale*unitvector(a + os_p), [v_init*cos(a - os_v), v_init*sin(a - os_v)]) for
            a in as
        ]...,
    )
    
    # Setup cost models
    player_cost_models = map(enumerate(as)) do (ii, a)
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

    # Solve 
    constraint_parameters = (;adjacency_matrix, ωs = solution_inverse.ωs, αs = solution_inverse.αs, ρs = solution_inverse.ρs)
    _, solution_forwards = 
        solve_game(KKTGameSolverBarrier(), control_system, player_cost_models, x0, constraint_parameters, T;
        init = (; s = solution_inverse.s),
        solver = Ipopt.Optimizer, 
        solver_attributes = (; max_wall_time = 20.0, print_level = 5),
        μ = μs[1]
        )

    # Plot  
    visualize_rotating_hyperplanes(
        solution_forwards.x,
        (; ΔT, adjacency_matrix, ωs = solution_inverse.ωs, αs = solution_inverse.αs, ρs = solution_inverse.ρs , title = "New FWD", n_players, n_states_per_player);
        koz = true, fps = 10.0
    )

    Main.@infiltrate
end