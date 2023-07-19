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

include("utils/misc.jl")

function main()

    # User input
    ΔT = 0.1
    n_players = 4
    n_states_per_player = 4 
    scale = 1
    T_activate_goalcost = 1
    ρmin = 0.05
    # adjacency_matrix = [false true; 
    #                     false false]
    # adjacency_matrix = [false true true;
    #                     false false true;
    #                     false false false]
    adjacency_matrix = [false true true true;
                        false false true true;
                        false false false true;
                        false false false false]
    μs = [1.0*(1/2)^(i - 1) for i in 1:2]
    
    # Load data 
    data_states   = Matrix(CSV.read("data/f_di_s.csv", DataFrame, header = false))
    data_inputs   = Matrix(CSV.read("data/f_di_c.csv", DataFrame, header = false))
    y = (;x = data_states, u = data_inputs)

    # Setup control system 
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

    # # Initial solve 
    (_, solution) = solve_inverse_game(
            InverseHyperplaneSolver(),
            y, 
            adjacency_matrix;
            control_system,
            player_cost_models,
            # init = (; player_weights = solution_weights.player_weights, x = data_states, u = data_inputs, s = 1.5),
            init = (; x = data_states, u = data_inputs, s = 1.5),
            solver = Ipopt.Optimizer,
            solver_attributes = (; max_wall_time = 60.0, print_level = 5),
            cmin = 1e-5,
            ρmin,
            μ = μs[1],
        )

    # Iterate over μs
    # solution_warmstart = (; x = data_states, u = data_inputs, λ_e = solution.λ_e, λ_i = solution.λ_i, s = solution.s)
    solution_warmstart = solution
    for μ in μs
        (converged, solution) = solve_inverse_game(
            InverseHyperplaneSolver(),
            y,
            adjacency_matrix;
            control_system,
            player_cost_models,
            init = solution_warmstart,
            solver = Ipopt.Optimizer,
            solver_attributes = (; max_wall_time = 20.0, print_level = 5),
            cmin = 1e-5,
            ρmin,
            μ,
        )
        # if converged
        #     solution = solution_new
        # else
        #     println("Did not converge for μ = $μ")
        #     solution = solution_warmstart
        #     break
        # end
        solution_warmstart = solution
        println("μ = $μ")
    end
    # Plot
    visualize_rotating_hyperplanes(
        solution.x,
        (; adjacency_matrix, ωs = solution.ωs, αs = solution.αs, ρs = solution.ρs , title = "Inverse", n_players, n_states_per_player);
        koz = true, fps = 10.0
    )
    Main.@infiltrate

end