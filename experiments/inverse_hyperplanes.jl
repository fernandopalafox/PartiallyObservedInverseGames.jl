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
    n_states_per_player = 4 
    scale = 1
    T_activate_goalcost = 1
    ρmin = 0.1

    # adjacency_matrix = [false true; 
    #                     false false]
    adjacency_matrix = [false true true;
                        false false true;
                        false false false]
    # adjacency_matrix = [false true true true;
    #                     false false true true;
    #                     false false false true;
    #                     false false false false]    
    # adjacency_matrix = [false false false false]
    # adjacency_matrix = nothing
    μs = [0.1*(1/10)^(i - 1) for i in 1:5]
    
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

    # Initial solve 
    (converged, solution_outside) = solve_inverse_game(
            InverseHyperplaneSolver(),
            y, 
            adjacency_matrix;
            control_system,
            player_cost_models,
            # init = (; s = 1.5),
            solver = Ipopt.Optimizer,
            solver_attributes = (; max_wall_time = 60.0, print_level = 5),
            cmin = 1e-5,
            ρmin,
            μ = μs[1],
        )

    @assert converged 

    # Check whether it converges when warmstarted 
    # (converged, solution_2) = solve_inverse_game(
    #         InverseHyperplaneSolver(),
    #         y, 
    #         adjacency_matrix;
    #         control_system,
    #         player_cost_models,
    #         init = solution_outside,
    #         solver = Ipopt.Optimizer,
    #         solver_attributes = (; max_wall_time = 60.0, print_level = 5),
    #         cmin = 1e-5,
    #         ρmin,
    #         μ = μs[1],
    #     )
    # solution_outside = solution_2

    Main.@infiltrate

    # Iterate over μs
    # solution_warmstart = solution_outside
    # for μ in μs
    #     (converged, solution_inside) = solve_inverse_game(
    #         InverseHyperplaneSolver(),
    #         y,
    #         adjacency_matrix;
    #         control_system,
    #         player_cost_models,
    #         # init = (; 
    #         #     x = solution_warmstart.x,
    #         #     u = solution_warmstart.u,
    #         #     s = solution_warmstart.s,
    #         #     player_weights = solution_warmstart.player_weights,
    #         #     ωs = solution_warmstart.ωs,
    #         #     αs = solution_warmstart.αs,
    #         #     ρs = solution_warmstart.ρs,
    #         # ),
    #         # init = solution_warmstart,
    #         solver = Ipopt.Optimizer,
    #         solver_attributes = (; max_wall_time = 60.0, print_level = 5),
    #         cmin = 1e-5,
    #         ρmin,
    #         μ,
    #     )
    #     if !converged
    #         println("μ = $μ did not converge")
    #         solution_outside = solution_warmstart
    #         break
    #     end
    #     solution_warmstart = solution_inside
    #     solution_outside   = solution_inside
    #     println("μ = $μ")
    # end
    # Plot
    visualize_rotating_hyperplanes(
        solution_outside.x,
        (; adjacency_matrix, ωs = solution_outside.ωs, αs = solution_outside.αs, ρs = solution_outside.ρs , title = "Inverse", n_players, n_states_per_player);
        koz = true, fps = 10.0
    )
end