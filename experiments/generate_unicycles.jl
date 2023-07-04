# Imports
const project_root_dir = realpath(joinpath(@__DIR__, ".."))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "test/utils"))))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "experiments/utils"))))

using Revise

import TestDynamics

using PartiallyObservedInverseGames.ForwardGame: IBRGameSolver, KKTGameSolver, KKTGameSolverBarrier, solve_game
using JuMP: @objective
using VegaLite: VegaLite
using PartiallyObservedInverseGames.TrajectoryVisualization:
    TrajectoryVisualization, animate_trajectory
using CollisionAvoidanceGame: CollisionAvoidanceGame
using CSV, DataFrames

include("utils/misc.jl")

# ---- Setup ---- 
T = 75

# Dynamics
# TODO: constraint params should be specified by each player when the control system is initialized. Not by a global adj matrix
control_system =
    TestDynamics.ProductSystem([TestDynamics.Unicycle(0.25), 
                                TestDynamics.Unicycle(0.25), 
                                TestDynamics.Unicycle(0.25)])
# control_system =
#     TestDynamics.ProductSystem([TestDynamics.Unicycle(0.25), 
#                                 TestDynamics.Unicycle(0.25)])


# Initial position 
player_angles = [0.0, pi/2, pi]
x0 = [-1.0,  0.0, 0.3, -deg2rad(30), 
       0.0, -1.0, 0.3,  pi/2 - deg2rad(30),
       1.0,  0.0, 0.3,  pi - deg2rad(30)]
# player_angles = [0.0, pi]
# x0 = [-1.0,  0.0, 0.5, -pi/2, 
#        1.0,  0.0, 0.5,  pi/2]
# Costs
player_cost_models = map(enumerate(player_angles)) do (ii, player_angle)
    cost_model_p1 = CollisionAvoidanceGame.generate_player_cost_model(;
        player_idx = ii,
        control_system,
        T,
        goal_position = round.(unitvector(player_angle)),
        weights = (; 
            state_proximity = 1, 
            state_velocity = 1, 
            control_Δv = 1, 
            control_Δθ = 1),
    )
end

# ---- Solve FG ---- 
kkt_converged, kkt_solution, kkt_model = 
        solve_game(KKTGameSolver(), control_system, player_cost_models, x0, T; 
        solver = Ipopt.Optimizer, 
        solver_attributes = (; max_wall_time = 120.0, print_level = 5))


# ---- Save trajectory to file ----
CSV.write("data/fwd_unicycles_state.csv", DataFrame(kkt_solution.x, :auto), header = false)
CSV.write("data/fwd_unicycles_control.csv", DataFrame(kkt_solution.u, :auto), header = false)

# ---- Animation trajectories ----
animate_trajectory(kkt_solution.x, (;title = "unicycles"))