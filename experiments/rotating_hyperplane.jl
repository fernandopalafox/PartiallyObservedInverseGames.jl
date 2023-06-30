# Imports
const project_root_dir = realpath(joinpath(@__DIR__, ".."))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "test/utils"))))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "experiments/utils"))))

using Revise

import Ipopt
import TestDynamics

using PartiallyObservedInverseGames.ForwardGame: IBRGameSolver, KKTGameSolver, KKTGameSolverBarrier, solve_game
using JuMP: @objective
using VegaLite: VegaLite
using PartiallyObservedInverseGames.TrajectoryVisualization:
    TrajectoryVisualization, visualize_trajectory, visualize_rotating_hyperplane
using CollisionAvoidanceGame: CollisionAvoidanceGame
using CSV, DataFrames

include("utils/misc.jl")

# ---- Setup ---- 
T = 75
ωs = [0.0 0.03] 
ρs = [0.0 0.25]
αs = [0.0 3/4*pi] 
adj_mat = [false true; 
           false false]
constraint_params = (; adj_mat, ωs, ρs, αs)


# Dynamics
control_system =
    TestDynamics.ProductSystem([TestDynamics.HyperUnicycle(0.25, 0.0, 0.0), TestDynamics.Unicycle(0.25)])

# Initial position 
player_angles = [0.0, pi/2]
x0 = [-1.0, 0.0, 0.1, player_angles[1], 
      0.0, -1.0, 0.1, player_angles[2]]

# Costs
player_cost_models = map(enumerate(player_angles)) do (ii, player_angle)
    cost_model_p1 = CollisionAvoidanceGame.generate_player_cost_model_simple(;
        player_idx = ii,
        control_system,
        T,
        goal_position = unitvector(player_angle),
    )
end

# ---- Solve FG ---- 
# Use IBR to warmstart KKT
ibr_converged, ibr_solution, ibr_models =
        solve_game(IBRGameSolver(), control_system, player_cost_models, x0, T)
kkt_converged, kkt_solution, kkt_model = 
        solve_game(KKTGameSolverBarrier(), control_system, player_cost_models, x0, T; 
        solver = Ipopt.Optimizer, 
        solver_attributes = (; max_wall_time = 60.0, print_level = 5),
        init = (;x = ibr_solution.x, u = ibr_solution.u),
        constraint_params = constraint_params)


# ---- Save trajectory to file ----
CSV.write("data/KKT_trajectory_state.csv", DataFrame(kkt_solution.x, :auto), header = false)
CSV.write("data/KKT_trajectory_control.csv", DataFrame(kkt_solution.u, :auto), header = false)
# CSV.write("data/IBR_trajectory_state.csv", DataFrame(ibr_solution.x, :auto), header = false)
# CSV.write("data/IBR_trajectory_control.csv", DataFrame(ibr_solution.u, :auto), header = false)

# ---- Animation with rotating hyperplane ----
visualize_rotating_hyperplane(ibr_solution.x,(; ωs, ρs, αs, title = "IBR"))
visualize_rotating_hyperplane(kkt_solution.x,(; ωs, ρs, αs, title = "KKT"))