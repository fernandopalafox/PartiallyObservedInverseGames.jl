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
    TrajectoryVisualization, visualize_trajectory, visualize_intersection
using CollisionAvoidanceGame: CollisionAvoidanceGame
using CSV, DataFrames

include("utils/misc.jl")

# ---- Setup ---- 
T = 75
# ωs = [0.0 0.03] 
# ρs = [0.0 0.25]
# αs = [0.0 3/4*pi] 
# adj_mat = [false true; 
#            false false]
# constraint_params = (; adj_mat, ωs, ρs, αs)


# Dynamics
control_system =
    TestDynamics.ProductSystem([TestDynamics.Unicycle(0.25), 
                                TestDynamics.Unicycle(0.25),
                                TestDynamics.Unicycle(0.25),
                                TestDynamics.Unicycle(0.25)])
# control_system =
#     TestDynamics.ProductSystem([TestDynamics.Unicycle(0.25), 
#                                 TestDynamics.Unicycle(0.25)])

# Initial position 
player_angles = [pi/4, pi/4, 5/4*pi, 7/4*pi]
x0 = [-0.7, -0.7, 0.1, player_angles[1], 
       0.7, -0.7, 0.1, player_angles[2],
       0.7,  0.7, 0.1, player_angles[3],
      -0.7,  0.7, 0.1, player_angles[4]]

# Costs
player_cost_models = map(enumerate(player_angles)) do (ii, player_angle)
    cost_model_p1 = CollisionAvoidanceGame.generate_player_cost_model(;
        player_idx = ii,
        control_system,
        T,
        goal_position = unitvector(player_angle),
        weights = (; state_proximity = 1, state_velocity = 20, control_Δv = 20, control_Δθ = 10)
    )
end

# ---- Solve FG ---- 
# Use IBR to warmstart KKT
ibr_converged, ibr_solution, ibr_models =
        solve_game(IBRGameSolver(), control_system, player_cost_models, x0, T)
kkt_converged, kkt_solution, kkt_model = 
        solve_game(KKTGameSolverBarrier(), control_system, player_cost_models, x0, T; 
        solver = Ipopt.Optimizer, 
        solver_attributes = (; max_wall_time = 300.0, print_level = 5),
        init = (;x = ibr_solution.x, u = ibr_solution.u))


# ---- Save trajectory to file ----
CSV.write("data/KKT_intersection_4_state.csv", DataFrame(kkt_solution.x, :auto), header = false)
CSV.write("data/KKT_intersection_4_control.csv", DataFrame(kkt_solution.u, :auto), header = false)
CSV.write("data/IBR_intersection_4_state.csv", DataFrame(ibr_solution.x, :auto), header = false)
CSV.write("data/IBR_intersection_4_control.csv", DataFrame(ibr_solution.u, :auto), header = false)

# ---- Animation with rotating hyperplane ----
# visualize_rotating_hyperplane(ibr_solution.x,(; ωs, ρs, αs, title = "IBR_intersection"))
# visualize_rotating_hyperplane(kkt_solution.x,(; ωs, ρs, αs, title = "KKT_intersection"))
visualize_intersection(ibr_solution.x, (;title = "IBR"))
visualize_intersection(kkt_solution.x, (;title = "KKT"))