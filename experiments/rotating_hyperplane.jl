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
ω = 0.05
ω = 0.025
ρ = 0.25

# Constraints

# Dynamics
control_system =
    TestDynamics.ProductSystem([TestDynamics.HyperUnicycle(0.25, ω, ρ), TestDynamics.Unicycle(0.25)])
# control_system =
#     TestDynamics.ProductSystem([TestDynamics.Unicycle(0.25), TestDynamics.Unicycle(0.25)])

# control_system =
#     TestDynamics.ProductSystem([TestDynamics.Unicycle(0.25), TestDynamics.Unicycle(0.25), TestDynamics.Unicycle(0.25)])

# Initial conditions
# player_angles = let
#     n_players = length(control_system.subsystems)
#     map(eachindex(control_system.subsystems)) do ii
#         angle_fraction = n_players == 2 ? pi / 2 : 2pi / n_players
#         (ii - 1) * angle_fraction
#     end
# end
# x0 = mapreduce(vcat, player_angles) do player_angle
#     [unitvector(player_angle + pi); 0.1; player_angle + deg2rad(10)]
# end

# This works 
player_angles = [0.0, pi/2]
x0 = [-1.0, 0.0, 0.1, player_angles[1], 0.0, -1.0, 0.1, player_angles[2]]


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
ibr_converged, ibr_solution, ibr_models =
        solve_game(IBRGameSolver(), control_system, player_cost_models, x0, T)
kkt_converged, kkt_solution, kkt_model = 
        solve_game(KKTGameSolverBarrier(), control_system, player_cost_models, x0, T; solver = Ipopt.Optimizer, 
        init = (;x = ibr_solution.x, u = ibr_solution.u))

        
# visualize_trajectory(control_system, ibr_solution.x, canvas = VegaLite.@vlplot(width = 400, height = 400))

# ---- Save trajectory to file ----
CSV.write("data/KKT_trajectory_state.csv", DataFrame(kkt_solution.x, :auto), header = false)
CSV.write("data/KKT_trajectory_control.csv", DataFrame(kkt_solution.u, :auto), header = false)
CSV.write("data/IBR_trajectory_state.csv", DataFrame(ibr_solution.x, :auto), header = false)
CSV.write("data/IBR_trajectory_control.csv", DataFrame(ibr_solution.u, :auto), header = false)

# ---- Plot Trajectory ----

# trajectory_data_gt =
#     TrajectoryVisualization.trajectory_data(control_system, ibr_solution.x)

# trajectory_viz_config = (;
#     x_position_domain = (-1.2, 1.2),
#     y_position_domain = (-1.2, 1.2),
#     opacity = 0.5,
#     legend = false,
# )

# groups = [
#     "Ground Truth"
# ]
# color_map = Dict([
#     "Ground Truth" => "black"
# ])
# color_scale =
#     VegaLite.@vlfrag(domain = groups, range = [color_map[g] for g in groups])

# ground_truth_viz =
#     TrajectoryVisualization.visualize_trajectory(
#         trajectory_data_gt;
#         canvas = VegaLite.@vlplot(width = 400, height = 400),
#         legend = VegaLite.@vlfrag(orient = "top", offset = 5),
#         trajectory_viz_config...,
#         group = "Ground Truth",
#         color_scale,
#     ) + VegaLite.@vlplot(
#         data = filter(s -> s.t == 1, trajectory_data_gt),
#         mark = {"text", dx = 8, dy = 8},
#         text = "player",
#         x = "px:q",
#         y = "py:q",
#     )

# ---- Animation with rotating hyperplane ----
# visualize_rotating_hyperplane(ibr_solution.x) 
visualize_rotating_hyperplane(kkt_solution.x,(; ω = ω, ρ = ρ, title = "Forward"))
visualize_rotating_hyperplane(ibr_solution.x,(; ω = ω, ρ = ρ, title = "IBR"))