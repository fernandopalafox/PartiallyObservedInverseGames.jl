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
n_players = 4
v_init = 0.3
offset = 90
scale  = 1.0
control_system = TestDynamics.ProductSystem([TestDynamics.Unicycle(0.25) for _ in 1:n_players])
angles = [2*pi/n_players * (i-1) for i in 1:n_players]
angles = [angle > pi ? angle - 2*pi : angle for angle in angles]

x0 = vcat(
    [
        vcat(-scale*round.(unitvector(angle)), [v_init, angle - deg2rad(offset)]) for
        angle in angles
    ]...,
)

# Costs
player_cost_models = map(enumerate(angles)) do (ii, angle)
    cost_model_p1 = CollisionAvoidanceGame.generate_player_cost_model(;
        player_idx = ii,
        control_system,
        T,
        goal_position = scale*round.(unitvector(angle), digits = 6),
        weights = (; 
            state_proximity = 0.5, 
            state_velocity = 1, 
            control_Δv = 1, 
            control_Δθ = 1),
    )
end

# ---- Solve FG ---- 
kkt_converged, kkt_solution, kkt_model = 
        solve_game(KKTGameSolver(), control_system, player_cost_models, x0, T; 
        solver = Ipopt.Optimizer, 
        solver_attributes = (; max_wall_time = 300.0, print_level = 5))


# ---- Save trajectory to file ----
CSV.write("data/fwd_unicycles_state.csv", DataFrame(kkt_solution.x, :auto), header = false)
CSV.write("data/fwd_unicycles_control.csv", DataFrame(kkt_solution.u, :auto), header = false)

# ---- Animation trajectories ----
animate_trajectory(
        kkt_solution.x, 
        (;
            title = "unicycles", 
            n_players = length(control_system.subsystems), 
            n_states_per_player = 4
        )
    )