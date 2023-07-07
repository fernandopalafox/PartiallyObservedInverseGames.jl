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
using Ipopt

include("utils/misc.jl")

# ---- Setup ---- 

T_activate_goalcost = 1
ΔT = 0.1
n_players = 4
scale = 1

T = 30
v_init = 0.4
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
player_cost_models = map(enumerate(as)) do (ii, a)
    cost_model_p1 = CollisionAvoidanceGame.generate_integrator_cost(;
        player_idx = ii,
        control_system,
        T,
        goal_position = scale*unitvector(a),
        weights = (; 
            state_proximity = 0.05, 
            state_goal = 1,
            control_Δvx = 1, 
            control_Δvy = 1),
        T_activate_goalcost,
        prox_min_regularization = 0.1
    )
end

# player_cost_models = map(enumerate(as)) do (ii, a)
#     cost_model_p1 = CollisionAvoidanceGame.generate_player_cost_model(;
#         player_idx = ii,
#         control_system,
#         T,
#         goal_position = unitvector(a),
#         weights = (; 
#             state_proximity = 1, 
#             control_Δv = 1,
#             control_Δθ = 1),
#         T_activate_goalcost = 25
#     )
# end

# ---- Solve FG ---- 
kkt_converged, kkt_solution, kkt_model = 
        solve_game(KKTGameSolver(), control_system, player_cost_models, x0, T; 
        solver = Ipopt.Optimizer, 
        solver_attributes = (;max_wall_time, print_level = 5))


# ---- Save trajectory to file ----
CSV.write("data/f_di_s.csv", DataFrame(kkt_solution.x, :auto), header = false)
CSV.write("data/f_di_c.csv", DataFrame(kkt_solution.u, :auto), header = false)

# ---- Animation trajectories ----
animate_trajectory(
        kkt_solution.x, 
        (;
            title = "double integrators", 
            n_players = length(control_system.subsystems), 
            n_states_per_player = 4
        )
    )