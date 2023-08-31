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
    TrajectoryVisualization, animate_trajectory, display_3D_trajectory
using CollisionAvoidanceGame: CollisionAvoidanceGame
using CSV, DataFrames
using Ipopt
using Plots

include("utils/misc.jl")

# ---- Setup ---- 

let 

ΔT = 0.1
n_players = 3
scale = 1
t_real = 10.0
t_real_activate_goalcost = t_real
T_activate_goalcost = Int(t_real_activate_goalcost / ΔT)

v_init = 0.5
os = deg2rad(90) # init. angle offset
max_wall_time = 60.0

# Satellite parameters
m   = 100.0 # kg
r₀ = (400 + 6378.137) # km
grav_param  = 398600.4418 # km^3/s^2

n = sqrt(grav_param/(r₀^3)) # rad/s

# Setup system
control_system = TestDynamics.ProductSystem([TestDynamics.Satellite3D(ΔT, n, m) for _ in 1:n_players])
as = [2*pi/n_players * (i-1) for i in 1:n_players] # angles
as = [a > pi ? a - 2*pi : a for a in as]
zs = [1.0, 1.0, 1.0]

x0 = vcat(
    [
        vcat(-scale*unitvector(a), 0, [v_init*cos(a - os), v_init*sin(a - os)], 0) for
        a in as
    ]...,
)

# Costs
weights = repeat([0.0 10.0 0.0001], outer = n_players) # works well enough      

T = Int(t_real / ΔT)
player_cost_models = map(enumerate(as)) do (ii, a)
    cost_model_p1 = CollisionAvoidanceGame.generate_3dintegrator_cost(;
        player_idx = ii,
        control_system,
        T,
        goal_position = vcat(scale*unitvector(a), zs[ii]),
        weights = (; 
            state_proximity = weights[ii, 1], 
            state_goal      = weights[ii, 2],
            control_Δv      = weights[ii, 3]),
        T_activate_goalcost,
        prox_min_regularization = 0.1
    )
end

# ---- Solve FG ---- 
_, kkt_solution, _ = 
        solve_game(KKTGameSolver(), control_system, player_cost_models, x0, T; 
        solver = Ipopt.Optimizer, 
        solver_attributes = (;max_wall_time, print_level = 5))


# ---- Save trajectory to file ----
CSV.write("data/f_3d_3p_s.csv", DataFrame(kkt_solution.x, :auto), header = false)
CSV.write("data/f_3d_3p_c.csv", DataFrame(kkt_solution.u, :auto), header = false)

# ---- Animation trajectories ----
animate_trajectory(
        vcat(kkt_solution.x[1:4, :], kkt_solution.x[7:10, :], kkt_solution.x[13:16, :]), # dirty hack 
        (;
            ΔT = ΔT,
            title = "satellites_xy", 
            n_players, 
            n_states_per_player = 4,
            goals = [player_cost_models[player].goal_position for player in 1:n_players]
        );
        fps = 10
    )


# Print maximum control effort
display_3D_trajectory(
    kkt_solution.x,
    (;
        n_players = n_players,
        n_states_per_player = 6,
        goals = [player_cost_models[player].goal_position for player in 1:n_players],
    );
    title = "3D_trajectory",
    filename = "forward_data.gif"
)


end