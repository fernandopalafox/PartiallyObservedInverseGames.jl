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
using Plots

include("utils/misc.jl")

# ---- Setup ---- 

let 

ΔT = 0.1
n_players = 2
scale = 1
t_real = 10.0
t_real_activate_goalcost = t_real
T_activate_goalcost = Int(t_real_activate_goalcost / ΔT)
# T_activate_goalcost = Int(T_activate_goalcost/2)

v_init = 0.5
os = deg2rad(90) # init. angle offset
max_wall_time = 60.0

# Satellite parameters
m   = 100.0 # kg
r₀ = (400 + 6378.137) # km
grav_parameter  = 398600.4418 # km^3/s^2

n = sqrt(grav_parameter/(r₀^3)) # rad/s

# Setup system
control_system = TestDynamics.ProductSystem([TestDynamics.Satellite2D(ΔT, n, m) for _ in 1:n_players])
as = [2*pi/n_players * (i-1) for i in 1:n_players] # angles
as = [a > pi ? a - 2*pi : a for a in as]

x0 = vcat(
    [
        vcat(-scale*unitvector(a), [v_init*cos(a - os), v_init*sin(a - os)]) for
        a in as
    ]...,
)

# Costs 
weights = repeat([0.0001 10.0 0.0001], outer = n_players) # works well enough 

T = Int(t_real / ΔT)
player_cost_models = map(enumerate(as)) do (ii, a)
    cost_model_p1 = CollisionAvoidanceGame.generate_integrator_cost(;
        player_idx = ii,
        control_system,
        T,
        goal_position = scale*unitvector(a),
        weights = (; 
            state_proximity = weights[ii, 1], 
            state_goal      = weights[ii, 2],
            control_Δv      = weights[ii, 3]),
        T_activate_goalcost,
        prox_min_regularization = 0.1
    )
end

# ---- Solve FG ---- 
kkt_converged, kkt_solution, kkt_model = 
        solve_game(KKTGameSolver(), control_system, player_cost_models, x0, T; 
        solver = Ipopt.Optimizer, 
        solver_attributes = (;max_wall_time, print_level = 5))


# ---- Save trajectory to file ----
CSV.write("data/f_2d_s.csv", DataFrame(kkt_solution.x, :auto), header = false)
CSV.write("data/f_2d_c.csv", DataFrame(kkt_solution.u, :auto), header = false)

# ---- Animation trajectories ----
animate_trajectory(
        kkt_solution.x, 
        (;
            ΔT = ΔT,
            title = "fwd_game_2d_3p", 
            n_players, 
            n_states_per_player = 4,
            goals = [player_cost_models[player].goal_position for player in 1:n_players]
        );
        fps = 10
    )

# Print maximum control effort
println("Maximum control effort: ", maximum(abs.(kkt_solution.u)))

# Plot controls
plot(kkt_solution.u', label = ["1_ux" "1_uy" "2_ux" "2_uy"], xlabel = "t", ylabel = "u", title = "Controls")

end