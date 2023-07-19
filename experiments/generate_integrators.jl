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

let 

ΔT = 0.1
n_players = 4
scale = 1
t_real = 3.0
T_activate_goalcost = 1

v_init = 0.5
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
weights = [0.05  0.9 0.05;
           0.05  0.05 0.9;
           0.9  0.05 0.05;
           0.33  0.33 0.33];
# weights = [0.1  0.7 0.2;
#            0.1  0.7 0.2;
#            0.1  0.7 0.2;
#            0.1  0.7 0.2];
# weights = [0.05  0.05 0.9;
#            0.05  0.05 0.9;
#            0.05  0.05 0.9];
# weights = [0.05  0.05 0.9;
#            0.05  0.05 0.9]
           
           

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
CSV.write("data/f_di_s.csv", DataFrame(kkt_solution.x, :auto), header = false)
CSV.write("data/f_di_c.csv", DataFrame(kkt_solution.u, :auto), header = false)

# ---- Animation trajectories ----
animate_trajectory(
        kkt_solution.x, 
        (;
            ΔT = ΔT,
            title = "double integrators", 
            n_players, 
            n_states_per_player = 4
        );
        fps = 10
    )

end