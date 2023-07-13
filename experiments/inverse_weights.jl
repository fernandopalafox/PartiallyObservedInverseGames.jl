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

include("utils/misc.jl")

# TEMPORARY FUNCTIONS THAT SHOULD BE IN THEIR OWN MODULE
function set_solver_attributes!(opt_model; solver_attributes...)
    foreach(
        ((k, v),) -> JuMP.set_optimizer_attribute(opt_model, string(k), v),
        pairs(solver_attributes),
    )
end
get_values(; jump_vars...) = (; map(((k, v),) -> k => JuMP.value.(v), collect(jump_vars))...)
function init_if_hasproperty!(v, init, sym; default = nothing)
    init_value = hasproperty(init, sym) ? getproperty(init, sym) : default
    if !isnothing(init_value)
        JuMP.set_start_value.(v, init_value)
    end
end

solution = let 
# ---- USER INPUT: Solver settings ----

T_activate_goalcost = 1
ΔT = 0.1
n_players = 2
scale = 1

solver_attributes = (; print_level = 5, expect_infeasible_problem = "no")

# Setup warmstart
init = nothing

# ---- Solve ---- 

# Presumed system dynamics

control_system = TestDynamics.ProductSystem([TestDynamics.DoubleIntegrator(ΔT) for _ in 1:n_players])

@unpack n_states, n_controls = control_system
n_states_per_player = Int(n_states / n_players)

# Setup solver
opt_model = JuMP.Model(Ipopt.Optimizer)
set_solver_attributes!(opt_model; solver_attributes...)
JuMP.set_time_limit_sec(opt_model, 80.0)

# Load observation data
data_states = Matrix(CSV.read("data/f_di_s.csv", DataFrame, header = false))
data_inputs = Matrix(CSV.read("data/f_di_c.csv", DataFrame, header = false))

# Compute values from data
T             = size(data_states,2)
x0            = data_states[:,1]
as            = data_states[n_states_per_player:n_states_per_player:n_states,1]

# ---- USER INPUT: Setup unknown parameters ----

# Cost parameters
n_weights = 3
player_weights = @variable(opt_model, [1:n_players, 1:n_weights], lower_bound = 1e-5)
@constraint(opt_model, [p = 1:n_players], sum(player_weights[p,:]) == 1) #regularization 

# Costs
player_cost_models = map(enumerate(as)) do (ii, a)
    cost_model_p1 = CollisionAvoidanceGame.generate_integrator_cost(;
        player_idx = ii,
        control_system,
        T,
        goal_position = scale*unitvector(a),
        weights = (; 
            state_proximity = 0.1 * player_weights[ii, 1], 
            state_goal      = 1   * player_weights[ii, 2],
            control_Δvx     = 10  * player_weights[ii, 3], 
            control_Δvy     = 10  * player_weights[ii, 3]),
        T_activate_goalcost,
        prox_min_regularization = 0.1
    )
end

# ---- Decision variables ----
x = @variable(opt_model, [1:n_states, 1:T])
u = @variable(opt_model, [1:n_controls, 1:T])
λ_e = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])

# ---- Initialize ----
JuMP.set_start_value.(x, data_states)
JuMP.set_start_value.(u, data_inputs)

# ---- Solve ----

# Dynamics constraints
DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)
df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)

for (player_idx, cost_model) in enumerate(player_cost_models)
    @unpack player_inputs, weights = cost_model
    dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)

    # Gradient of the Lagrangian wrt x is zero 
    @constraint(
        opt_model,
        [t = 2:(T - 1)],
        dJ.dx[:, t]' + λ_e[:, t - 1, player_idx]' -
        λ_e[:, t, player_idx]' * df.dx[:, :, t] .== 0
    )
    @constraint(
        opt_model,
        dJ.dx[:, T]' + λ_e[:, T - 1, player_idx]' .== 0
    ) 

    # Gradient of the Lagrangian wrt u is zero
    @constraint(
        opt_model,
        [t = 1:(T - 1)],
        dJ.du[player_inputs, t]' - λ_e[:, t, player_idx]' * df.du[:, player_inputs, t] .==
        0
    )
    @constraint(opt_model, dJ.du[player_inputs, T]' .== 0)
 
end

# Solve problem 
time = @elapsed JuMP.optimize!(opt_model)
@info time

# merge((;x,u), get_values(;weights))
get_values(;player_weights, x, u)
end