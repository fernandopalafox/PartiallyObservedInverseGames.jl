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

# ---- Setup ----


# User inputs
n_players = 4
solver_attributes = (; print_level = 5, expect_infeasible_problem = "no")

# ω = 0.05 # REMOVE THIS

# ---- Solve ---- 

# Presumed system dynamics
control_system =
    TestDynamics.ProductSystem([TestDynamics.Unicycle(0.25), 
                                TestDynamics.Unicycle(0.25)])
@unpack n_states, n_controls = control_system

# Setup solver
opt_model = JuMP.Model(Ipopt.Optimizer)
set_solver_attributes!(opt_model; solver_attributes...)
JuMP.set_time_limit_sec(opt_model, 20.0)

# Load observation data
data_states = Matrix(CSV.read("data/KKT_intersection_4_state.csv", DataFrame, header = false))
data_controls = Matrix(CSV.read("data/KKT_intersection_4_control.csv", DataFrame, header = false))
T = size(data_states,2)

# Compute initial values
x0 = data_states[:,1]

# Setup decision variable for unknown parameter
xgs = @variable(opt_model, [1:2,1:n_players])

# Set cost models
player_cost_models = map(enumerate(player_angles)) do (ii, player_angle)
    cost_model_p1 = CollisionAvoidanceGame.generate_player_cost_model(;
        player_idx = ii,
        control_system,
        T,
        goal_position = xgs[:,ii],
        weights = (; state_proximity = 1, state_velocity = 20, control_Δv = 20, control_Δθ = 10),
    )
end

# Other decision variables
x   = @variable(opt_model, [1:n_states, 1:T])
u   = @variable(opt_model, [1:n_controls, 1:T])
λ_e = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])

# Warms start on decision variables
init = (;x = data_states, u = data_controls)
init_if_hasproperty!(λ_e, init, :λ_e)
init_if_hasproperty!(x, init, :x)
init_if_hasproperty!(u, init, :u)

# Constraint on initial position
@constraint(opt_model, x[:, 1] .== x0)

# Dyanmics constraints
DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)
df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)

# Add KKT constraints for each player 
for (player_idx, cost_model) in enumerate(player_cost_models)
    @unpack player_inputs, weights = cost_model
    dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)

    # KKT Nash constraints
    @constraint(
        opt_model,
        [t = 2:(T - 1)],
        dJ.dx[:, t] + λ_e[:, t - 1, player_idx] - (λ_e[:, t, player_idx]' * df.dx[:, :, t])' .== 0
    )
    @constraint(opt_model, dJ.dx[:, T] + λ_e[:, T - 1, player_idx] .== 0)

    @constraint(
        opt_model,
        [t = 1:(T - 1)],
        dJ.du[player_inputs, t] - (λ_e[:, t, player_idx]' * df.du[:, player_inputs, t])' .== 0
    )
    @constraint(opt_model, dJ.du[player_inputs, T] .== 0)      
end

# Match equilibirum
@objective(opt_model, Min, sum(el -> el^2, data_states .- x))

time = @elapsed JuMP.optimize!(opt_model)
@info time

solution = get_values(;x, u, xgs)

visualize_intersection(solution.x, (;title = "4-player inverse"))