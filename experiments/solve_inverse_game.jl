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
μ = 0.00001
n_players = 2
ρ = 0.25
solver_attributes = (; print_level = 5, expect_infeasible_problem = "yes")

# ω = 0.05 # REMOVE THIS

# ---- Solve ---- 

# Presumed system dynamics
control_system =
    TestDynamics.ProductSystem([TestDynamics.HyperUnicycle(0.25, 0.0, ρ), TestDynamics.Unicycle(0.25)])
@unpack n_states, n_controls = control_system

# Setup solver
opt_model = JuMP.Model(Ipopt.Optimizer)
set_solver_attributes!(opt_model; solver_attributes...)
JuMP.set_time_limit_sec(opt_model, 20.0)

# Load observation data
data_states = Matrix(CSV.read("data/KKT_trajectory_state.csv", DataFrame, header = false))
data_controls = Matrix(CSV.read("data/KKT_trajectory_control.csv", DataFrame, header = false))
data_states = ibr_solution.x
data_controls = ibr_solution.u
T = size(data_states,2)
data = reshape(vcat(data_states, data_controls), (n_states + n_controls)*T, 1)

# Compute initial values
x0 = data_states[:,1]
player_angles = data_states[[4,8],1]
n0 = x0[1:2] - x0[(1 + 4):(2 + 4)]
α = atan(n0[2],n0[1])

# Set costs
player_cost_models = map(enumerate(player_angles)) do (ii, player_angle)
    cost_model_p1 = CollisionAvoidanceGame.generate_player_cost_model_simple(;
        player_idx = ii,
        control_system,
        T,
        goal_position = unitvector(player_angle),
    )
end

# Setup decision variable for unknown parameter
ω = @variable(opt_model)

# Other decision variables
x   = @variable(opt_model, [1:n_states, 1:T])
u   = @variable(opt_model, [1:n_controls, 1:T])
λ_e = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])
λ_i = @variable(opt_model, [1:T])
s   = @variable(opt_model, [1:T], start = 0.001, lower_bound = 0.0)

# Warms start on decision variables
init = (;x = data_states, u = data_controls)
init_if_hasproperty!(λ_e, init, :λ_e)
init_if_hasproperty!(λ_i, init, :λ_i)
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

    # Make this more general 
    if player_idx == 1

        # Extract inequality constraints (without setting them)
        h = DynamicsModelInterface.add_inequality_constraints!(
            control_system.subsystems[player_idx], opt_model, x, u, 
            (; ω = ω, α = α)
            ; set = false
        )

        # Add Jacobians 
        dh = DynamicsModelInterface.add_inequality_jacobians!(
            control_system.subsystems[player_idx], opt_model, x, u, 
            (; ω = ω, α = α)
        )
        
        # Gradient of the Lagrangian wrt x is zero 
        @constraint(opt_model, 
            [t = 2:(T-1)],
            dJ.dx[:, t]' + λ_e[:, t - 1, player_idx]' - λ_e[:, t, player_idx]'*df.dx[:, :, t] + λ_i[t]*dh.dx[t, :]' .== 0
        )
        @constraint(opt_model, 
            dJ.dx[:, T]' + λ_e[:, T - 1, player_idx]' + λ_i[T]*dh.dx[T, :]' .== 0
        )   

        # Gradient of the Lagrangian wrt player's own inputs is zero
        @constraint(opt_model, 
        [t = 1:(T-1)], 
        dJ.du[player_inputs, t]' - λ_e[:,t,player_idx]'*df.du[:,player_inputs,t] .== 0)
        @constraint(opt_model, dJ.du[player_inputs, T]' .== 0)

        # Gradient of the Lagrangian wrt s is zero
        s_inv = @variable(opt_model, [1:T])
        @NLconstraint(opt_model, [t = 1:T], s_inv[t] == 1/s[t])
        @constraint(opt_model, [t = 1:T], -μ*s_inv[t] - λ_i[t] == 0)

        # Feasiblity of barrier-ed inequality constraints          
        @constraint(opt_model, [t = 1:T], h(t) - s[t] == 0)
    else
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
end

# Match equilibirum
@objective(opt_model, Min, sum(el -> el^2, data_states .- x))

time = @elapsed JuMP.optimize!(opt_model)
@info time

solution = get_values(;x, u, ω)

visualize_rotating_hyperplane(solution.x,(; ω = solution.ω, ρ = ρ, title = "Inverse"))