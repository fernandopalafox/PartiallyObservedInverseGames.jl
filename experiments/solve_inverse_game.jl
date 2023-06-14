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

# This script solves for the rotating hyperplane obstacle avoidance scenario 
# Assumptions
# - The number of players is 2
# - Data is formatted as a CSV file where 
#   - Every column corresponds to a time step
#   - Every row corresponds to a state variable
#   - States for player 1 are listed first, appended by states for player 2
#   - We know the players' objectives

solution = let
# Known parameters
player_angles = [0, pi/2]
μ = 0.1
n_players = 2
ρ = 0.25
opt_model = JuMP.Model(Ipopt.Optimizer)
JuMP.set_time_limit_sec(opt_model, 20.0)

# Compute initial positions
x0 = mapreduce(vcat, player_angles) do player_angle
    [unitvector(player_angle + pi); 0.1; player_angle + deg2rad(10)]
end

# Compute initial angle for the hyperplane normal 
n0 = x0[1:2] - x0[(1 + 4):(2 + 4)]
α = atan(n0[2],n0[1])

# Load data 
data_states = Matrix(CSV.read("data/trajectory_state.csv", DataFrame, header = false))
data_control = Matrix(CSV.read("data/trajectory_control.csv", DataFrame, header = false))
T = size(data_states,2)
n_states = size(data_states,1)
n_controls = size(data_control,1)
data = reshape(vcat(data_states, data_control), (n_states + n_controls)*T, 1)

# System dynamics
# control_system =
#     TestDynamics.ProductSystem([TestDynamics.HyperUnicycle(0.25, 0.0, ρ), TestDynamics.Unicycle(0.25)])

                                                    # TEMPORARY REMOVE LATER
                                                    control_system =
                                                    TestDynamics.ProductSystem([TestDynamics.Unicycle(0.25), TestDynamics.Unicycle(0.25)])
                                                    # control_system =
                                                    # TestDynamics.ProductSystem([TestDynamics.Unicycle(0.25), TestDynamics.Unicycle(0.25), TestDynamics.Unicycle(0.25)])

# Cost models 
player_cost_models = map(enumerate(player_angles)) do (ii, player_angle)
    cost_model_p1 = CollisionAvoidanceGame.generate_player_cost_model_simple(;
        player_idx = ii,
        control_system,
        T,
        goal_position = unitvector(player_angle),
    )
end

# Decision variables 
@unpack n_states, n_controls = control_system
x   = @variable(opt_model, [1:n_states, 1:T])
u   = @variable(opt_model, [1:n_controls, 1:T])
λ_e = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players]) # One for each eq constraints
# λ_i = @variable(opt_model, [1:T]) # One for each ineq constraints. Only one player uses it (for now)
# s   = @variable(opt_model, [1:T], start = 0.1) # One for each ineq constraints. Only one player uses it (for now) 
# ω   = @variable(opt_model) # Unknown parameter

# -------------- REMOVE THIS!!! For debugging purposes only --------------
# set_start_value.(x, data_states + 0.1*randn(size(data_states)))
# set_start_value.(u, data_control + 0.1*randn(size(data_control)))
# x   = @variable(opt_model, [1:4, 1:T])
# u   = @variable(opt_model, [1:2, 1:T])
# λ_e = @variable(opt_model, [1:4, 1:(T-1)]) # One for each eq constraints
# λ_i = zeros(T)
# s = zeros(T)
# x0 = x0[1:4]
x0 = @variable(opt_model, [1:n_states])
λ0 = @variable(opt_model, [1:n_states, 1:n_players])

# NOTE: MAKE THIS MORE GENERAL. SHOULD BE A LOOP AND SHOULDN'T HAVE TO GO SYSTEM BY SYSTEM

# Add dynamics constraints for all players
DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u) #ok

# Add dynamics gradients (can be done a single time and product_system.jl should handle it)
df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u) #ok 

# First constraint: initial condition (covers both players)
@constraint(opt_model, x[:, 1] .== x0) #ok

# For player 1: HyperUnicycle
player_idx = 1
control_system_single = control_system.subsystems[player_idx]
player_inputs = player_cost_models[player_idx].player_inputs
# ρ = control_system_single.ρ
index_offset = control_system_single.n_states

    # Add objective gradients 
    dJ = player_cost_models[player_idx].add_objective_gradients!(opt_model, x, u; player_cost_models[player_idx].weights)

    # Add hyperplane constraints for player 1
    # DynamicsModelInterface.add_inequality_constraints!(control_system_single, opt_model, x, u, (; ω = ω, α = α))

    # Add Jacobians and gradients
    # dh = DynamicsModelInterface.add_inequality_jacobians!(control_system_single, opt_model, x, u, (; ω = ω, α = α))      

    # Add KKT conditions as constraints

                                # TEMPORARY Gradient of the Lagrangian wrt x is zero 
                                @constraint(opt_model, 
                                    dJ.dx[:, 1]' - λ_e[:, 1, player_idx]'*df.dx[:, :, 1] + λ0[:, player_idx]' .== 0
                                )
                                @constraint(opt_model, 
                                    [t = 2:(T - 1)],
                                    dJ.dx[:, t]' + λ_e[:, t - 1, player_idx]' - λ_e[:, t, player_idx]'*df.dx[:, :, t] .== 0
                                )
                                @constraint(opt_model, 
                                    dJ.dx[:, T]' + λ_e[:, T - 1, player_idx]' .== 0
                                ) 

        # Gradient of the Lagrangian wrt x is zero 
        # @constraint(opt_model, 
        #     dJ.dx[:, 1]' - λ_e[:, 1, player_idx]'*df.dx[:, :, 1] - λ_i[1]*dh.dx[1, :]' .== 0
        # )
        # @constraint(opt_model, 
        #     [t = 2:(T-1)],
        #     dJ.dx[:, t]' + λ_e[:, t - 1, player_idx]' - λ_e[:, t, player_idx]'*df.dx[:, :, t] - λ_i[t]*dh.dx[t, :]' .== 0
        # )
        # @constraint(opt_model, 
        #     dJ.dx[:, T]' + λ_e[:, T - 1, player_idx]' - λ_i[T]*dh.dx[T, :]' .== 0
        # )    

        # Gradient of the Lagrangian wrt player's own inputs is zero
        @constraint(opt_model, [t = 1:(T-1)], dJ.du[player_inputs, t]' - λ_e[:,t,player_idx]'*df.du[:,player_inputs,t] .== 0)
        @constraint(opt_model, dJ.du[player_inputs, T]' .== 0)

        # # Gradient of the Lagrangian wrt s is zero
        # s_inv = @variable(opt_model, [1:T])
        # @NLconstraint(opt_model, [t = 1:T], s_inv[t] == 1/s[t])
        # @constraint(opt_model, [t = 1:T], -μ*s_inv[t] + λ_i[t] == 0)

        # # Feasiblity of barrier-ed inequality constraint
        #     # Hyperplane normal (using given initial conditions)
        #     n0 = x0[1:2] - x0[(1 + index_offset):(2 + index_offset)]
        #     α = atan(n0[2],n0[1])

        #     # Note indexing using (t-1) 
        #     n_cos = @variable(opt_model, [1:T])
        #     n_sin = @variable(opt_model, [1:T])
        #     @NLconstraint(opt_model, [t = 1:T], n_cos[t] == cos(α + ω * (t-1)))
        #     @NLconstraint(opt_model, [t = 1:T], n_sin[t] == sin(α + ω * (t-1)))
        #     function n(t)
        #     [n_cos[t],n_sin[t]]
        #     end

        #     # Intersection of hyperplane w/ KoZ
        #     function p(t)
        #         x_other = x[(1 + index_offset):(2 + index_offset), t]
        #         x_other + ρ .* n(t)
        #     end

        #     # Actual inequality constraint
        #     function h(t)
        #         n(t)' * (x[1:2, t] - p(t))
        #     end
            
        # @constraint(opt_model, [t = 1:T], h(t) - s[t] == 0)

        # Feasiblity of equality constraints (already taken care of when adding dynamics constraints)

# Main.@infiltrate

# For player 2: Unicycle
player_idx = 2
control_system_single = control_system.subsystems[player_idx]
player_inputs = player_cost_models[player_idx].player_inputs

    # Add objective gradients 
    dJ = player_cost_models[player_idx].add_objective_gradients!(opt_model, x, u; player_cost_models[player_idx].weights)
    
    # Add KKT conditions as constraints
        
        # Gradient of the Lagrangian wrt x is zero 
        @constraint(opt_model, 
            dJ.dx[:, 1]' - λ_e[:, 1, player_idx]'*df.dx[:, :, 1] .== 0
        )
        @constraint(opt_model, 
            [t = 2:(T-1)],
            dJ.dx[:, t]' + λ_e[:, t - 1, player_idx]' - λ_e[:, t, player_idx]'*df.dx[:, :, t] .== 0
        )
        @constraint(opt_model, 
            dJ.dx[:, T]' + λ_e[:, T - 1, player_idx]' .== 0
        )   

        # Gradient of the Lagrangian wrt player's own inputs is zero
        @constraint(opt_model, [t = 1:(T-1)], dJ.du[player_inputs, t]' - λ_e[:,t,player_idx]'*df.du[:,player_inputs,t] .== 0)
        @constraint(opt_model, dJ.du[player_inputs, T]' .== 0)

        # Feasiblity of equality constraints (already taken care of when adding dynamics constraints)

# TEMPORARY 
# # For player 2: Unicycle
# player_idx = 3
# control_system_single = control_system.subsystems[player_idx]
# player_inputs = player_cost_models[player_idx].player_inputs

#     # Add objective gradients 
#     dJ = player_cost_models[player_idx].add_objective_gradients!(opt_model, x, u; player_cost_models[player_idx].weights)
    
#     # Add KKT conditions as constraints
        
#         # Gradient of the Lagrangian wrt x is zero 
#         @constraint(opt_model, 
#             dJ.dx[:, 1]' - λ_e[:, 1, player_idx]'*df.dx[:, :, 1] .== 0
#         )
#         @constraint(opt_model, 
#             [t = 2:(T-1)],
#             dJ.dx[:, t]' + λ_e[:, t - 1, player_idx]' - λ_e[:, t, player_idx]'*df.dx[:, :, t] .== 0
#         )
#         @constraint(opt_model, 
#             dJ.dx[:, T]' + λ_e[:, T - 1, player_idx]' .== 0
#         )   

#         # Gradient of the Lagrangian wrt player's own inputs is zero
#         @constraint(opt_model, [t = 1:(T-1)], dJ.du[player_inputs, t]' - λ_e[:,t,player_idx]'*df.du[:,player_inputs,t] .== 0)
#         @constraint(opt_model, dJ.du[player_inputs, T]' .== 0)

#         # Feasiblity of equality constraints (already taken care of when adding dynamics constraints)

Main.@infiltrate

# Inverse objective: 
# Match the observed trajectory
# Reshape x and u into "observation vector"
y = reshape(vcat(x,u), (n_states + n_controls)*T, 1)
@objective(opt_model, Min, sum(el -> el^2, data .- y))

time = @elapsed JuMP.optimize!(opt_model)
@info time

# solution = JuMP.get_values(; x, u, λ_e, λ_i, s, ω)

get_values(; jump_vars...) = (; map(((k, v),) -> k => JuMP.value.(v), collect(jump_vars))...)

solution = get_values(; x, u, λ_e)

end