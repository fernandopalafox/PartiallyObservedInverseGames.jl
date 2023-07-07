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
using PartiallyObservedInverseGames.TrajectoryVisualization: visualize_obs_pred

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

# ---- USER INPUT: Solver settings ----

T_activate_goalcost = 1
ΔT = 0.1
n_players = 4
scale = 1

μ = 0.00001
solver_attributes = (; print_level = 5, expect_infeasible_problem = "no")
time_limit = 80.0
n_couples = 6
T_obs = 15
T_hor = 4  # Must be at least 2 (or ∇xL will be break)

control_system = TestDynamics.ProductSystem([TestDynamics.DoubleIntegrator(ΔT) for _ in 1:n_players])

init = nothing

# ---- Solve ---- 

# Useful stuff
@unpack n_states, n_controls = control_system
n_states_per_player = Int(n_states / n_players)

# Setup solver
opt_model = JuMP.Model(Ipopt.Optimizer)
set_solver_attributes!(opt_model; solver_attributes...)
JuMP.set_time_limit_sec(opt_model, time_limit)

# Load observation data
data_states = Matrix(CSV.read("data/f_di_s.csv", DataFrame, header = false))
data_inputs = Matrix(CSV.read("data/f_di_c.csv", DataFrame, header = false))

# Compute values from data
T_tot         = T_obs + T_hor
x0            = data_states[:,1]
as            = data_states[n_states_per_player:n_states_per_player:n_states,1]

# ---- USER INPUT: Setup unknown parameters ----

# Constraint parameters 
uk_ωs = @variable(opt_model, [1:n_couples], lower_bound = -0.7, upper_bound = 0.7)
uk_αs = @variable(opt_model, [1:n_couples], lower_bound = -pi,  upper_bound = pi)
uk_ρs = @variable(opt_model, [1:n_couples], lower_bound = 0.05, upper_bound = 10)

# 4p
ωs = [0.0 uk_ωs[1] uk_ωs[2] uk_ωs[4];
      0.0 0.0      uk_ωs[3] uk_ωs[5];
      0.0 0.0      0.0      uk_ωs[6];
      0.0 0.0      0.0      0.0]
ρs = [0.0 uk_ρs[1] uk_ρs[2] uk_ρs[4];
      0.0 0.0      uk_ρs[3] uk_ρs[5];
      0.0 0.0      0.0      uk_ρs[6];
      0.0 0.0      0.0      0.0]
αs = [0.0 uk_αs[1] uk_αs[2] uk_αs[4];
      0.0 0.0      uk_αs[3] uk_αs[5];
      0.0 0.0      0.0      uk_αs[6];
      0.0 0.0      0.0      0.0]
adj_mat = [false true  true  true; 
           false false true  true;
           false false false true;
           false false false false]

# Cost weights 
uk_weights = @variable(opt_model, [1:n_players, 1:4], lower_bound = 0.0)
@constraint(opt_model, [p = 1:n_players], sum(uk_weights[p,:]) == 1) #regularization 

# Params tuple 
constraint_params = (; adj_mat, ωs, αs, ρs)

# ---- Setup cost models ----

# Setup costs 
player_cost_models = map(enumerate(as)) do (ii, a)
    cost_model_p1 = CollisionAvoidanceGame.generate_integrator_cost(;
        player_idx = ii,
        control_system,
        T,
        goal_position = scale*unitvector(a),
        weights = (; 
            state_proximity = uk_weights[ii,1],
            state_goal      = uk_weights[ii,2],
            control_Δvx     = uk_weights[ii,3], 
            control_Δvy     = uk_weights[ii,4]),
        T_activate_goalcost,
        prox_min_regularization = 0.1
    )
end

# ---- Setup decision variables ----

if !isnothing(constraint_params.adj_mat)
    couples = findall(constraint_params.adj_mat)
    player_couples = [findall(couple -> couple[1] == player_idx, couples) for player_idx in 1:n_players] 
end

x       = @variable(opt_model, [1:n_states, 1:T_tot])
u       = @variable(opt_model, [1:n_controls, 1:T_tot])
λ_e     = @variable(opt_model, [1:n_states, 1:(T_tot - 1), 1:n_players])
λ_i_all = @variable(opt_model, [1:length(couples), 1:T_tot]) # Assumes constraints apply to all timesteps. All constraints the same
s_all   = @variable(opt_model, [1:length(couples), 1:T_tot], start = 0.001, lower_bound = 0.0)    

x_obs   = x[:,1:T_obs]
x_pred  = x[:,T_obs + 1:end]
u_obs   = u[:,1:T_obs]
u_pred  = u[:,T_obs + 1:end]

# ---- Solve inverse problem ----

# Fix observerd variables 
JuMP.fix.(x_obs, data_states[:,1:T_obs])
JuMP.fix.(u_obs, data_inputs[:,1:T_obs])

# KKT constraints 
function add_constraints_obs!(control_system, opt_model, x, u, constraint_params)
    T = size(x,2)

    # Dynamics constraints
    DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)

    # KKT conditions 
    for (player_idx, cost_model) in enumerate(player_cost_models)
        @unpack player_inputs, weights = cost_model

        # Adjacency matrix denotes shared inequality constraint
        if !isnothing(constraint_params.adj_mat) && !isempty(player_couples[player_idx])

            # Extract relevant lms and slacks
            λ_i = λ_i_all[player_couples[player_idx], 1:T]
            s = s_all[player_couples[player_idx], 1:T]

            for (couple_idx, couple) in enumerate(couples[player_couples[player_idx]])
                params = (; couple, T_offset = 0, constraint_params...)

                # Extract shared constraint for player couple 
                hs = DynamicsModelInterface.add_shared_constraint!(
                    control_system.subsystems[player_idx],
                    opt_model,
                    x,
                    u,
                    params;
                    set = false,
                )

                # Feasibility of barrier-ed constraints
                @constraint(opt_model, [t = 1:T], hs(t) - s[couple_idx, t] == 0.0)
            end

            # Gradient of the Lagrangian wrt s is zero
            n_slacks = length(s)
            λ_i_reshaped = reshape(λ_i', (1, :))
            s_reshaped = reshape(s', (1, :))
            s_inv = @variable(opt_model, [2:n_slacks])
            @NLconstraint(opt_model, [t = 2:n_slacks], s_inv[t] == 1 / s_reshaped[t])
            @constraint(opt_model, [t = 2:n_slacks], -μ * s_inv[t] - λ_i_reshaped[t] == 0)
        end
    end
end

function add_constraints_pred!(control_system, opt_model, x, u, constraint_params, T_obs)
    T = size(x, 2)

    # Dynamics constraints
    DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)
    df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)

    # KKT conditions 
    for (player_idx, cost_model) in enumerate(player_cost_models)

        @unpack player_inputs, weights = cost_model
        dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)

        # Adjacency matrix denotes shared inequality constraint
        if !isnothing(constraint_params.adj_mat) && !isempty(player_couples[player_idx])
            
            # Extract relevant lms and slacks
            λ_i = λ_i_all[player_couples[player_idx], T_obs + 1:end]
            s = s_all[player_couples[player_idx], T_obs + 1:end]
            dhdx_container = []

            for (couple_idx, couple) in enumerate(couples[player_couples[player_idx]])
                params = (; couple, T_offset = T_obs, constraint_params...)
                # Extract shared constraint for player couple 
                hs = DynamicsModelInterface.add_shared_constraint!(
                    control_system.subsystems[player_idx],
                    opt_model,
                    x,
                    u,
                    params;
                    set = false,
                )
                # Extract shared constraint Jacobian 
                dhs = DynamicsModelInterface.add_shared_jacobian!(
                    control_system.subsystems[player_idx],
                    opt_model,
                    x,
                    u,
                    params,
                )
                # Stack shared constraint Jacobian. 
                # One row per couple, timestep indexing along 3rd axis
                append!(dhdx_container, [dhs.dx])
                # Feasibility of barrier-ed constraints
                @constraint(opt_model, [t = 1:T], hs(t) - s[couple_idx, t] == 0.0)
            end
            dhdx = vcat(dhdx_container...)

            # Gradient of the Lagrangian wrt x is zero 
            @constraint(
                opt_model,
                [t = 2:(T - 1)],
                dJ.dx[:, t]' + λ_e[:, t - 1, player_idx]' -
                λ_e[:, t, player_idx]' * df.dx[:, :, t] + λ_i[:, t]' * dhdx[:, :, t] .== 0
            )
            @constraint(
                opt_model,
                dJ.dx[:, T]' + λ_e[:, T - 1, player_idx]' + λ_i[:, T]' * dhdx[:, :, T] .== 0
            )
            # Gradient of the Lagrangian wrt player's own inputs is zero
            @constraint(
                opt_model,
                [t = 1:(T - 1)],
                dJ.du[player_inputs, t]' - λ_e[:, t, player_idx]' * df.du[:, player_inputs, t] .==
                0
            )
            @constraint(opt_model, dJ.du[player_inputs, T]' .== 0)
            # Gradient of the Lagrangian wrt s is zero
            n_slacks = length(s)
            λ_i_reshaped = reshape(λ_i', (1, :))
            s_reshaped = reshape(s', (1, :))
            s_inv = @variable(opt_model, [2:n_slacks])
            @NLconstraint(opt_model, [t = 2:n_slacks], s_inv[t] == 1 / s_reshaped[t])
            @constraint(opt_model, [t = 2:n_slacks], -μ * s_inv[t] - λ_i_reshaped[t] == 0)

        else
            # KKT Nash constraints
            @constraint(
                opt_model,
                [t = 2:(T - 1)],
                dJ.dx[:, t] + λ_e[:, t - 1, player_idx] -
                (λ_e[:, t, player_idx]' * df.dx[:, :, t])' .== 0
            )
            @constraint(opt_model, dJ.dx[:, T] + λ_e[:, T - 1, player_idx] .== 0)
            @constraint(
                opt_model,
                [t = 1:(T - 1)],
                dJ.du[player_inputs, t] - (λ_e[:, t, player_idx]' * df.du[:, player_inputs, t])' .==
                0
            )
            @constraint(opt_model, dJ.du[player_inputs, T] .== 0)
        end
    end
end
add_constraints_obs!(control_system, opt_model, x_obs, u_obs, constraint_params) 
add_constraints_pred!(control_system, opt_model, x_pred, u_pred, constraint_params, T_obs)
DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, hcat(x_obs[:,end],x_pred[:,1]), u_obs[:,end])

# ---- Run optimization ----
time = @elapsed JuMP.optimize!(opt_model)
@info time
solution = get_values(;x, u, uk_ωs, uk_αs, uk_ρs, uk_weights)

# ---- Plot ----

# 4p
k_ωs = [0.0 solution.uk_ωs[1] solution.uk_ωs[2] solution.uk_ωs[4];
        0.0 0.0               solution.uk_ωs[3] solution.uk_ωs[5];
        0.0 0.0               0.0               solution.uk_ωs[6];
        0.0 0.0               0.0               0.0]
k_αs = [0.0 solution.uk_αs[1] solution.uk_αs[2] solution.uk_αs[4];
        0.0 0.0               solution.uk_αs[3] solution.uk_αs[5];
        0.0 0.0               0.0               solution.uk_αs[6];
        0.0 0.0               0.0               0.0]
k_ρs = [0.0 solution.uk_ρs[1] solution.uk_ρs[2] solution.uk_ρs[4];
        0.0 0.0               solution.uk_ρs[3] solution.uk_ρs[5];
        0.0 0.0               0.0               solution.uk_ρs[6];
        0.0 0.0               0.0               0.0]
        
visualize_obs_pred(
    solution.x, T_obs,
    (; adj_mat, ωs = k_ωs, αs = k_αs, ρs = k_ρs, title = "4p", n_players, n_states_per_player);
    koz = true,
    fps = 2.5,
)