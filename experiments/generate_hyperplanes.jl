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

let

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
t_real = 4.0

n_couples = 6

μ = 0.00001
solver_attributes = (; print_level = 5, expect_infeasible_problem = "no")
max_wall_time = 180.0

# Setup warmstart
data_states = Matrix(CSV.read("data/f_di_s.csv", DataFrame, header = false))
data_inputs = Matrix(CSV.read("data/f_di_c.csv", DataFrame, header = false))
x0 = data_states[:, 1]
init = (;x = data_states, u = data_inputs)

# ---- Solve ---- 

# Presumed system dynamics 
control_system = TestDynamics.ProductSystem([TestDynamics.DoubleIntegrator(ΔT) for _ in 1:n_players])

# Useful values
@unpack n_states, n_controls = control_system
n_states_per_player = Int(n_states / n_players)
T  = Int(round(t_real / ΔT))
T_obs = 0
as = [2*pi/n_players * (i-1) for i in 1:n_players] # angles
as = [a > pi ? a - 2*pi : a for a in as]

# Setup solver
opt_model = JuMP.Model(Ipopt.Optimizer)
set_solver_attributes!(opt_model; solver_attributes...)
JuMP.set_time_limit_sec(opt_model, max_wall_time)

# ---- USER INPUT: Setup unknown parameters ----

# Constraint parameters 
# uk_ωs = @variable(opt_model, [1:n_couples], lower_bound = -0.1, upper_bound = 0.1)
# uk_αs = @variable(opt_model, [1:n_couples], lower_bound = -pi,  upper_bound = pi)
# uk_ρs = @variable(opt_model, [1:n_couples], lower_bound = 0.0)

# 2p 
# ωs = [0.0 uk_ωs[1]
#       0.0 0.0]
# ρs = [0.0 uk_ρs[1]
#       0.0 0.0]
# αs = [0.0 uk_αs[1]
#       0.0 0.0]
# adj_mat = [false true 
#            false false]

# 3p
# ωs = [0.0 uk_ωs[1] uk_ωs[2];
#       0.0 0.0      uk_ωs[3];
#       0.0 0.0      0.0]
# ρs = [0.0 uk_ρs[1] uk_ρs[2];
#       0.0 0.0      uk_ρs[3];
#       0.0 0.0      0.0]
# αs = [0.0 uk_αs[1] uk_αs[2];
#       0.0 0.0      uk_αs[3];
#       0.0 0.0      0.0]

# 4p
# ωs = [0.0 uk_ωs[1] uk_ωs[2] uk_ωs[4];
#       0.0 0.0      uk_ωs[3] uk_ωs[5];
#       0.0 0.0      0.0      uk_ωs[6];
#       0.0 0.0      0.0      0.0]
# ρs = [0.0 uk_ρs[1] uk_ρs[2] uk_ρs[4];
#       0.0 0.0      uk_ρs[3] uk_ρs[5];
#       0.0 0.0      0.0      uk_ρs[6];
#       0.0 0.0      0.0      0.0]
# αs = [0.0 uk_αs[1] uk_αs[2] uk_αs[4];
#       0.0 0.0      uk_αs[3] uk_αs[5];
#       0.0 0.0      0.0      uk_αs[6];
#       0.0 0.0      0.0      0.0]
adj_mat = [false true  true  true; 
           false false true  true;
           false false false true;
           false false false false]

# 3p exfiltrated (results from a run of inverse_hyperplanes.jl)
ωs = safehouse.k_ωs
ρs = safehouse.k_ρs
αs = safehouse.k_αs

constraint_params = (; adj_mat, ωs, αs, ρs)

# Cost parameters
player_cost_models = map(enumerate(as)) do (ii, a)
    cost_model_p1 = CollisionAvoidanceGame.generate_integrator_cost(;
        player_idx = ii,
        control_system,
        T,
        goal_position = scale*unitvector(a),
        weights = (; 
            state_proximity = 0.05, 
            state_goal = 1,
            control_Δvx = 20, 
            control_Δvy = 20),
        T_activate_goalcost,
        prox_min_regularization = 0.1
    )
end

# ---- Setup decision variables ----

if !isnothing(constraint_params.adj_mat)
    couples = findall(constraint_params.adj_mat)
    player_couples = [findall(couple -> couple[1] == player_idx, couples) for player_idx in 1:n_players] 
end

x       = @variable(opt_model, [1:n_states, 1:T])
u       = @variable(opt_model, [1:n_controls, 1:T])
λ_e     = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])
λ_i_all = @variable(opt_model, [1:length(couples), 1:T]) # Assumes constraints apply to all timesteps. All constraints the same
s_all   = @variable(opt_model, [1:length(couples), 1:T], lower_bound = 0.0)   

# Initialize
init_if_hasproperty!(x, init, :x)
init_if_hasproperty!(u, init, :u)

# Initial state constraint
@constraint(opt_model, x[:, 1] .== x0)
# JuMP.fix.(x[:,1], x0)

# Dynamics constraints
DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)
df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)

# KKT conditions 
for (player_idx, cost_model) in enumerate(player_cost_models)

    @unpack player_inputs, weights = cost_model
    dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)

    # Add terminal state constraint
    # pos_idx = [1, 2] .+ (player_idx - 1)*n_states_per_player
    # @constraint(opt_model, x[pos_idx, T] .== cost_model.goal_position)

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

        # Gradient of the Lagrangian wrt s is zero
        @constraint(
            opt_model,
            [t = 1:(T - 1)],
            dJ.du[player_inputs, t]' - λ_e[:, t, player_idx]' * df.du[:, player_inputs, t] .==
            0
        )
        @constraint(opt_model, dJ.du[player_inputs, T]' .== 0)
    end
end

# Solve problem 
time = @elapsed JuMP.optimize!(opt_model)
@info time
# solution = get_values(;x, u, s_all, uk_ωs, uk_αs, uk_ρs)
solution = get_values(;x, u)

# 2p 
# k_ωs = [0.0 solution.uk_ωs[1]]
# k_ρs = [0.0 solution.uk_ρs[1]]
# k_αs = [0.0 solution.uk_αs[1]]

# 3p 
# k_ωs = [0.0 solution.uk_ωs[1] solution.uk_ωs[2];
#         0.0 0.0               solution.uk_ωs[3];
#         0.0 0.0               0.0]
# k_αs = [0.0 solution.uk_αs[1] solution.uk_αs[2];
#         0.0 0.0               solution.uk_αs[3];
#         0.0 0.0               0.0]
# k_ρs = [0.0 solution.uk_ρs[1] solution.uk_ρs[2];
#         0.0 0.0               solution.uk_ρs[3];
#         0.0 0.0               0.0]

# 3p exfiltrated
k_ωs = safehouse.k_ωs
k_αs = safehouse.k_αs
k_ρs = safehouse.k_ρs

# 4p
# k_ωs = [0.0 solution.uk_ωs[1] solution.uk_ωs[2] solution.uk_ωs[4];
#         0.0 0.0               solution.uk_ωs[3] solution.uk_ωs[5];
#         0.0 0.0               0.0               solution.uk_ωs[6];
#         0.0 0.0               0.0               0.0]
# k_αs = [0.0 solution.uk_αs[1] solution.uk_αs[2] solution.uk_αs[4];
#         0.0 0.0               solution.uk_αs[3] solution.uk_αs[5];
#         0.0 0.0               0.0               solution.uk_αs[6];
#         0.0 0.0               0.0               0.0]
# k_ρs = [0.0 solution.uk_ρs[1] solution.uk_ρs[2] solution.uk_ρs[4];
#         0.0 0.0               solution.uk_ρs[3] solution.uk_ρs[5];
#         0.0 0.0               0.0               solution.uk_ρs[6];
#         0.0 0.0               0.0               0.0]

visualize_rotating_hyperplanes(
    solution.x,
    (; adj_mat, ωs = k_ωs, αs = k_αs, ρs = k_ρs, title = "Hyperplanes", n_players, n_states_per_player);
    koz = true, fps = 10
)

Main.@infiltrate

end