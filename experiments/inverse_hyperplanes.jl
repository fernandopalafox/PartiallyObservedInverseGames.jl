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
using PartiallyObservedInverseGames.CostUtils
using PartiallyObservedInverseGames.JuMPUtils

include("utils/misc.jl")

let 
# ---- USER INPUT: Solver settings ----

ΔT = 0.1
n_players = 2
scale = 1
t_real = 5.0
T_activate_goalcost = 1

n_couples = 1

μ = 1.0
cmin = 1e-3
solver_attributes = (; max_wall_time = 20.0, print_level = 5)

data_states   = Matrix(CSV.read("data/f_di_s.csv", DataFrame, header = false))
data_inputs   = Matrix(CSV.read("data/f_di_c.csv", DataFrame, header = false))
init = (; x = data_states, u = data_inputs)
# init = nothing

# ---- Solve ---- 

control_system = TestDynamics.ProductSystem([TestDynamics.DoubleIntegrator(ΔT) for _ in 1:n_players])

@unpack n_states, n_controls = control_system
n_states_per_player = Int(n_states / n_players)

# Setup solver
opt_model = JuMP.Model(Ipopt.Optimizer)
set_solver_attributes!(opt_model; solver_attributes...)

# Compute values from data
T  = size(data_states,2)
as = [2*pi/n_players * (i-1) for i in 1:n_players]
as = [a > pi ? a - 2*pi : a for a in as]

# ---- USER INPUT: Setup unknown parameters ----

# Constraint parameters 
# uk_ωs = @variable(opt_model, [1:n_couples], lower_bound = -0.7, upper_bound = 0.7)
# uk_αs = @variable(opt_model, [1:n_couples], lower_bound = -pi,  upper_bound = pi)
# uk_ρs = @variable(opt_model, [1:n_couples], lower_bound = 0.0)

uk_ωs = @variable(opt_model, lower_bound = -0.6, upper_bound = 0.6)
uk_αs = @variable(opt_model, lower_bound = -pi,  upper_bound = pi)
uk_ρs = @variable(opt_model, lower_bound = 0.1)

ωs = [0.0 uk_ωs]
ρs = [0.0 uk_ρs]
αs = [0.0 uk_αs]
adj_mat = [false true; 
           false false]

# # 3p
# ωs = [0.0 uk_ωs[1] uk_ωs[2];
#       0.0 0.0      uk_ωs[3];
#       0.0 0.0      0.0]
# ρs = [0.0 uk_ρs[1] uk_ρs[2];
#       0.0 0.0      uk_ρs[3];
#       0.0 0.0      0.0]
# αs = [0.0 uk_αs[1] uk_αs[2];
#       0.0 0.0      uk_αs[3];
#       0.0 0.0      0.0]

# adj_mat = [false true  true; 
#            false false true;
#            false false false]

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
# adj_mat = [false true  true  true; 
#            false false true  true;
#            false false false true;
#            false false false false]

constraint_params = (; adj_mat, ωs, αs, ρs)
constraint_params = (;adj_mat = nothing)

# Cost models with dummy weights
player_cost_models = map(enumerate(as)) do (ii, a)
    cost_model_p1 = CollisionAvoidanceGame.generate_integrator_cost(;
        player_idx = ii,
        control_system,
        T,
        goal_position = scale*unitvector(a),
        weights = (; 
            state_proximity = 0.05, 
            state_goal      = 0.05,
            control_Δv      = 0.9),
        T_activate_goalcost,
        prox_min_regularization = 0.1
    )
end

# ---- Setup decision variables ----

# Shared constraint vars
if !isnothing(constraint_params.adj_mat)
    couples = findall(constraint_params.adj_mat)
    player_couples = [findall(couple -> couple[1] == player_idx, couples) for player_idx in 1:n_players] 

    λ_i_all = @variable(opt_model, [1:length(couples), 1:T])
    s_all   = @variable(opt_model, [1:length(couples), 1:T], lower_bound = 0.0, start = 1.5) 

    JuMPUtils.init_if_hasproperty!(λ_i_all, init, :λ_i_all)
    JuMPUtils.init_if_hasproperty!(s_all, init, :s_all)
end

# Other vars 
# player_weights =
#         [@variable(opt_model, [keys(cost_model.weights)]) for cost_model in player_cost_models]

x   = @variable(opt_model, [1:n_states, 1:T])
u   = @variable(opt_model, [1:n_controls, 1:T])
λ_e = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])

JuMPUtils.init_if_hasproperty!(λ_e, init, :λ_e)
JuMPUtils.init_if_hasproperty!(x, init, :x)
JuMPUtils.init_if_hasproperty!(u, init, :u)

# ---- Setup constraints ----

# Dynamics constraints
@constraint(opt_model, x[:, 1] .== data_states[:,1])
DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)
df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)

# KKT conditions
for (player_idx, cost_model) in enumerate(player_cost_models)
    # weights = player_weights[player_idx]
    @unpack player_inputs, weights = cost_model
    dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)

    # Adjacency matrix denotes shared inequality constraint
    if !isnothing(constraint_params.adj_mat) && 
       !isempty(player_couples[player_idx])

        # Extract relevant lms and slacks
        λ_i = λ_i_all[player_couples[player_idx], :]
        s = s_all[player_couples[player_idx], :]
        dhdx_container = []

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

        # Gradient of the Lagrangian wrt u is zero
        @constraint(
            opt_model,
            [t = 1:(T - 1)],
            dJ.du[player_inputs, t]' - λ_e[:, t, player_idx]' * df.du[:, player_inputs, t] .==
            0
        )
        @constraint(opt_model, dJ.du[player_inputs, T]' .== 0)
    end
 
end

# weight regularization
# for weights in player_weights
#     @constraint(opt_model, weights .>= cmin)
#     @constraint(opt_model, sum(weights) .== 1)
# end

# objective
@objective(
    opt_model,
    Min,
    sum(el -> el^2, x .- data_states)
)

# Solve problem 
time = @elapsed JuMP.optimize!(opt_model)
@info time

# solution =  merge(get_values(; x, u, uk_ωs, uk_αs, uk_ρs), (; player_weights = map(w -> CostUtils.namedtuple(JuMP.value.(w)), player_weights)))
solution =  get_values(; x, u, uk_ωs, uk_αs, uk_ρs)

# 2p 
k_ωs = [0.0 solution.uk_ωs[1]];
k_αs = [0.0 solution.uk_αs[1]];
k_ρs = [0.0 solution.uk_ρs[1]];

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
    (; adj_mat, ωs = k_ωs, αs = k_αs, ρs = k_ρs, title = "Inverse", n_players, n_states_per_player);
    koz = true, fps = 10.0
)

end