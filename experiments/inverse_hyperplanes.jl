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

# ---- USER INPUT: Solver settings ----

T_activate_goalcost = 1
ΔT = 0.1
n_players = 4
scale = 1

μ = 0.00001
solver_attributes = (; print_level = 5, expect_infeasible_problem = "no")
n_couples = 6

T_trim = T

# Setup warmstart
init = nothing

# ---- Solve ---- 

# Presumed system dynamics
# control_system =
#     TestDynamics.ProductSystem([TestDynamics.HyperUnicycle(ΔT, 0.0, 0.0), 
#                                 TestDynamics.HyperUnicycle(ΔT, 0.0, 0.0),
#                                 TestDynamics.HyperUnicycle(ΔT, 0.0, 0.0),
#                                 TestDynamics.Unicycle(ΔT)])

control_system = TestDynamics.ProductSystem([TestDynamics.DoubleIntegrator(ΔT) for _ in 1:n_players])

@unpack n_states, n_controls = control_system
n_states_per_player = Int(n_states / n_players)

# Setup solver
opt_model = JuMP.Model(Ipopt.Optimizer)
set_solver_attributes!(opt_model; solver_attributes...)
JuMP.set_time_limit_sec(opt_model, 80.0)

# Load observation data
data_states_full   = Matrix(CSV.read("data/f_di_s.csv", DataFrame, header = false))
data_controls_full = Matrix(CSV.read("data/f_di_c.csv", DataFrame, header = false))
data_states   = data_states_full[:,1:T_trim]
data_controls = data_controls_full[:,1:T_trim]

# Compute values from data
T             = size(data_states,2)
x0            = data_states[:,1]
as            = data_states[n_states_per_player:n_states_per_player:n_states,1]

# ---- USER INPUT: Setup unknown parameters ----

# Constraint parameters 
uk_ωs = @variable(opt_model, [1:n_couples], lower_bound = -0.7, upper_bound = 0.7)
uk_αs = @variable(opt_model, [1:n_couples], lower_bound = -pi,  upper_bound = pi)
uk_ρs = @variable(opt_model, [1:n_couples], lower_bound = 0.05, upper_bound = 10)

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

# adj_mat = [false true  true; 
#            false false true;
#            false false false]

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

constraint_params = (; adj_mat, ωs, αs, ρs)

# Cost parameters
uk_weights = @variable(opt_model, [1:n_players, 1:4], lower_bound = 0.0)
@constraint(opt_model, [p = 1:n_players], sum(uk_weights[p,:]) == 1) #regularization 

# Costs
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

# # Costs
# player_cost_models = map(enumerate(as)) do (ii, a)
#     cost_model_p1 = CollisionAvoidanceGame.generate_integrator_cost(;
#         player_idx = ii,
#         control_system,
#         T,
#         goal_position = scale*unitvector(a),
#         weights = (; 
#             state_proximity = 0.05, 
#             state_goal = 1,
#             control_Δvx = 1, 
#             control_Δvy = 1),
#         T_activate_goalcost,
#         prox_min_regularization = 0.1
#     )
# end

# ---- Setup decision variables ----

if !isnothing(constraint_params.adj_mat)
    couples = findall(constraint_params.adj_mat)
    player_couples = [findall(couple -> couple[1] == player_idx, couples) for player_idx in 1:n_players] 
end

x       = data_states
u       = data_controls
λ_e     = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])
λ_i_all = @variable(opt_model, [1:length(couples), 1:T]) # Assumes constraints apply to all timesteps. All constraints the same
s_all   = @variable(opt_model, [1:length(couples), 1:T], start = 0.001, lower_bound = 0.0)     

# ---- Warmstart on decision variables ----
init_if_hasproperty!(λ_e, init, :λ_e)
init_if_hasproperty!(λ_i_all, init, :λ_i_all)
init_if_hasproperty!(s_all, init, :s_all)

# Dynamics constraints
DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)
df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)

for (player_idx, cost_model) in enumerate(player_cost_models)
    @unpack player_inputs, weights = cost_model
    dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)

    # Adjacency matrix denotes shared inequality constraint
    if !isnothing(constraint_params.adj_mat) && 
       !isempty(player_couples[player_idx])

        # Extract relevant lms and slacks
        λ_i  = λ_i_all[player_couples[player_idx], :]
        s    = s_all[player_couples[player_idx], :]

        dhdx_container = []
        for (couple_idx, couple) in enumerate(couples[player_couples[player_idx]])
            params = (;couple, constraint_params...)

            # Extract shared constraint for player couple 
            hs = DynamicsModelInterface.add_shared_constraint!(
                                    control_system.subsystems[player_idx], opt_model, x, u, params; set = false
                                )

            # Feasibility of barrier-ed constraints
            @constraint(opt_model, [t = 1:T], hs(t) - s[couple_idx, t] == 0.0)
        end

        # Gradient of the Lagrangian wrt s is zero
        n_slacks     = length(s)
        λ_i_reshaped = reshape(λ_i', (1, :))
        s_reshaped   = reshape(s' , (1, :))
        s_inv        = @variable(opt_model, [2:n_slacks])
        @NLconstraint(opt_model, [t = 2:n_slacks], s_inv[t] == 1/s_reshaped[t])
        @constraint(opt_model, [t = 2:n_slacks], -μ*s_inv[t] - λ_i_reshaped[t] == 0)
    end
 
end

# Solve problem 
time = @elapsed JuMP.optimize!(opt_model)
@info time

solution = merge((;x,u), get_values(;uk_ωs, uk_αs, uk_ρs, uk_weights))
# solution = merge((;x,u), get_values(;uk_ωs, uk_αs, uk_ρs))

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

visualize_rotating_hyperplanes(
    solution.x,
    (; adj_mat, ωs = k_ωs, αs = k_αs, ρs = k_ρs, title = "Inverse", n_players, n_states_per_player);
    koz = true, fps = 2.5
)