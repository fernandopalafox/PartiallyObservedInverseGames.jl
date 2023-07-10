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
using PartiallyObservedInverseGames.ForwardGame: KKTGameSolver, solve_game

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

# User inputs 
T_activate_goalcost = 1
ΔT = 0.1
n_players = 4
scale = 1

μ = 0.00001
solver_attributes = (; print_level = 5, expect_infeasible_problem = "no")
time_limit = 80.0
n_couples = 6

t_real_obs = 3.0
t_real_pred = 3.0

# Presumed control system 
control_system = TestDynamics.ProductSystem([TestDynamics.DoubleIntegrator(ΔT) for _ in 1:n_players])

# Observed data (all of it)
data_states = Matrix(CSV.read("data/f_di_s.csv", DataFrame, header = false))
data_inputs = Matrix(CSV.read("data/f_di_c.csv", DataFrame, header = false))


# ---- USER INPUT: Setup unknown parameters ----

# Useful stuff
@unpack n_states, n_controls = control_system
n_states_per_player = Int(n_states / n_players)
T_obs         = Int(round(t_real_obs / ΔT))
T_pred        = Int(round(t_real_pred / ΔT))  # Must be at least 2 (or ∇xL will be break)
T_tot         = T_obs + T_pred
as            = data_states[n_states_per_player:n_states_per_player:n_states,1]

opt_model, constraint_params, player_cost_models, unknowns = let 
    # Horizon 
    T = T_obs

    # Setup solver
    opt_model = JuMP.Model(Ipopt.Optimizer)
    set_solver_attributes!(opt_model; solver_attributes...)
    JuMP.set_time_limit_sec(opt_model, time_limit)

    # Constraint parameters 
    uk_ωs = @variable(opt_model, [1:n_couples], lower_bound = -0.7, upper_bound = 0.7)
    uk_αs = @variable(opt_model, [1:n_couples], lower_bound = -pi,  upper_bound = pi)
    uk_ρs = @variable(opt_model, [1:n_couples], lower_bound = 0.1)

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

    # Cost and weights 
    uk_weights = @variable(opt_model, [1:n_players, 1:4], lower_bound = 0.0)
    @constraint(opt_model, [p = 1:n_players], sum(uk_weights[p,:]) == 1) #regularization 

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

    # player_cost_models = map(enumerate(as)) do (ii, a)
    #     cost_model_p1 = CollisionAvoidanceGame.generate_integrator_cost(;
    #         player_idx = ii,
    #         control_system,
    #         T,
    #         goal_position = scale*unitvector(a),
    #         weights = (; 
    #             state_proximity = 0.05, 
    #             state_goal = 1,
    #             control_Δvx = 20, 
    #             control_Δvy = 20),
    #         T_activate_goalcost,
    #         prox_min_regularization = 0.1
    #     )
    # end

    # Params tuple 
    constraint_params = (; adj_mat, ωs, αs, ρs)

    # Unknowns 
    unknowns = (;uk_ωs, uk_αs, uk_ρs, uk_weights)
    # unknowns = (;uk_ωs, uk_αs, uk_ρs)

    opt_model, constraint_params, player_cost_models, unknowns 
end

# Ugly. Fix this.
if !isnothing(constraint_params.adj_mat)
    couples = findall(constraint_params.adj_mat)
    player_couples = [findall(couple -> couple[1] == player_idx, couples) for player_idx in 1:n_players] 
end

# --- Solve for unknown parameters ---
# Input = opt_model

solution_obs = let 
    # Horizon
    T = T_obs

    # Decision variables 
    x       = data_states[:, 1:T_obs]
    u       = data_inputs[:, 1:T_obs]
    λ_e     = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])
    λ_i_all = @variable(opt_model, [1:length(couples), 1:T]) 
    s_all   = @variable(opt_model, [1:length(couples), 1:T], lower_bound = 0.0)     

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
                params = (;couple, T_offset = 0, constraint_params...)

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

    # Run optimization 
    time = @elapsed JuMP.optimize!(opt_model)
    @info time

    merge((;x,u), get_values(;unknowns...))
end

# Kinda nasty, fix this. 
ωs = [0.0 solution_obs.uk_ωs[1] solution_obs.uk_ωs[2] solution_obs.uk_ωs[4];
      0.0 0.0                   solution_obs.uk_ωs[3] solution_obs.uk_ωs[5];
      0.0 0.0                   0.0                   solution_obs.uk_ωs[6];
      0.0 0.0                   0.0                   0.0]
αs = [0.0 solution_obs.uk_αs[1] solution_obs.uk_αs[2] solution_obs.uk_αs[4];
      0.0 0.0                   solution_obs.uk_αs[3] solution_obs.uk_αs[5];
      0.0 0.0                   0.0                   solution_obs.uk_αs[6];
      0.0 0.0                   0.0                   0.0]
ρs = [0.0 solution_obs.uk_ρs[1] solution_obs.uk_ρs[2] solution_obs.uk_ρs[4];
      0.0 0.0                   solution_obs.uk_ρs[3] solution_obs.uk_ρs[5];
      0.0 0.0                   0.0                   solution_obs.uk_ρs[6];
      0.0 0.0                   0.0                   0.0]
adj_mat = constraint_params.adj_mat

# --- Trajectory warmstart using learned parameters ---
# Input: horizon, weights, x0

solution_pred_warmstart, player_cost_models = let 
    # Horizon
    T = T_pred

    # Initial value 
    x0 = DynamicsModelInterface.next_x(control_system, data_states[:, T_obs], data_inputs[:, T_obs])

    # New params tuple 
    constraint_params = (; adj_mat, ωs, αs, ρs)

    # Learned cost functions 
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

    # Generate warmstart data 
    _, solution_pred_warmstart, _ = 
        solve_game(KKTGameSolver(), control_system, player_cost_models, x0, T; 
        solver = Ipopt.Optimizer, 
        solver_attributes = (;max_wall_time = time_limit, print_level = 5))

    solution_pred_warmstart, player_cost_models
end

# --- Predict trajectory ---
# Input: horizon, player cost models, constraint parameters, x0

solution_pred = let 
    # Horizon 
    T = T_pred
    constraint_params = (; adj_mat, ωs, αs, ρs) 

    # Initial position 
    x0 = solution_pred_warmstart.x[:,1]

    # Setup new solver 
    opt_model = JuMP.Model(Ipopt.Optimizer)
    set_solver_attributes!(opt_model; solver_attributes...)
    JuMP.set_time_limit_sec(opt_model, time_limit)

    if !isnothing(constraint_params.adj_mat)
        couples = findall(constraint_params.adj_mat)
        player_couples = [findall(couple -> couple[1] == player_idx, couples) for player_idx in 1:n_players] 
         # Assumes constraints apply to all timesteps. All constraints the same
    end

    # Decision variables
    x       = @variable(opt_model, [1:n_states, 1:T])
    u       = @variable(opt_model, [1:n_controls, 1:T])
    λ_e     = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players]) #Only need these for the T_pred steps
    λ_i_all = @variable(opt_model, [1:length(couples), 1:T])
    s_all   = @variable(opt_model, [1:length(couples), 1:T], lower_bound = 0.0, start = 1.5)   

    # Initialize trajectories
    JuMP.set_start_value.(x[:, 2:end], solution_pred_warmstart.x[:, 2:end])
    JuMP.set_start_value.(u[:, 2:end], solution_pred_warmstart.u[:, 2:end])

    # Initial position must be consistent with x0
    # @constraint(opt_model, x[:, 1] .== x0)
    JuMP.fix.(x[:,1], x0)

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
            λ_i = λ_i_all[player_couples[player_idx], :]
            s = s_all[player_couples[player_idx], :]
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

    # Run optimization 
    time = @elapsed JuMP.optimize!(opt_model)
    @info time

    get_values(;x, s_all)
end

# ---- Plot ----

        
visualize_obs_pred(
    hcat(solution_obs.x, solution_pred.x), T_obs,
    (; ΔT, adj_mat, ωs, αs, ρs, title = "4p observe + predict", n_players, n_states_per_player);
    koz = true,
    fps = 10,
)