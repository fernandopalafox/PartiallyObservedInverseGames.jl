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

solution = let 
    # ---- USER INPUT: Solver settings ----

    ΔT = 0.1
    n_players = 2
    scale = 1
    t_real = 5.0
    T_activate_goalcost = 1

    n_couples = 1

    μ = 1.0
    cmin = 1e-3
    solver_attributes = (; max_wall_time = 60.0, print_level = 5)

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
    ωs = @variable(opt_model, [1:n_couples], lower_bound = -0.7, upper_bound = 0.7)
    αs = @variable(opt_model, [1:n_couples], lower_bound = -pi,  upper_bound = pi)
    ρs = @variable(opt_model, [1:n_couples], lower_bound = 0.1, upper_bound = 2.0, start = 0.1)

    adjacency_matrix = [false true; 
               false false]

    # adjacency_matrix = [false true  true; 
    #                     false false true;
    #                     false false false]

    # adjacency_matrix = [false true  true  true; 
    #            false false true  true;
    #            false false false true;
    #            false false false false]

    constraint_parameters = (; adjacency_matrix, ωs, αs, ρs)
    # constraint_parameters = (;adjacency_matrix = nothing)

    # Cost models with dummy weights
    player_cost_models = map(enumerate(as)) do (ii, a)
        cost_model_p1 = CollisionAvoidanceGame.generate_integrator_cost(;
            player_idx = ii,
            control_system,
            T,
            goal_position = scale*unitvector(a),
            weights = (; 
                state_proximity = -1, 
                state_goal      = -1,
                control_Δv      = -1),
            T_activate_goalcost,
            prox_min_regularization = 0.1
        )
    end

    # ---- Setup decision variables ----

    # Shared constraint vars
    if !isnothing(constraint_parameters.adjacency_matrix)
        couples = findall(constraint_parameters.adjacency_matrix)
        player_couples = [findall(couple -> couple[1] == player_idx, couples) for player_idx in 1:n_players] 

        λ_i = @variable(opt_model, [1:length(couples), 1:T])
        s   = @variable(opt_model, [1:length(couples), 1:T], lower_bound = 0.0, start = 1.5) 

        JuMPUtils.init_if_hasproperty!(λ_i, init, :λ_i)
        JuMPUtils.init_if_hasproperty!(s,   init, :s)
    end

    # Other vars 
    player_weights =
            [@variable(opt_model, [keys(cost_model.weights)]) for cost_model in player_cost_models]
    x   = @variable(opt_model, [1:n_states, 1:T])
    u   = @variable(opt_model, [1:n_controls, 1:T])
    λ_e = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])

    JuMPUtils.init_if_hasproperty!(λ_e, init, :λ_e)
    JuMPUtils.init_if_hasproperty!(x,   init, :x)
    JuMPUtils.init_if_hasproperty!(u,   init, :u)

    # ---- Setup constraints ----

    # Dynamics constraints
    @constraint(opt_model, x[:, 1] .== data_states[:,1])
    DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)
    df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)

    # KKT conditions
    for (player_idx, cost_model) in enumerate(player_cost_models)
        weights = player_weights[player_idx]
        @unpack player_inputs = cost_model
        dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)

        # Adjacency matrix denotes shared inequality constraint
        if !isnothing(constraint_parameters.adjacency_matrix) && 
        !isempty(player_couples[player_idx])

            # Extract relevant lms and slacks
            λ_i_couple = λ_i[player_couples[player_idx], :]
            s_couple = s[player_couples[player_idx], :]
            dhdx_container = []

            for (couple_idx, couple) in enumerate(couples[player_couples[player_idx]])
                parameters = (; couple, ω = ωs[couple_idx], α = αs[couple_idx], ρ = ρs[couple_idx], T_offset = 0)
                # Extract shared constraint for player couple 
                hs = DynamicsModelInterface.add_shared_constraint!(
                    control_system.subsystems[player_idx],
                    opt_model,
                    x,
                    u,
                    parameters;
                    set = false,
                )
                # Extract shared constraint Jacobian 
                dhs = DynamicsModelInterface.add_shared_jacobian!(
                    control_system.subsystems[player_idx],
                    opt_model,
                    x,
                    u,
                    parameters,
                )
                # Stack shared constraint Jacobian. 
                # One row per couple, timestep indexing along 3rd axis
                append!(dhdx_container, [dhs.dx])
                # Feasibility of barrier-ed constraints
                @constraint(opt_model, [t = 1:T], hs(t) - s_couple[couple_idx, t] == 0.0)
            end
            dhdx = vcat(dhdx_container...)

            # Gradient of the Lagrangian wrt x is zero 
            @constraint(
                opt_model,
                [t = 2:(T - 1)],
                dJ.dx[:, t]' + λ_e[:, t - 1, player_idx]' -
                λ_e[:, t, player_idx]' * df.dx[:, :, t] + λ_i_couple[:, t]' * dhdx[:, :, t] .== 0
            )
            @constraint(
                opt_model,
                dJ.dx[:, T]' + λ_e[:, T - 1, player_idx]' + λ_i_couple[:, T]' * dhdx[:, :, T] .== 0
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
            n_s_couple = length(s_couple)
            λ_i_couple_reshaped = reshape(λ_i_couple', (1, :))
            s_couple_reshaped = reshape(s_couple', (1, :))
            s_couple_inv = @variable(opt_model, [2:n_s_couple])
            @NLconstraint(opt_model, [t = 2:n_s_couple], s_couple_inv[t] == 1 / s_couple_reshaped[t])
            @constraint(opt_model, [t = 2:n_s_couple], -μ * s_couple_inv[t] - λ_i_couple_reshaped[t] == 0)

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
    for weights in player_weights
        @constraint(opt_model, weights .>= cmin)
        @constraint(opt_model, sum(weights) .== 1)
    end

    # objective
    @objective(
        opt_model,
        Min,
        sum(el -> el^2, x .- data_states)
    )

    # Solve problem 
    time = @elapsed JuMP.optimize!(opt_model)
    @info time

    solution =  merge(
                    get_values(; x, u, ωs, αs, ρs),
                    (; player_weights = map(w -> CostUtils.namedtuple(JuMP.value.(w)), player_weights))
                )

    visualize_rotating_hyperplanes(
        solution.x,
        (; adjacency_matrix, ωs = solution.ωs, αs = solution.αs, ρs = solution.ρs , title = "Inverse", n_players, n_states_per_player);
        koz = true, fps = 10.0
    )

    Main.@infiltrate

end