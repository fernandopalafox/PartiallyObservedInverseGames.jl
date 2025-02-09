module ForwardGame

import JuMP
import Ipopt
import ..DynamicsModelInterface
import ..JuMPUtils
import ..ForwardOptimalControl

using JuMP: @variable, @constraint, @objective, @NLconstraint
using UnPack: @unpack

export IBRGameSolver, KKTGameSolver, solve_game, KKTGameSolverBarrier

#================================ Iterated Best Open-Loop Response =================================#

struct IBRGameSolver end

# TODO: We could allow to pass an inner solver
function solve_game(
    ::IBRGameSolver,
    control_system,
    player_cost_models,
    x0,
    T;
    init = (; x = zeros(control_system.n_states, T), u = zeros(control_system.n_controls, T)),
    max_ibr_rounds = 10,
    ibr_convergence_tolerance = 1e-3,
    verbose = false,
    inner_solver_kwargs...,
)
    @unpack n_states, n_controls = control_system
    n_players = length(player_cost_models)

    last_ibr_solution = init
    last_player_solution = last_ibr_solution
    player_opt_models = resize!(JuMP.Model[], n_players)
    converged = false

    for i_ibr in 1:max_ibr_rounds
        for (player_idx, player_cost_model) in enumerate(player_cost_models)
            last_player_converged, last_player_solution, player_opt_models[player_idx] =
                ForwardOptimalControl.solve_optimal_control(
                    control_system,
                    player_cost_model,
                    x0,
                    T;
                    fixed_inputs = filter(i -> i ∉ player_cost_model.player_inputs, 1:n_controls),
                    init = last_player_solution,
                    inner_solver_kwargs...,
                    verbose,
                )
            @assert last_player_converged
        end

        converged =
            sum(Δu -> Δu^2, last_player_solution.u - last_ibr_solution.u) <=
            ibr_convergence_tolerance
        last_ibr_solution = last_player_solution

        if converged
            verbose && @info "Converged at ibr iterate: $i_ibr"
            break
        end
    end

    converged || @warn "IBR terminated pre-maturely."

    converged, last_ibr_solution, player_opt_models
end

#================================= Open-Loop KKT Nash Constraints ==================================#

struct KKTGameSolver end

function solve_game(
    ::KKTGameSolver,
    control_system,
    player_cost_models,
    x0,
    T;
    solver = Ipopt.Optimizer,
    solver_attributes = (; print_level = 3),
    init = (),
    match_equilibrium = nothing,
    verbose = false,
)
    n_players = length(player_cost_models)
    @unpack n_states, n_controls = control_system
    opt_model = JuMP.Model(solver)
    JuMPUtils.set_solver_attributes!(opt_model; solver_attributes...)
    # Decision Variables
    x = @variable(opt_model, [1:n_states, 1:T])
    u = @variable(opt_model, [1:n_controls, 1:T])
    λ = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])
    # Initialization
    JuMPUtils.init_if_hasproperty!(λ, init, :λ)
    JuMPUtils.init_if_hasproperty!(x, init, :x)
    JuMPUtils.init_if_hasproperty!(u, init, :u)
    # constraints
    @constraint(opt_model, x[:, 1] .== x0)
    DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)
    df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)
    for (player_idx, cost_model) in enumerate(player_cost_models)
        @unpack player_inputs, weights = cost_model
        dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)
        # KKT Nash constraints
        @constraint(
            opt_model,
            [t = 2:(T - 1)],
            dJ.dx[:, t] + λ[:, t - 1, player_idx] - (λ[:, t, player_idx]' * df.dx[:, :, t])' .== 0
        )
        @constraint(opt_model, dJ.dx[:, T] + λ[:, T - 1, player_idx] .== 0)
        @constraint(
            opt_model,
            [t = 1:(T - 1)],
            dJ.du[player_inputs, t] - (λ[:, t, player_idx]' * df.du[:, player_inputs, t])' .== 0
        )
        @constraint(opt_model, dJ.du[player_inputs, T] .== 0)
    end
    if !isnothing(match_equilibrium)
        @objective(
            opt_model,
            Min,
            sum(el -> el^2, x - match_equilibrium.x)
        )
    end
    time = @elapsed JuMP.optimize!(opt_model)
    verbose && @info time
    JuMPUtils.isconverged(opt_model), JuMPUtils.get_values(; x, u, λ), opt_model
end

struct KKTGameSolverBarrier end

function solve_game(
    ::KKTGameSolverBarrier,
    control_system,
    player_cost_models,
    x0,
    constraint_parameters,
    T;
    solver = Ipopt.Optimizer,
    solver_attributes = (; print_level = 5),
    init = (),
    μ = 0.01,
)

    # ---- Initial settings ---- 
    n_players = length(player_cost_models)
    n_couples = length(findall(constraint_parameters.adjacency_matrix))

    # ---- Setup solver ---- 

    # Solver
    opt_model = JuMP.Model(solver)
    JuMPUtils.set_solver_attributes!(opt_model; solver_attributes...)

    # Useful values
    @unpack n_states, n_controls = control_system

    # Other decision variables
    # Shared constraint decision variables 
    if !isnothing(constraint_parameters.adjacency_matrix) && n_couples > 0
        couples = findall(constraint_parameters.adjacency_matrix)
        # player_couple_list = [findall(couple -> couple[1] == player_idx, couples) for player_idx in 1:n_players] 
        player_couple_list = [findall(couple -> any(hcat(couple[1], couple[2]) .== player_idx), couples) for player_idx in 1:n_players] 

        λ_hyperplanes = @variable(opt_model, [1:length(couples), 2:T])
        # s_hyperplanes   = @variable(opt_model, [1:length(couples), 2:T], lower_bound = 1e-16) 
        s_hyperplanes   = @variable(opt_model, [1:length(couples), 2:T])

        JuMPUtils.init_if_hasproperty!(s_hyperplanes, init, :s_hyperplanes)
        JuMPUtils.init_if_hasproperty!(λ_hyperplanes, init, :λ_hyperplanes)

        @unpack ωs, αs, ρs = constraint_parameters
    end
    x   = @variable(opt_model, [1:n_states, 1:T])
    u   = @variable(opt_model, [1:n_controls, 1:T])
    λ_dynamics = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])

    # s_thrust_limits = @variable(opt_model,[1:n_controls, 1:T, 1:2], lower_bound = 1e-16)
    s_thrust_limits = @variable(opt_model,[1:n_controls, 1:T, 1:2])
    λ_thrust_limits = @variable(opt_model, [1:n_controls, 1:T, 1:2])

    # Warmstart
    JuMPUtils.init_if_hasproperty!(x, init, :x)
    JuMPUtils.init_if_hasproperty!(u, init, :u)
    JuMPUtils.init_if_hasproperty!(λ_dynamics, init,:λ_dynamics)
    JuMPUtils.init_if_hasproperty!(s_thrust_limits, init, :s_thrust_limits)
    JuMPUtils.init_if_hasproperty!(λ_thrust_limits, init, :λ_thrust_limits)

    # ---- Setup constraints ----

    # Initialize angles (only works if fully observable)
   
    if n_couples > 0
        θs = zeros(n_couples)
        for player_idx in 1:n_players
            for (couple_idx_local, couple) in enumerate(couples[player_couple_list[player_idx]])
                parameter_idx = player_couple_list[player_idx][couple_idx_local]
                idx_ego   = (1:2) .+ (couple[1] - 1)*Int(n_states/n_players)
                idx_other = (1:2) .+ (couple[2] - 1)*Int(n_states/n_players)
                x_ego = x0[idx_ego,1]
                x_other = x0[idx_other,1]
                x_diff = x_ego - x_other
                θ = atan(x_diff[2], x_diff[1])

                θs[parameter_idx] = θ
            end
        end
    end

    # Dynamics constraints
    @constraint(opt_model, x[:, 1] .== x0[:,1])
    DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)
    df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)

    # TEMPORARY: Thrust limits constraints (should be in definition of dynamics)
    for (player, cost_model) in enumerate(player_cost_models)
        @unpack player_inputs = cost_model
        player_control_indices = player_inputs
        player_n_controls = control_system.subsystems[player].n_controls
        u_max = control_system.subsystems[player].u_max
        player_controls = u[player_control_indices, :]
        player_s = s_thrust_limits[player_control_indices, :, :]
        player_λ = λ_thrust_limits[player_control_indices, :, :]

        # Primal feasibility 
        for control_idx in 1:player_n_controls
            @constraint(opt_model, [t = 1:T],  player_controls[control_idx, t] + u_max - player_s[control_idx, t, 1] == 0)
            @constraint(opt_model, [t = 1:T], -player_controls[control_idx, t] + u_max - player_s[control_idx, t, 2] == 0)
        end

        # Vanishing Lagrangian wrt s_thrust_limits
        @constraint(
            opt_model,
            [t = 1:T, control_idx = 1:player_n_controls],
            -player_s[control_idx, t, 1] * player_λ[control_idx, t, 1] - μ == 0
        )
        @constraint(
            opt_model,
            [t = 1:T, control_idx = 1:player_n_controls],
            -player_s[control_idx, t, 2] * player_λ[control_idx, t, 2] - μ == 0
        )
        # s_player_inv_1 = @variable(opt_model, [t = 1:T, control_idx = 1:player_n_controls])
        # s_player_inv_2 = @variable(opt_model, [t = 1:T, control_idx = 1:player_n_controls])
        # @NLconstraint(opt_model, [t = 1:T, control_idx = 1:player_n_controls], s_player_inv_1[t, control_idx] == 1 / player_s[control_idx, t, 1])
        # @NLconstraint(opt_model, [t = 1:T, control_idx = 1:player_n_controls], s_player_inv_2[t, control_idx] == 1 / player_s[control_idx, t, 2])
        # @constraint(opt_model, [t = 1:T, control_idx = 1:player_n_controls], -μ * s_player_inv_1[t, control_idx] - player_λ[control_idx, t, 1] == 0)
        # @constraint(opt_model, [t = 1:T, control_idx = 1:player_n_controls], -μ * s_player_inv_2[t, control_idx] - player_λ[control_idx, t, 2] == 0)

    end

    # KKT conditions
    used_couples = []
    for (player_idx, cost_model) in enumerate(player_cost_models)
        @unpack player_inputs, weights = cost_model
        dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)

        # Adjacency matrix denotes shared inequality constraint
        if n_couples > 0 && 
        !isempty(player_couple_list[player_idx])

            # Print adding KKT constraints to player $player_idx with couples $player_couple_list[player_idx]
            # println("Adding KKT constraints to player $player_idx with couples $(player_couple_list[player_idx])")

            # Extract relevant lms and slacks
            λ_hyperplanes_couples = λ_hyperplanes[player_couple_list[player_idx], :]
            # s_couples = s_hyperplanes[player_couple_list[player_idx], :]
            dhdx_container = []

            for (couple_idx_local, couple) in enumerate(couples[player_couple_list[player_idx]])
                couple_idx_global = player_couple_list[player_idx][couple_idx_local]

                # Extract relevant parameters
                parameters = (;
                    couple,
                    θ = θs[couple_idx_global],
                    ω = ωs[couple_idx_global],
                    α = αs[couple_idx_global],
                    ρ = ρs[couple_idx_global],
                    T_offset = 0,
                )

                # Extract shared constraint Jacobian 
                dhs = DynamicsModelInterface.add_shared_jacobian!(
                    control_system.subsystems[player_idx],
                    opt_model,
                    x,
                    u,
                    parameters,
                )
                # println("   Computing constraint Jacobian for couple $couple_idx_global: $couple")

                # Stack shared constraint Jacobian. 
                # One row per couple, timestep indexing along 3rd axis
                append!(dhdx_container, [dhs.dx])

                # Feasibility of barrier-ed constraints
                if couple_idx_global ∉ used_couples # Add constraint only if not already added
                    # Extract shared constraint for player couple 
                    hs = DynamicsModelInterface.add_shared_constraint!(
                        control_system.subsystems[player_idx],
                        opt_model,
                        x,
                        u,
                        parameters;
                        set = false,
                    )

                    # Enforce shared constraint feasibility
                    @constraint(opt_model, [t = 2:T], hs(t) - s_hyperplanes[couple_idx_global, t] == 0)
                    push!(used_couples, couple_idx_global) # Add constraint to list of used constraints
                    # println("   Adding shared constraint feasiblity for couple $couple_idx_global: $couple")

                    # ∇ₛL = -μ * s⁻¹ - λ_hyperplanes = 0
                    # s_couple_inv = @variable(opt_model, [t = 2:T])
                    # @NLconstraint(opt_model, [t = 2:T], s_couple_inv[t] == 1 / s_hyperplanes[couple_idx_global, t])
                    # @constraint(opt_model, [t = 2:T], -μ * s_couple_inv[t] - λ_hyperplanes[couple_idx_global, t] == 0)

                    # Equation 19.5b) in Ch.19 of Nocedal and Wright 
                    @constraint(opt_model, [t = 2:T], -s_hyperplanes[couple_idx_global, t] * λ_hyperplanes[couple_idx_global, t] - μ == 0)
                    # @warn "verify gradient of lagrangian is correct"
                end   
            end
            dhdx = vcat(dhdx_container...)

            # Gradient of the Lagrangian wrt x is zero 
            @constraint(
                opt_model,
                [t = 2:(T - 1)],
                dJ.dx[:, t]' + λ_dynamics[:, t - 1, player_idx]' -
                λ_dynamics[:, t, player_idx]' * df.dx[:, :, t] + λ_hyperplanes_couples[:, t].data' * dhdx[:, :, t] .== 0
            )
            @constraint(
                opt_model,
                dJ.dx[:, T]' + λ_dynamics[:, T - 1, player_idx]' + λ_hyperplanes_couples[:, T].data' * dhdx[:, :, T] .== 0
            ) 

            # Gradient of the Lagrangian wrt player's_hyperplanes own inputs is zero
            @constraint(
                opt_model,
                [t = 1:(T - 1)],
                dJ.du[player_inputs, t]' -
                λ_dynamics[:, t, player_idx]' * df.du[:, player_inputs, t] +
                (λ_thrust_limits[player_inputs, t, 1] - λ_thrust_limits[player_inputs, t, 2])'
                .== 0
            )
            @constraint(
                opt_model,
                dJ.du[player_inputs, T]' +
                (λ_thrust_limits[player_inputs, T, 1] - λ_thrust_limits[player_inputs, T, 2])' 
                .== 0
            )
        else
            # Adding non-shared constraints
            # println("Adding KKT constraints to player $player_idx with no shared constraints")

            # Gradient of the Lagrangian wrt x is zero 
            @constraint(
                opt_model,
                [t = 2:(T - 1)],
                dJ.dx[:, t]' + λ_dynamics[:, t - 1, player_idx]' -
                λ_dynamics[:, t, player_idx]' * df.dx[:, :, t] .== 0
            )
            @constraint(
                opt_model,
                dJ.dx[:, T]' + λ_dynamics[:, T - 1, player_idx]' .== 0
            ) 

            # Gradient of the Lagrangian wrt player's_hyperplanes own inputs is zero
            @constraint(
                opt_model,
                [t = 1:(T - 1)],
                dJ.du[player_inputs, t]' -
                λ_dynamics[:, t, player_idx]' * df.du[:, player_inputs, t] +
                (λ_thrust_limits[player_inputs, t, 1] - λ_thrust_limits[player_inputs, t, 2])'
                .== 0
            )
            @constraint(
                opt_model,
                dJ.du[player_inputs, T]' +
                (λ_thrust_limits[player_inputs, T, 1] - λ_thrust_limits[player_inputs, T, 2])' 
                .== 0
            )
        end
    
    end

    
    # Match equilbrium 
    # @objective(opt_model, Min, sum((x[:, :] .- init.x).^2))

    # Set all start values 
    JuMPUtils.init_model_if_hasproperty!(opt_model, init, :model)

    # Solve problem 
    time = @elapsed JuMP.optimize!(opt_model)
    # @info time
    n_couples > 0 ? solution = JuMPUtils.get_values(; x, u, λ_hyperplanes, λ_dynamics, s_hyperplanes, s_thrust_limits, λ_thrust_limits) : solution = JuMPUtils.get_values(; x, u, λ_dynamics, s_thrust_limits, λ_thrust_limits)

    # Print out difference of solved initial state vs. nominal initial state
    # println("Maximum error in initial state = $(maximum(abs.(solution.x[:,1] - x0[:,1])))")

    # Print difference vectors between solved and nominal initial state
    # println(
    #     "Δv = \n    P1 = $(solution.x[3:4,1] - x0[3:4,1])\n    P2 = $(solution.x[7:8,1] - x0[7:8,1])",
    # )
    
    (JuMPUtils.isconverged(opt_model), time, solution, opt_model)
end

end