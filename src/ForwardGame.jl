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
    μ = 0.00001,
)

    # ---- Initial settings ---- 
    n_players = length(control_system.subsystems)
    n_couples = length(findall(constraint_parameters.adjacency_matrix))

    # ---- Setup solver ---- 

    # Solver
    opt_model = JuMP.Model(solver)
    JuMPUtils.set_solver_attributes!(opt_model; solver_attributes...)

    # Useful values
    @unpack n_states, n_controls = control_system
    @unpack ωs, αs, ρs = constraint_parameters

    # Other decision variables
    x   = @variable(opt_model, [1:n_states, 1:T])
    u   = @variable(opt_model, [1:n_controls, 1:T])
    λ_e = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])

    JuMPUtils.init_if_hasproperty!(x, init, :x)
    JuMPUtils.init_if_hasproperty!(u, init, :u)
    JuMPUtils.init_if_hasproperty!(λ_e, init,:λ_e)

    # Shared constraint decision variables 
    if !isnothing(constraint_parameters.adjacency_matrix) && n_couples > 0
        couples = findall(constraint_parameters.adjacency_matrix)
        # player_couple_list = [findall(couple -> couple[1] == player_idx, couples) for player_idx in 1:n_players] 
        player_couple_list = [findall(couple -> any(hcat(couple[1], couple[2]) .== player_idx), couples) for player_idx in 1:n_players] 


        λ_i = @variable(opt_model, [1:length(couples), 1:T])
        s   = @variable(opt_model, [1:length(couples), 1:T], lower_bound = 0.0) 

        JuMPUtils.init_if_hasproperty!(s, init, :s)
        JuMPUtils.init_if_hasproperty!(λ_i, init,:λ_i)
    end

    # ---- Match equilibrium ----

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

    # KKT conditions
    used_couples = []
    for (player_idx, cost_model) in enumerate(player_cost_models)
        @unpack player_inputs, weights = cost_model
        dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)

        # Adjacency matrix denotes shared inequality constraint
        if n_couples > 0 && 
        !isempty(player_couple_list[player_idx])

            # Print adding KKT constraints to player $player_idx with couples $player_couple_list[player_idx]
            println("Adding KKT constraints to player $player_idx with couples $(player_couple_list[player_idx])")

            # Extract relevant lms and slacks
            λ_i_couple = λ_i[player_couple_list[player_idx], :]
            s_couple = s[player_couple_list[player_idx], :]
            dhdx_container = []

            for (couple_idx_local, couple) in enumerate(couples[player_couple_list[player_idx]])
                couple_idx_global = player_couple_list[player_idx][couple_idx_local]

                # Switch couple indices so that player_idx is always first. 
                # This is to ensure constraint Jacobian is correct
                if couple[1] != player_idx
                    couple = CartesianIndex(couple[2], couple[1])
                end

                # Extract relevant parameters
                parameter_idx = player_couple_list[player_idx][couple_idx_local]
                parameters = (;
                    couple,
                    θ = θs[parameter_idx],
                    ω = ωs[parameter_idx],
                    α = αs[parameter_idx],
                    ρ = ρs[parameter_idx],
                    T_offset = 0,
                )
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
                println("   Computing constraint Jacobian for couple $couple_idx_global: $couple")

                # Stack shared constraint Jacobian. 
                # One row per couple, timestep indexing along 3rd axis
                append!(dhdx_container, [dhs.dx])
                # Feasibility of barrier-ed constraints
                # @constraint(opt_model, [t = 1:T], hs(t) - s_couple[couple_idx_local, t] == z[player_idx, t])
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
                    @constraint(opt_model, [t = 1:T], hs(t) - s_couple[couple_idx_local, t] == 0)
                    push!(used_couples, couple_idx_global) # Add constraint to list of used constraints
                    println("   Adding shared constraint feasiblity for couple $couple_idx_global: $couple")
                end   

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
            if player_idx == 1
                n_s_couple = length(s_couple)
                λ_i_couple_reshaped = reshape(λ_i_couple', (1, :))
                s_couple_reshaped = reshape(s_couple', (1, :))
                s_couple_inv = @variable(opt_model, [2:n_s_couple])
                @NLconstraint(opt_model, [t = 2:n_s_couple], s_couple_inv[t] == 1 / s_couple_reshaped[t])
                @constraint(opt_model, [t = 2:n_s_couple], -μ * s_couple_inv[t] - λ_i_couple_reshaped[t] == 0)
            end

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

    # Solve problem 
    time = @elapsed JuMP.optimize!(opt_model)
    # @info time
    n_couples > 0 ? solution = JuMPUtils.get_values(; x, u, λ_i, λ_e, s) : solution = JuMPUtils.get_values(; x, u, λ_e)
    
    (JuMPUtils.isconverged(opt_model), time, solution)
end

end