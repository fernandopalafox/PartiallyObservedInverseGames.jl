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
    constraint_params = (; adj_mat = nothing),
)

    n_players = length(player_cost_models)
    @unpack n_states, n_controls = control_system
    n_states_per_player = Int(n_states/n_players)

    opt_model = JuMP.Model(solver)
    JuMPUtils.set_solver_attributes!(opt_model; solver_attributes...)

    # Couples and vars for shared constraints
    if !isnothing(constraint_params.adj_mat)
        couples = findall(constraint_params.adj_mat)
        player_couples = [findall(couple -> couple[1] == player_idx, couples) for player_idx in 1:n_players] 
        λ_i_all = @variable(opt_model, [1:length(couples), 1:T])
    end

    # Decision Variables
    x = @variable(opt_model, [1:n_states, 1:T])
    u = @variable(opt_model, [1:n_controls, 1:T])
    λ_e     = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])

    # Initialization
    JuMPUtils.init_if_hasproperty!(λ_e, init, :λ_e)
    JuMPUtils.init_if_hasproperty!(x, init, :x)
    JuMPUtils.init_if_hasproperty!(u, init, :u)

    # constraints
    @constraint(opt_model, x[:, 1] .== x0)
    DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)
    df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)

    # for (player_idx, cost_model) in enumerate(player_cost_models)
    #     @unpack player_inputs, weights = cost_model
    #     dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)

    #     # KKT Nash constraints
    #     @constraint(
    #         opt_model,
    #         [t = 2:(T - 1)],
    #         dJ.dx[:, t] + λ[:, t - 1, player_idx] - (λ[:, t, player_idx]' * df.dx[:, :, t])' .== 0
    #     )
    #     @constraint(opt_model, dJ.dx[:, T] + λ[:, T - 1, player_idx] .== 0)

    #     @constraint(
    #         opt_model,
    #         [t = 1:(T - 1)],
    #         dJ.du[player_inputs, t] - (λ[:, t, player_idx]' * df.du[:, player_inputs, t])' .== 0
    #     )
    #     @constraint(opt_model, dJ.du[player_inputs, T] .== 0)
    # end

    for (player_idx, cost_model) in enumerate(player_cost_models)
        @unpack player_inputs, weights = cost_model
        dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)

        # Add terminal state constraint
        pos_idx = [1, 2] .+ (player_idx - 1)*n_states_per_player
        @constraint(opt_model, x[pos_idx, T] .== cost_model.goal_position)

        # Adjacency matrix denotes shared inequality constraint
        if !isnothing(constraint_params.adj_mat) && 
            !isempty(player_couples[player_idx])

            # Extract relevant lms and slacks
            λ_i  = λ_i_all[player_couples[player_idx], :]

            dhdx_container = []
            for (couple_idx, couple) in enumerate(couples[player_couples[player_idx]])
                params = (;couple, constraint_params...)

                # Extract shared constraint for player couple 
                _ = DynamicsModelInterface.add_shared_constraint!(
                                        control_system.subsystems[player_idx], opt_model, x, u, params; set = true
                                    )

                # Extract shared constraint Jacobian 
                dhs = DynamicsModelInterface.add_shared_jacobian!(
                                        control_system.subsystems[player_idx], opt_model, x, u, params
                                    )

                # Stack shared constraint Jacobian. 
                # One row per couple, timestep indexing along 3rd axis
                append!(dhdx_container, [dhs.dx]) 

            end
            dhdx = vcat(dhdx_container...)
        
            # Gradient of the Lagrangian wrt x is zero 
            @constraint(opt_model, 
            [t = 2:(T-1)],
                dJ.dx[:, t]' + λ_e[:, t - 1, player_idx]' - λ_e[:, t, player_idx]'*df.dx[:, :, t] + λ_i[:,t]'*dhdx[:, :, t] .== 0
            )
            @constraint(opt_model, 
                dJ.dx[:, T]' + λ_e[:, T - 1, player_idx]' + λ_i[:,T]'*dhdx[:, :, T] .== 0
            )   

            # Gradient of the Lagrangian wrt player's own inputs is zero
            @constraint(opt_model, 
            [t = 1:(T-1)], 
                dJ.du[player_inputs, t]' - λ_e[:, t, player_idx]'*df.du[:,player_inputs,t] .== 0)
            @constraint(opt_model, dJ.du[player_inputs, T]' .== 0)
            
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
            # println("Added KKT conditions for player $player_idx")
        end
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

    var_vals = !isnothing(constraint_params.adj_mat) ? 
    JuMPUtils.get_values(; x, u, λ_e, λ_i_all) : JuMPUtils.get_values(; x, u, λ_e)

    JuMPUtils.isconverged(opt_model), var_vals, opt_model
end

struct KKTGameSolverBarrier end

function solve_game(
    ::KKTGameSolverBarrier,
    control_system,
    player_cost_models,
    x0,
    T;
    solver = Ipopt.Optimizer,
    solver_attributes = (; print_level = 5),
    init = (),
    match_equilibrium = nothing,
    verbose = false,
    constraint_params = (; adj_mat = nothing),
)

    # TEMPORARY PARAMETERS
    μ = 0.00001

    n_players = length(player_cost_models)
    @unpack n_states, n_controls = control_system
    n_states_per_player = Int(n_states/n_players)

    opt_model = JuMP.Model(solver)
    JuMPUtils.set_solver_attributes!(opt_model; solver_attributes...)

    # Indices used to match each player with its relevant couples
    if !isnothing(constraint_params.adj_mat)
        couples = findall(constraint_params.adj_mat)
        player_couples = [findall(couple -> couple[1] == player_idx, couples) for player_idx in 1:n_players] 
        λ_i_all = @variable(opt_model, [1:length(couples), 1:T]) # Assumes constraints apply to all timesteps. All constraints the same
        s_all   = @variable(opt_model, [1:length(couples), 1:T], start = 0.001, lower_bound = 0.0)     
    end

    # Decision Variables
    x       = @variable(opt_model, [1:n_states, 1:T])
    u       = @variable(opt_model, [1:n_controls, 1:T])
    λ_e     = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])
      
    # Initialization
    # JuMPUtils.init_if_hasproperty!(λ_e, init, :λ_e)
    # JuMPUtils.init_if_hasproperty!(λ_i, init, :λ_i)
    JuMPUtils.init_if_hasproperty!(x, init, :x)
    JuMPUtils.init_if_hasproperty!(u, init, :u)

    # Initial state constraint
    @constraint(opt_model, x[:, 1] .== x0)

    # Dynamics constraints
    DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)
    df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)

    for (player_idx, cost_model) in enumerate(player_cost_models)
        @unpack player_inputs, weights = cost_model
        dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)

        # Add terminal state constraint
        pos_idx = [1, 2] .+ (player_idx - 1)*n_states_per_player
        @constraint(opt_model, x[pos_idx, T] .== cost_model.goal_position)

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

                # Extract shared constraint Jacobian 
                dhs = DynamicsModelInterface.add_shared_jacobian!(
                                        control_system.subsystems[player_idx], opt_model, x, u, params
                                    )

                # Stack shared constraint Jacobian. 
                # One row per couple, timestep indexing along 3rd axis
                append!(dhdx_container, [dhs.dx]) 

                # Feasibility of barrier-ed constraints
                @constraint(opt_model, [t = 1:T], hs(t) - s[couple_idx, t] == 0.0)
            end
            dhdx = vcat(dhdx_container...)
        
            # Gradient of the Lagrangian wrt x is zero 
            @constraint(opt_model, 
            [t = 2:(T-1)],
                dJ.dx[:, t]' + λ_e[:, t - 1, player_idx]' - λ_e[:, t, player_idx]'*df.dx[:, :, t] + λ_i[:,t]'*dhdx[:, :, t] .== 0
            )
            @constraint(opt_model, 
                dJ.dx[:, T]' + λ_e[:, T - 1, player_idx]' + λ_i[:,T]'*dhdx[:, :, T] .== 0
            )   

            # Gradient of the Lagrangian wrt player's own inputs is zero
            @constraint(opt_model, 
            [t = 1:(T-1)], 
                dJ.du[player_inputs, t]' - λ_e[:, t, player_idx]'*df.du[:,player_inputs,t] .== 0)
            @constraint(opt_model, dJ.du[player_inputs, T]' .== 0)

            # Gradient of the Lagrangian wrt s is zero
            n_slacks     = length(s)
            λ_i_reshaped = reshape(λ_i', (1, :))
            s_reshaped   = reshape(s' , (1, :))
            s_inv        = @variable(opt_model, [2:n_slacks])
            @NLconstraint(opt_model, [t = 2:n_slacks], s_inv[t] == 1/s_reshaped[t])
            @constraint(opt_model, [t = 2:n_slacks], -μ*s_inv[t] - λ_i_reshaped[t] == 0)
            
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
            # println("Added KKT conditions for player $player_idx")
        end
     
    end

    # if !isnothing(match_equilibrium)
    #     @objective(
    #         opt_model,
    #         Min,
    #         sum(el -> el^2, x - match_equilibrium.x)
    #     )
    # end

    time = @elapsed JuMP.optimize!(opt_model)
    verbose && @info time

    JuMPUtils.isconverged(opt_model), JuMPUtils.get_values(; x, u, λ_e, λ_i_all, s_all), opt_model
end

end