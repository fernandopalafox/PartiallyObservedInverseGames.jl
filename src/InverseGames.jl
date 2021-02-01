module InverseGames

import JuMP
import Ipopt
import ..DynamicsModelInterface
import ..SolverUtils
import ..InverseOptimalControl

using JuMP: @variable, @constraint, @objective
using UnPack: @unpack

export solve_inverse_game

struct InverseIBRSolver end

# TODO: allow for partial and noisy state observations
# TODO: This probably does not work so well if the initial strategy for the players does not match
# the observation well.
function solve_inverse_game(
    ::InverseIBRSolver,
    x̂;
    observation_model,
    control_system,
    player_cost_models,
    # TODO: dirty hack
    u_init = nothing,
    inner_solver_kwargs = (),
    max_ibr_rounds = 5,
    ibr_convergence_tolerance = 0.01,
)
    @unpack n_controls = control_system
    n_players = length(player_cost_models)

    last_ibr_solution = (; x = x̂, u = u_init, λ = nothing)
    last_player_solution = last_ibr_solution
    player_opt_models = resize!(JuMP.Model[], n_players)
    # TODO: dirty hack
    player_weights = Any[nothing for _ in 1:n_players]
    converged = false

    for i_ibr in 1:max_ibr_rounds
        for (player_idx, player_cost_model) in enumerate(player_cost_models)
            cost_model = player_cost_models[player_idx]
            last_player_solution, player_opt_models[player_idx] =
                InverseOptimalControl.solve_inverse_optimal_control(
                    x̂;
                    control_system,
                    cost_model,
                    observation_model,
                    fixed_inputs = filter(i -> i ∉ player_cost_model.player_inputs, 1:n_controls),
                    init = (;
                        weights = player_weights[player_idx],
                        # TODO: think about whether we should initialize with `x̂` or with
                        # `last_player_solution.x` here
                        # TODO: think about whether we should also hand over λ or whether it is
                        # better to start from λ = 0 to make sure we don't enforce the contraints
                        # immediately.
                        x = last_player_solution.x,
                        u = last_player_solution.u,
                        λ = last_player_solution.λ,
                    ),
                    inner_solver_kwargs...,
                )

            player_weights[player_idx] = last_player_solution.weights
        end

        converged =
            sum(Δu -> Δu^2, last_player_solution.u .- last_ibr_solution.u) <=
            ibr_convergence_tolerance
        last_ibr_solution = last_player_solution

        if converged
            @info "Converged at ibr iterate: $i_ibr"
            break
        end
    end

    converged || @warn "IBR terminated pre-maturely."

    converged, last_ibr_solution, player_opt_models, player_weights
end

#================================ Inverse Games via KKT constraints ================================#

struct InverseKKTSolver end

# TODO: allow for partial and noisy state observations
function solve_inverse_game(
    ::InverseKKTSolver,
    x̂;
    control_system,
    player_cost_models,
    init = (),
    solver = Ipopt.Optimizer,
    solver_attributes = (),
    silent = false,
    cmin = 1e-5,
    max_observation_error_sq = nothing,
)

    T = size(x̂)[2]
    n_players = length(player_cost_models)
    @unpack n_states, n_controls = control_system

    opt_model = JuMP.Model(solver)
    SolverUtils.set_solver_attributes!(opt_model; silent, solver_attributes...)

    # Decision Variables
    player_weights =
        [@variable(opt_model, [keys(cost_model.weights)]) for cost_model in player_cost_models]
    x = @variable(opt_model, [1:n_states, 1:T])
    u = @variable(opt_model, [1:n_controls, 1:T])
    λ = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])

    x0 = @variable(opt_model, [1:n_states])
    λ0 = @variable(opt_model, [1:n_states, 1:n_players])

    # Initialization
    JuMP.set_start_value.(x, x̂)
    SolverUtils.init_if_hasproperty!(u, init, :u)
    SolverUtils.init_if_hasproperty!(λ, init, :λ)

    # # TODO: think about initialization for player weights
    # for weights in player_weights
    #     JuMP.set_start_value.(weights, 1 / length(weights))
    # end

    # constraints
    DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)
    df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)

    @constraint(opt_model, x[:, 1] .== x0)

    for (player_idx, cost_model) in enumerate(player_cost_models)
        weights = player_weights[player_idx]
        @unpack player_inputs = cost_model
        dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)

        # KKT Nash constraints
        @constraint(
            opt_model,
            dJ.dx[:, 1] - (λ[:, 1, player_idx]' * df.dx[:, :, 1])' + λ0[:, player_idx] .== 0
        )
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

        # regularization
        @constraint(opt_model, weights .>= cmin)
        @constraint(opt_model, sum(weights) .== 1)
    end

    # TODO: dirty hack
    # Only search in a reasonable neighborhood of hte demonstration.
    if !isnothing(max_observation_error_sq)
        @constraint(opt_model, (x - x̂) .^ 2 .<= max_observation_error_sq)
    end

    # The inverse objective: match the observed demonstration
    @objective(opt_model, Min, sum(el -> el^2, x .- x̂))

    @time JuMP.optimize!(opt_model)
    merge(
        SolverUtils.get_values(; x, u, λ),
        (; player_weights = map(w -> JuMP.value.(w), player_weights)),
    ),
    opt_model
end

end
