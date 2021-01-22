module ForwardGame

import JuMP

using JuMP: @variable, @constraint
using UnPack: @unpack
using ..ForwardOptimalControl: solve_optimal_control
using ..SolverUtils

export solve_ol_nash_ibr

#================================ Iterated Best Open-Loop Response =================================#

# TODO: make solution technique (e.g. KKT vs IBR) a dispatch argument
function solve_ol_nash_ibr(
    control_system,
    cost_models,
    x0,
    T;
    inner_solver_kwargs = (),
    max_ibr_rounds = 10,
)
    @unpack n_states, n_controls = control_system
    last_solution = (; x = zeros(n_states, T), u = zeros(n_controls, T))

    for i_ibr in 1:max_ibr_rounds, (player_idx, player_cost_model) in enumerate(cost_models)
        player_solution, player_model = solve_optimal_control(
            control_system,
            player_cost_model,
            x0,
            T;
            fix_inputs = filter(!=(player_idx), eachindex(cost_models)),
            init = last_solution,
            inner_solver_kwargs...,
        )

        if i_ibr > 1 && sum(x -> x^2, player_solution.x - last_solution.x) < 0.01
            @info "Convereged at IBR iterate $i_ibr"
            return player_solution
        end

        last_solution = player_solution
    end
end

#================================= Open-Loop KKT Nash Constraints ==================================#

function solve_ol_nash_kkt(
    control_system,
    cost_model,
    x0,
    T;
    solver = Ipopt.Optimizer,
    solver_attributes = (),
    silent = false,
)

    @unpack n_states = control_system
    model = JuMP.Model(solver)
    SolverUtils.set_solver_attributes!(model; silent, solver_attributes...)

    # TODO: fix the variable access here.
    x = @variable(model, x[1:n_states, 1:T])
    u1 = @variable(model, u1[1:1, 1:T])
    u2 = @variable(model, u2[1:1, 1:T])
    # TODO: think about where/if we have to share lagrange multipliers
    λ1 = @variable(model, λ1[1:n_states, 1:T])
    λ2 = @variable(model, λ2[1:n_states, 1:T])

    # TODO: fix ugly hack. Hard-code a dynamically feasible initial trajectory
    # x_init = reduce(1:(T - 1); init = x0) do x, t
    #     [x x[:, end] + [x0[3], 0, 0, 0]]
    # end
    JuMP.set_start_value.(λ1, 0)
    JuMP.set_start_value.(x, x_init)
    JuMP.set_start_value.(u1, u_init[1, :])
    JuMP.set_start_value.(u2, u_init[2, :])

    u = [u1; u2]

    # constraints
    # Initial condition
    @constraint(model, x[:, 1] .== x0)
    #TODO: continue here
    # Joint state feasibility
    control_system.add_dynamics_constraints!(model, x, u)
    df = control_system.add_dynamics_jacobians!(model, x, u)
    dJ1 = cost_model.objective_gradients_p1(x, u1; cost_model.weights)
    dJ2 = cost_model.objective_gradients_p2(x, u2; cost_model.weights)

    # TODO: figure out whether/which multipliers need to be shared
    # P1 KKT
    dL1dx = [dJ1.dx[:, t] + λ1[:, t - 1] - df.dx[:, :, t]' * λ1[:, t] for t in 2:T]
    dL1du1 = [dJ1.du1[:, t] - (df.du[:, 1:1, t]' * λ1[:, t]) for t in 2:T]
    @constraint(model, [t = eachindex(dL1dx)], dL1dx[t] .== 0)
    @constraint(model, [t = eachindex(dL1du1)], dL1du1[t] .== 0)
    # P2 KKT
    dL2dx = [dJ2.dx[:, t] + λ2[:, t - 1] - df.dx[:, :, t]' * λ2[:, t] for t in 2:T]
    dL2du2 = [dJ2.du2[:, t] - (df.du[:, 2:2, t]' * λ2[:, t]) for t in 2:T]
    @constraint(model, [t = eachindex(dL2dx)], dL2dx[t] .== 0)
    @constraint(model, [t = eachindex(dL2du2)], dL2du2[t] .== 0)

    @time JuMP.optimize!(model)
    SolverUtils.get_model_values(model, :x, :u1, :u2, :λ1, :λ2), model
end


end
