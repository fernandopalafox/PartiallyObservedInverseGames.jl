module ForwardOptimalControl

using ..DynamicsModelInterface: DynamicsModelInterface
using ..JuMPUtils: JuMPUtils
using Ipopt: Ipopt
using JuMP: JuMP
using LinearAlgebra: norm

using JuMP: @variable, @constraint, @objective
using UnPack: @unpack

export forward_quadratic_objective, linear_dynamics_constraints, solve_lqr, solve_optimal_control

#============================================= LQ case =============================================#

"The performance index for the forward optimal control problem."
function forward_quadratic_objective(x, u; Q, R)
    T = last(size(x))
    sum(x[:, t]' * Q * x[:, t] + u[:, t]' * R * u[:, t] for t in 1:T)
end

function linear_dynamics_constraints(x, u; A, B)
    reduce(hcat, ((x[:, t + 1] - A[t] * x[:, t] - B[t] * u[:, t]) for t in axes(x)[2][1:(end - 1)]))
end

"Solves a forward LQR problem using JuMP."
function solve_lqr(
    A,
    B,
    Q,
    R,
    x0,
    T;
    solver = Ipopt.Optimizer,
    solver_attributes = (; print_level = 3),
)
    n_states, n_controls = size(only(unique(B)))
    opt_model = JuMP.Model(solver)
    JuMPUtils.set_solver_attributes!(opt_model; solver_attributes...)

    x = @variable(opt_model, [1:n_states, 1:T])
    u = @variable(opt_model, [1:n_controls, 1:T])
    @constraint(opt_model, linear_dynamics_constraints(x, u; A, B) .== 0)
    @constraint(opt_model, x[:, 1] .== x0)
    @objective(opt_model, Min, forward_quadratic_objective(x, u; Q, R))
    @time JuMP.optimize!(opt_model)
    JuMPUtils.isconverged(opt_model), JuMPUtils.get_values(; x, u), opt_model
end

#=========================================== Non-LQ-Case ===========================================#

"Solves a forward optimal control problem with protentially nonlinear dynamics and nonquadratic
costs using JuMP."
function solve_optimal_control(
    control_system,
    cost_model,
    x0,
    T;
    fixed_inputs = (),
    init = (),
    solver = Ipopt.Optimizer,
    solver_attributes = (; print_level = 3),
    verbose = false,
)
    @unpack n_states, n_controls = control_system

    opt_model = JuMP.Model(solver)
    JuMPUtils.set_solver_attributes!(opt_model; solver_attributes...)

    # decision variables
    x = @variable(opt_model, [1:n_states, 1:T])
    u = @variable(opt_model, [1:n_controls, 1:T])

    # initial guess
    JuMPUtils.init_if_hasproperty!(x, init, :x)
    JuMPUtils.init_if_hasproperty!(u, init, :u)

    # fix certain inputs
    for i in fixed_inputs
        JuMP.fix.(u[i, :], init.u[i, :])
    end

    # Add hyperplane constraints
    # TODO: 
    #   - Make a more general function of the form p_for_pn(t, n) where n is the player number 
    #   - make this work for more players (do we need this tho?)
    #   - Put this somewhere else? Not sure this is the best place to do it. 
    #   - Prettier way of calculating norm when calculating n0
    #   - Make sure indexing is right. Should we use p(t) or p(t+1)? 

    # Parameters
    index_offset = 4 # Depends on state space, can get tide of this if I put it in cost definition tho
    rho = 0.25 # KoZ radius
    ω = 0.05 # Angular velocity of hyperplane

    # Calculate n0 (vector point from player 2 to player 1), and find its angle wrt to x-axis
    n0_full = x0[1:2] - x0[(1 + index_offset):(2 + index_offset)]
    α = atan(n0_full[2],n0_full[1])

    # Define useful vectors
    # Note indexing using (t-1) 
    function n_for_p1(t) 
        [cos(α + ω * (t-1)), sin(α + ω * (t-1))]
    end
    function n_for_p2(t)
        -[cos(α + ω * t), sin(α + ω * (t-1))]
    end

    # Only valid from 1:T
    function p_for_p1(t)
        x_other = x[(1 + index_offset):(2 + index_offset), t]
        x_other + rho .* n_for_p1(t)
    end
    function p_for_p2(t)
        x_other = x[1:2, t]
        x_other + rho .* n_for_p2(t)
    end

    # Define constraints
    @constraint(opt_model, [t = 1:T], n_for_p1(t)' * (x[1:2, t] - p_for_p1(t)) >= 0) # player 1
    # Main.@infiltrate
    # @constraint(
    #     opt_model,
    #     [t = 1:T],
    #     n_for_p2(t)' * (x[(1 + index_offset):(2 + index_offset), t] - p_for_p2(t)) >= 0
    # ) # player 2

    # Dynamics constraints
    DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)
    if !isnothing(x0)
        @constraint(opt_model, x[:, 1] .== x0)
    end

    # Cost function
    cost_model.add_objective!(opt_model, x, u; cost_model.weights)

    # Solve
    time = @elapsed JuMP.optimize!(opt_model)
    verbose && @info time
    solution = merge(JuMPUtils.get_values(; x, u), (; runtime = JuMP.solve_time(opt_model)))
    JuMPUtils.isconverged(opt_model), solution, opt_model
end

end
