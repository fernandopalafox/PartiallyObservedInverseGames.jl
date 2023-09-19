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

    
    @warn "remove hardocoded stuff (and hyperplane from here)"
    n_players = 3
    ωs = [0.05, 0.05, 0.05]
    ρs = [0.25, 0.25,  0.25]
    αs = [3/4*pi,  pi, 5/4*pi]
    # n_players = 2
    # ωs = [0.03, 0.03]
    # ρs = [0.25, 0.25]
    n_states_per_player = control_system.subsystems[1].n_states

    # Add hyperplane constraints (centered around player 1)
    for i in 1:(n_players - 1)
        index_offset = n_states_per_player * i
        # Parameters
        ρ = ρs[i] # KoZ radius
        ω = ωs[i] # Angular velocity of hyperplane

        idx_ego = [1, 2]
        idx_other = [1 + index_offset, 2 + index_offset]

        # Calculate hyperplane normal 
        α = αs[i]

        # Define useful vectors
        function n(t) 
            [cos(α + ω * (t-1)), sin(α + ω * (t-1))]
        end
        function p(t)
            x_other = x[idx_other, t]
            x_other + ρ .* n(t)
        end

        # Add constraint
        @constraint(opt_model, [t = 1:T], n(t)' * (x[idx_ego, t] - p(t)) >= 0) 
    end

    # # Add hyperplane constraint (centered around player 2)
    # for i in 3:3
    #     index_offset = n_states_per_player * i
    #     # Parameters
    #     ρ = ρs[i] # KoZ radius
    #     ω = ωs[i] # Angular velocity of hyperplane

    #     # Ego indices
    #     idx_ego = [5, 6]
    #     idx_other = [9, 10]
        
    #     # Calculate hyperplane normal 
    #     α = αs[i]

    #     # Define useful vectors
    #     function n(t) 
    #         [cos(α + ω * (t-1)), sin(α + ω * (t-1))]
    #     end
    #     function p(t)
    #         x_other = x[idx_other, t]
    #         x_other + ρ .* n(t)
    #     end

    #     # Add constraint
    #     @constraint(opt_model, [t = 1:T], n(t)' * (x[idx_ego, t] - p(t)) >= 0) 
    # end

    # Dynamics constraints
    DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)

    # Initial condition constraint
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
