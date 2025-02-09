struct Unicycle{T<:Real}
    ΔT::T
end

function Base.getproperty(system::Unicycle, sym::Symbol)
    if sym === :n_states
        4
    elseif sym === :n_controls
        2
    else
        getfield(system, sym)
    end
end

function DynamicsModelInterface.next_x(system::Unicycle, x_t, u_t)
    ΔT = system.ΔT
    @assert only(size(x_t)) == 4
    @assert only(size(u_t)) == 2
    px, py, v, θ = x_t
    Δv, Δθ = u_t
    [px + ΔT * v * cos(θ), py + ΔT * v * sin(θ), v + Δv, θ + Δθ]
end

# These constraints encode the dynamics of a unicycle with state layout x_t = [px, pyL, v, θ] and
# inputs u_t = [Δv, Δθ].
function DynamicsModelInterface.add_dynamics_constraints!(system::Unicycle, opt_model, x, u)
    ΔT = system.ΔT
    T = size(x, 2)

    # auxiliary variables for nonlinearities
    cosθ = @variable(opt_model, [1:T])
    @NLconstraint(opt_model, [t = 1:T], cosθ[t] == cos(x[4, t]))
    sinθ = @variable(opt_model, [1:T])
    @NLconstraint(opt_model, [t = 1:T], sinθ[t] == sin(x[4, t]))

    @constraint(
        opt_model,
        [t = 1:(T - 1)],
        x[:, t + 1] .== [
            x[1, t] + ΔT * x[3, t] * cosθ[t],
            x[2, t] + ΔT * x[3, t] * sinθ[t],
            x[3, t] + u[1, t],
            x[4, t] + u[2, t],
        ]
    )

    # @constraint(opt_model, x[:, T] == xg)
end

function DynamicsModelInterface.add_dynamics_jacobians!(system::Unicycle, opt_model, x, u)
    ΔT = system.ΔT
    n_states, T = size(x)
    n_controls = size(u, 1)

    # auxiliary variables for nonlinearities
    cosθ = @variable(opt_model, [1:T])
    @NLconstraint(opt_model, [t = 1:T], cosθ[t] == cos(x[4, t]))
    sinθ = @variable(opt_model, [1:T])
    @NLconstraint(opt_model, [t = 1:T], sinθ[t] == sin(x[4, t]))

    # jacobians of the dynamics in x
    dfdx = @variable(opt_model, [1:n_states, 1:n_states, 1:T])
    @constraint(
        opt_model,
        [t = 1:T],
        dfdx[:, :, t] .== [
            1 0 ΔT*cosθ[t] -ΔT*x[3, t]*sinθ[t]
            0 1 ΔT*sinθ[t] +ΔT*x[3, t]*cosθ[t]
            0 0 1 0
            0 0 0 1
        ]
    )

    # jacobians of the dynamics in u
    dfdu = [
        0 0
        0 0
        1 0
        0 1
    ] .* reshape(ones(T), 1, 1, :)

    (; dx = dfdx, du = dfdu)
end

#========================================== Visualization ==========================================#

function TrajectoryVisualization.trajectory_data(::Unicycle, x, player = 1)
    [
        (; px = xi[1], py = xi[2], v = xi[3], θ = xi[4], player, t) for
        (t, xi) in enumerate(eachcol(x))
    ]
end
