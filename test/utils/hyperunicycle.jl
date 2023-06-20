struct HyperUnicycle{T<:Real}
    ΔT::T
    ω::T
    ρ::T
end

function Base.getproperty(system::HyperUnicycle, sym::Symbol)
    if sym === :n_states
        4
    elseif sym === :n_controls
        2
    else
        getfield(system, sym)
    end
end

function DynamicsModelInterface.next_x(system::HyperUnicycle, x_t, u_t)
    ΔT = system.ΔT
    @assert only(size(x_t)) == 4
    @assert only(size(u_t)) == 2
    px, py, v, θ = x_t
    Δv, Δθ = u_t
    [px + ΔT * v * cos(θ), py + ΔT * v * sin(θ), v + Δv, θ + Δθ]
end

# These constraints encode the dynamics of a HyperUnicycle with state layout x_t = [px, pyL, v, θ] and
# inputs u_t = [Δv, Δθ].
function DynamicsModelInterface.add_dynamics_constraints!(system::HyperUnicycle, opt_model, x, u)
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
end

function DynamicsModelInterface.add_dynamics_jacobians!(system::HyperUnicycle, opt_model, x, u)
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
            0 1 ΔT*sinθ[t]  ΔT*x[3, t]*cosθ[t]
            0 0 1           0                 
            0 0 0           1                 
        ]
    )

    # jacobians of the dynamics in u
    # Last one is not used
    dfdu = [
        0 0
        0 0
        1 0
        0 1
    ] .* reshape(ones(T), 1, 1, :)

    (; dx = dfdx, du = dfdu)
end

function DynamicsModelInterface.add_inequality_constraints!(system::HyperUnicycle, opt_model, x, u, params; set = true)
    # Note: this is getting fed the FULL state vector, not just the player 1 state vector

    # Known parameters
    index_offset = system.n_states
    T = size(x, 2)
    ρ = system.ρ
    α = params.α
    
    # Unknown parameters
    ω = params.ω

    # Hyperplane normal 
    # Note indexing using (t-1) 
    n_cos = @variable(opt_model, [1:T])
    n_sin = @variable(opt_model, [1:T])
    @NLconstraint(opt_model, [t = 1:T], n_cos[t] == cos(α + ω * (t-1)))
    @NLconstraint(opt_model, [t = 1:T], n_sin[t] == sin(α + ω * (t-1)))
    function n(t)
        [n_cos[t],n_sin[t]]
    end

    # Intersection of hyperplane w/ KoZ
    function p(t)
        x_other = x[(1 + index_offset):(2 + index_offset), t]
        x_other + ρ .* n(t)
    end

    # Define constraint
    function h(t)
        n(t)' * (x[1:2, t] - p(t))
    end

    # Set constraints
    if set
        @constraint(opt_model, [t = 1:T], h(t) >= 0) # player 1
    end
    
    # Return constraint function
    return h
end 

function DynamicsModelInterface.add_inequality_jacobians!(system::HyperUnicycle, opt_model, x, u, params)
    # Note: this is getting fed the FULL state vector, not just the player 1 state vector

    # Known parameters
    n_states_all, T = size(x)
    n_states = system.n_states
    α = params.α

    # Unknown parameters
    ω  = params.ω

    # Gradients of hyperplane constraints with respect to x
    n_cos = @variable(opt_model, [2:T])
    n_sin = @variable(opt_model, [2:T])
    @NLconstraint(opt_model, [t = 2:T], n_cos[t] == cos(α + ω * (t-1)))
    @NLconstraint(opt_model, [t = 2:T], n_sin[t] == sin(α + ω * (t-1)))

    dhdx = @variable(opt_model, [1:T, 1:n_states_all])
    @constraint(
        opt_model,
        [t = 2:T],
        dhdx[t, :] .== 
        [
             n_cos[t]
             n_sin[t]
            zeros(n_states - 2)
            -n_cos[t]
            -n_sin[t]
            zeros(n_states - 2)
        ]
    )

    (; dx = dhdx)
end

#========================================== Visualization ==========================================#

function TrajectoryVisualization.trajectory_data(::HyperUnicycle, x, player = 1)
    [
        (; px = xi[1], py = xi[2], v = xi[3], θ = xi[4], player, t) for
        (t, xi) in enumerate(eachcol(x))
    ]
end
