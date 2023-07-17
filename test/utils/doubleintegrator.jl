struct DoubleIntegrator{T<:Real}
    ΔT::T
end

function Base.getproperty(system::DoubleIntegrator, sym::Symbol)
    if sym === :n_states
        4
    elseif sym === :n_controls
        2
    else
        getfield(system, sym)
    end
end

function DynamicsModelInterface.next_x(system::DoubleIntegrator, x_t, u_t)
    ΔT = system.ΔT
    @assert only(size(x_t)) == 4
    @assert only(size(u_t)) == 2
    px, py, vx, vy = x_t
    Δvx, Δvy = u_t
    [px + ΔT * vx, py + ΔT * vy, vx + ΔT * Δvx, vy + ΔT * Δvy]
end

# These constraints encode the dynamics of a DoubleIntegrator with state layout x_t = [px, py, vx, vy] and
# inputs u_t = [Δvx, Δvy].
function DynamicsModelInterface.add_dynamics_constraints!(system::DoubleIntegrator, opt_model, x, u)
    ΔT = system.ΔT
    T = size(x, 2)

    @constraint(
        opt_model,
        [t = 1:(T - 1)],
        x[:, t + 1] .== [
            x[1, t] + ΔT * x[3, t],
            x[2, t] + ΔT * x[4, t],
            x[3, t] + ΔT * u[1, t],
            x[4, t] + ΔT * u[2, t],
        ]
    )
end

function DynamicsModelInterface.add_dynamics_jacobians!(system::DoubleIntegrator, opt_model, x, u)
    ΔT = system.ΔT
    n_states, T = size(x)
    n_controls = size(u, 1)

    # jacobians of the dynamics in x
    dfdx = @variable(opt_model, [1:n_states, 1:n_states, 1:T])
    @constraint(
        opt_model,
        [t = 1:T],
        dfdx[:, :, t] .== [
            1 0 ΔT 0
            0 1 0  ΔT
            0 0 1  0                 
            0 0 0  1                 
        ]
    )

    # jacobians of the dynamics in u
    dfdu = [
        0  0
        0  0
        ΔT 0
        0 ΔT
    ] .* reshape(ones(T), 1, 1, :)

    (; dx = dfdx, du = dfdu)
end

function DynamicsModelInterface.add_shared_constraint!(system::DoubleIntegrator, opt_model, x, u, parameters; set = true)
    # Note: this is getting fed the FULL state vector, not just the player 1 state vector

    # Known parameters
    T = size(x, 2)
    T_offset = parameters.T_offset
    couple = parameters.couple
    ω = parameters.ω
    α = parameters.α
    ρ = parameters.ρ

    # Player indices 
    idx_ego   = (1:2) .+ (couple[1] - 1)*system.n_states
    idx_other = (1:2) .+ (couple[2] - 1)*system.n_states

    # Hyperplane normal 
    # Note indexing using (t-1) 
    n_cos = @variable(opt_model, [1:T])
    n_sin = @variable(opt_model, [1:T])
    @NLconstraint(opt_model, [t = 1:T], n_sin[t] == sin(α + ω * (t - 1 + T_offset)))
    @NLconstraint(opt_model, [t = 1:T], n_cos[t] == cos(α + ω * (t - 1 + T_offset)))
    function n(t)
        [n_cos[t],n_sin[t]]
    end

    # Intersection of hyperplane w/ KoZ
    p = @variable(opt_model, [1:2, 1:T])
    @NLconstraint(opt_model, [t = 1:T], p[1,t] == x[idx_other[1], t] + ρ*n_cos[t])
    @NLconstraint(opt_model, [t = 1:T], p[2,t] == x[idx_other[2], t] + ρ*n_sin[t])

    # Define constraint
    function h(t)
        n(t)' * (x[idx_ego, t] - p[:,t])
    end

    # Set constraints
    if set
        @constraint(opt_model, [t = 1:T], h(t) >= 0) # player 1
    end
    
    # Return constraint function and time indices where it applies
    return h
end 

function DynamicsModelInterface.add_shared_jacobian!(system::DoubleIntegrator, opt_model, x, u, parameters)
    # Note: this is getting fed the FULL state vector, not just the player 1 state vector

    # Known parameters
    n_states_all, T = size(x)
    T_offset = parameters.T_offset
    n_states = system.n_states
    n_players = Int(n_states_all/n_states)
    couple = parameters.couple
    ω = parameters.ω
    α = parameters.α
    ρ = parameters.ρ

    # Player indices 
    idx_ego   = (1:2) .+ (couple[1] - 1)*system.n_states
    idx_owner = (1:2) .+ (couple[2] - 1)*system.n_states
    idx_other = setdiff(1:n_states_all, vcat(idx_ego, idx_owner))

    # Gradients of hyperplane constraints with respect to x
    n_cos = @variable(opt_model, [2:T])
    n_sin = @variable(opt_model, [2:T])
    @NLconstraint(opt_model, [t = 2:T], n_cos[t] == cos(α + ω * (t - 1 + T_offset)))
    @NLconstraint(opt_model, [t = 2:T], n_sin[t] == sin(α + ω * (t - 1 + T_offset)))

    dhdx = @variable(opt_model, [[1], 1:n_states_all, 1:T])

    # Elements corresponding to ego 
    @constraint(
        opt_model,
        [t = 2:T],
        dhdx[[1], idx_ego, t] .== 
        [
             n_cos[t] 
             n_sin[t]
        ]'
    )
    # Elements corresponding to hyperplane owner 
    @constraint(
        opt_model,
        [t = 2:T],
        dhdx[[1], idx_owner, t] .== 
        [
            -n_cos[t] 
            -n_sin[t]
        ]'
    )
    # Rest of elements 
    @constraint(
        opt_model,
        [t = 2:T],
        dhdx[[1], idx_other, t] .== 
        zeros((n_states - 2)*2 + n_states*(n_players - 2))'
    )

    (; dx = dhdx)
end

#========================================== Visualization ==========================================#

# function TrajectoryVisualization.trajectory_data(::DoubleIntegrator, x, player = 1)
#     [
#         (; px = xi[1], py = xi[2], v = xi[3], θ = xi[4], player, t) for
#         (t, xi) in enumerate(eachcol(x))
#     ]
# end
