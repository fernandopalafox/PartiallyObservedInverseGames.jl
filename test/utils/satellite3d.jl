struct Satellite3D{T<:Real} 
    ΔT::T
    n::T
    m::T
    A::Matrix{T}
    B::Matrix{T}
    u_max::T
end

function Satellite3D(ΔT, n, m, u_max)
    A = [ 4-3*cos(n*ΔT)      0  0           1/n*sin(n*ΔT)     2/n*(1-cos(n*ΔT))        0;
          6*(sin(n*ΔT)-n*ΔT) 1  0          -2/n*(1-cos(n*ΔT)) 1/n*(4*sin(n*ΔT)-3*n*ΔT) 0;  
          0                  0  cos(n*ΔT)   0                 0                        1/n*sin(n*ΔT);
          3*n*sin(n*ΔT)      0  0           cos(n*ΔT)         2*sin(n*ΔT)              0;
         -6*n*(1-cos(n*ΔT))  0  0          -2*sin(n*ΔT)       4*cos(n*ΔT)-3            0;
          0                  0 -sin(n*ΔT)   0                 0                        cos(n*ΔT)];
    B = 1/m*[ 1/n^2(1-cos(n*ΔT))     2/n^2*(n*ΔT-sin(n*ΔT))       0;
             -2/n^2*(n*ΔT-sin(n*ΔT)) 4/n^2*(1-cos(n*ΔT))-3/2*ΔT^2 0;
              0                      0                            1/n^2(1-cos(n*ΔT));
              1/n*sin(n*ΔT)          2/n*(1-cos(n*ΔT))            0;
             -2/n*(1-cos(n*ΔT))      4/n*sin(n*ΔT)-3*ΔT           0;
              0                      0                            1/n*sin(n*ΔT)]
    # A = [ 4-3*cos(n*ΔT)          0  0   1/n*sin(n*ΔT)     2/n*(1-cos(n*ΔT))        0;
    #           6*(sin(n*ΔT)-n*ΔT) 1  0  -2/n*(1-cos(n*ΔT)) 1/n*(4*sin(n*ΔT)-3*n*ΔT) 0;  
    #           0                  0  0   0                 0                        0;
    #           3*n*sin(n*ΔT)      0  0           cos(n*ΔT)         2*sin(n*ΔT)      0;
    #          -6*n*(1-cos(n*ΔT))  0  0          -2*sin(n*ΔT)       4*cos(n*ΔT)-3    0;
    #           0                  0  0   0                 0                        0];
    # B = 1/m*[     1/n^2(1-cos(n*ΔT))         2/n^2*(n*ΔT-sin(n*ΔT))       0;
    #              -2/n^2*(n*ΔT-sin(n*ΔT)) 4/n^2*(1-cos(n*ΔT))-3/2*ΔT^2 0;
    #               0                      0                            0;
    #               1/n*sin(n*ΔT)          2/n*(1-cos(n*ΔT))            0;
    #              -2/n*(1-cos(n*ΔT))      4/n*sin(n*ΔT)-3*ΔT           0;
    #               0                      0                            0]
    return Satellite3D(ΔT, n, m, A, B, u_max)
end  

function Base.getproperty(system::Satellite3D, sym::Symbol)
    if sym === :n_states
        6
    elseif sym === :n_controls
        3
    else
        getfield(system, sym)
    end
end

function DynamicsModelInterface.next_x(system::Satellite3D, x_t, u_t)
    A = system.A
    B = system.B
    @assert only(size(x_t)) == 6
    @assert only(size(u_t)) == 3

    return A * x_t + B * u_t
end

# These constraints encode the dynamics of a 2d with state layout x_t = [px, py, vx, vy] and
# inputs u_t = [Δvx, Δvy].
function DynamicsModelInterface.add_dynamics_constraints!(system::Satellite3D, opt_model, x, u)
    A = system.A
    B = system.B
    T = size(x, 2)

    @constraint(
        opt_model,
        [t = 1:(T - 1)],
        x[:, t + 1] .== A * x[:, t] + B * u[:, t]
    )
end

function DynamicsModelInterface.add_dynamics_jacobians!(system::Satellite3D, opt_model, x, u)
    A = system.A
    B = system.B
    T = size(x, 2)

    # jacobians of the dynamics in x
    dfdx = A .* reshape(ones(T), 1, 1, :)

    # jacobians of the dynamics in u
    dfdu = B .* reshape(ones(T), 1, 1, :)

    (; dx = dfdx, du = dfdu)
end

function DynamicsModelInterface.add_shared_constraint!(system::Satellite3D, opt_model, x, u, parameters; set = true)
    # Note: this is getting fed the FULL state vector, not just the player 1 state vector

    # Known parameters
    ΔT = system.ΔT
    T = size(x, 2)
    T_offset = parameters.T_offset
    couple = parameters.couple
    ω = parameters.ω
    α = parameters.α
    ρ = parameters.ρ
    θ = parameters.θ

    # Player indices 
    idx_ego   = (1:2) .+ (couple[1] - 1)*system.n_states
    idx_other = (1:2) .+ (couple[2] - 1)*system.n_states

    # Hyperplane normal 
    # Note indexing using (t-1) 
    n_cos = @variable(opt_model, [1:T])
    n_sin = @variable(opt_model, [1:T])
    @NLconstraint(opt_model, [t = 1:T], n_sin[t] == sin(θ + α + ω * (t - 1 + T_offset) * ΔT))
    @NLconstraint(opt_model, [t = 1:T], n_cos[t] == cos(θ + α + ω * (t - 1 + T_offset) * ΔT))
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

function DynamicsModelInterface.add_shared_jacobian!(system::Satellite3D, opt_model, x, u, parameters)
    # Note: this is getting fed the FULL state vector, not just the player 1 state vector

    # Known parameters
    ΔT = system.ΔT
    n_states_all, T = size(x)
    T_offset = parameters.T_offset
    n_states = system.n_states
    n_players = Int(n_states_all/n_states)
    couple = parameters.couple
    ω = parameters.ω
    α = parameters.α
    ρ = parameters.ρ
    θ = parameters.θ

    # Player indices 
    idx_ego   = (1:2) .+ (couple[1] - 1)*system.n_states
    idx_owner = (1:2) .+ (couple[2] - 1)*system.n_states
    idx_other = setdiff(1:n_states_all, vcat(idx_ego, idx_owner))

    # Gradients of hyperplane constraints with respect to x
    n_cos = @variable(opt_model, [2:T])
    n_sin = @variable(opt_model, [2:T])
    @NLconstraint(opt_model, [t = 2:T], n_cos[t] == cos(θ + α + ω * (t - 1 + T_offset) * ΔT))
    @NLconstraint(opt_model, [t = 2:T], n_sin[t] == sin(θ + α + ω * (t - 1 + T_offset) * ΔT))
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
