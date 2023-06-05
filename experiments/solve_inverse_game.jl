using CSV, DataFrames
using Ipopt: Ipopt
using JuMP: JuMP, @variable, @constraint, @objective
using UnPack: @unpack

# This script solves for the rotating hyperplane obstacle avoidance scenario 
# Assumptions
# - The number of players is 2
# - Data is formatted as a CSV file where 
#   - Every column corresponds to a time step
#   - Every row corresponds to a state variable
#   - States for player 1 are listed first, appended by states for player 2
#   - We know the players' objectives

# Parameters
player_angles = [0, pi]
ΔT    = 0.25
μ = 0.1
n_players = 2

@unpack n_states, n_controls = control_system
# Compute initial positions
x0 = mapreduce(vcat, player_angles) do player_angle
    [unitvector(player_angle + pi); 0.1; player_angle + deg2rad(10)]
end

# Load data 
data = Matrix(CSV.read("data/hyperplane_trajectory.csv", DataFrame, header = false))
T = size(data,2)

# System dynamics
control_system =
    TestDynamics.ProductSystem([TestDynamics.Unicycle(0.25), TestDynamics.Unicycle(0.25)])

# Cost models 
# CLEAN OUT OBJECTIVE GRADIENTS. ONLY NEED GRADIENTS FROM GOAL COST AND CONTROL COST
# PUT THIS INTO CollisionAvoidanceGame.jl 
cost_model = map(enumerate(player_angles)) do (ii, player_angle)
    cost_model_p1 = CollisionAvoidanceGame.generate_player_cost_model_simple(;
        player_idx = ii,
        control_system,
        T,
        goal_position = unitvector(player_angle),
    )
end

# Decision variables 
x_1 = @variable(opt_model, [1:n_states, 1:T])
x_2 = @variable(opt_model, [1:n_states, 1:T])
u = @variable(opt_model, [1:n_controls, 1:T])
u = @variable(opt_model, [1:n_controls, 1:T])
λ = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])

# For player 1 

# Define dynamics and their Jacobians and add them 
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

    DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x_1, u_1)
    df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)
    dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)

    ### CONTINUE HERE ###
    # FIGURE OUT ADD OBJECTIVE GRADIENTS 
    # FIGURE OUT HOW TO ADD EXTRA SLACK VARIABLES 

# KKT Nash constraints

@constraint(opt_model,
    [t = 2:(T - 1)],
        dJ.dx[:, t] + λ[:, t - 1, player_idx] - (λ[:, t, player_idx]' * df.dx[:, :, t])' .== 0
)

# Vanishing Lagrangian for first timestep 
@constraint(
    opt_model,
    dJ.dx[:, 1] - (λ[:, 1, player_idx]' * df.dx[:, :, 1])' + λ0[:, player_idx] .== 0
)
# Vanishing Lagrangian for all the other steps 
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
@constraint(opt_model, dJ.du[player_inputs, T] .== 0) # This is the vanishing gradient of the Lagrangian with respect to u but since there are no constraint on the input it ends up looking just like a simple gradient set to zero. 









# ----------------- OLD CODE ----------------- #



# # Define optimization model 
# opt_model = JuMP.Model(Ipopt.Optimizer)
# @variable(opt_model, x_1[1:n_states,1:T])
# @variable(opt_model, x_2[1:n_states,1:T])
# @variable(opt_model, λ_e_1[1:T])
# @variable(opt_model, λ_e_2[1:T])
# @variable(opt_model, λ_i_1[1:T])
# @variable(opt_model, λ_i_2[1:T])
# @variable(opt_model, s_1[1:T])
# @variable(opt_model, s_2[1:T])

# # Gradient terms
#     # Objectives CHECK THIS
#     #∇ₓf_1 = 2*sum(el -> el^2, x_1[:,T] - x_g_1) 
#     #∇ₓf_2 = 2*sum(el -> el^2, x_2[:,T] - x_g_2) 

#     # Inequality constraints
#     ∇ₓh_1 = s_1.*x_1

# # Define KKT conditions for player 1
# @constraint(opt_model, x1 == data[1,1])

# # Turn into log-barrier problem

# # Solve optimization problem



# # Define objectives and constraints for each player 
# # Everything should be in terms of the decision variables of a single player 
# function objective(x,x_g)
#     return sum(el -> el^2, x[:,T] - x_g)
# end

# function equality_constraints(x,u)
#     reduce(vcat,
#         [x[:, t + 1] .- [
#             x[1, t] + ΔT * x[3, t] * cos(x[4,t]),
#             x[2, t] + ΔT * x[3, t] * sin(x[4,t]),
#             x[3, t] + u[1, t],
#             x[4, t] + u[2, t],
#         ] for t in 1:(T - 1)]
#     )
# end

# function dynamics_gradient_manual(x)
#     dfdx[:, :, t] == [
#             1 0 ΔT*cos(x[4,t]) -ΔT*x[3, t]*sin(x[4,t])
#             0 1 ΔT*sin(x[4,t]) +ΔT*x[3, t]*cos(x[4,t])
#             0 0 1 0
#             0 0 0 1
#         ]
# end

# function inequality_constraint(x_1,x_2)
#     # Normal to hyperplane
#     # Starts pointing from player 2 to player 1 and rotates with angular velocity ω
#     n0 = x_1[1:2,1] - x_2[1:2,1]
#     α = atan(n0[2],n0[1])
#     function n(t) 
#         [cos(α + ω * t), sin(α + ω * t)]
#     end

#     # Point on surface of KoZ where hyperplane is centered
#     function p(t)
#         x_2[:, t] + rho .* n(t)
#     end

#     # Inequality constraint vector 
#     return [dot(n(t), x_1[1:2, t] - p(t)) for t in 1:T]
# end
