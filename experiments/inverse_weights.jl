# Imports
using Revise

const project_root_dir = realpath(joinpath(@__DIR__, ".."))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "test/utils"))))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "experiments/utils"))))

import Ipopt
import TestDynamics

using CollisionAvoidanceGame: CollisionAvoidanceGame
using CSV, DataFrames
using Ipopt: Ipopt
using JuMP: JuMP, @variable, @constraint, @NLconstraint, @objective, set_start_value
using UnPack: @unpack
using PartiallyObservedInverseGames.TrajectoryVisualization: visualize_rotating_hyperplanes
using PartiallyObservedInverseGames.CostUtils
using PartiallyObservedInverseGames.JuMPUtils

include("utils/misc.jl")

solution = let 

    ΔT = 0.1
    n_players = 4
    scale = 1
    t_real = 5.0
    T_activate_goalcost = 1
    
    cmin = 1e-3
    solver_attributes = (; max_wall_time = 30.0, print_level = 5)

    # Load observation data
    data_states = Matrix(CSV.read("data/f_di_s.csv", DataFrame, header = false))
    data_inputs = Matrix(CSV.read("data/f_di_c.csv", DataFrame, header = false))
    init = (; x = data_states, u = data_inputs)
    init = nothing
    
    # Setup system
    control_system = TestDynamics.ProductSystem([TestDynamics.DoubleIntegrator(ΔT) for _ in 1:n_players])
    as = [2*pi/n_players * (i-1) for i in 1:n_players] # angles
    as = [a > pi ? a - 2*pi : a for a in as]
    T = Int(t_real / ΔT)
    
    # cost models with dummy weights
    player_cost_models = map(enumerate(as)) do (ii, a)
        cost_model_p1 = CollisionAvoidanceGame.generate_integrator_cost(;
            player_idx = ii,
            control_system,
            T,
            goal_position = scale*unitvector(a),
            weights = (; 
                state_proximity = -1, 
                state_goal      = -1,
                control_Δv      = -1),
            T_activate_goalcost,
            prox_min_regularization = 0.1
        )
    end

    # Setup solver 
    @unpack n_states, n_controls = control_system
    opt_model = JuMP.Model(Ipopt.Optimizer)
    JuMPUtils.set_solver_attributes!(opt_model; solver_attributes...)

    # Decision Variables
    player_weights =
        [@variable(opt_model, [keys(cost_model.weights)]) for cost_model in player_cost_models]
    x = @variable(opt_model, [1:n_states, 1:T])
    u = @variable(opt_model, [1:n_controls, 1:T])
    λ = @variable(opt_model, [1:n_states, 1:(T - 1), 1:n_players])
    # Initialization
    JuMPUtils.init_if_hasproperty!(λ, init, :λ)
    JuMPUtils.init_if_hasproperty!(x, init, :x)
    JuMPUtils.init_if_hasproperty!(u, init, :u)
    # constraints
    DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)
    df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)
    for (player_idx, cost_model) in enumerate(player_cost_models)
        weights = player_weights[player_idx]
        @unpack player_inputs = cost_model
        dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)
        # KKT Nash constraints
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
    end
    # regularization
    for weights in player_weights
        @constraint(opt_model, weights .>= cmin)
        @constraint(opt_model, sum(weights) .== 1)
        
    end
    # objective
    @objective(
        opt_model,
        Min,
        sum(el -> el^2, x .- data_states)
    )
    time = @elapsed JuMP.optimize!(opt_model)
    @info time
    merge(get_values(; x, u, λ), (; player_weights = map(w -> CostUtils.namedtuple(JuMP.value.(w)), player_weights)))
end