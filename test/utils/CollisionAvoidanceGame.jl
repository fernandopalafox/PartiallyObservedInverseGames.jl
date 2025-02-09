module CollisionAvoidanceGame

using JuMP: @NLconstraint, @NLexpression, @objective, @variable
using PartiallyObservedInverseGames.CostUtils: symbol

unique!(push!(LOAD_PATH, @__DIR__))
import TestDynamics

export generate_player_cost_model

function generate_player_cost_model(;
    player_idx,
    control_system::TestDynamics.ProductSystem,
    T,
    goal_position,
    weights = (; state_proximity = 1, state_velocity = 1, control_Δv = 1, control_Δθ = 1),
    cost_prescaling = (;
        state_proximity = 0.1,
        state_velocity = 1,
        control_Δv = 10,
        control_Δθ = 1,
    ),
    fix_costs = (; # encoding soft-constraints rather than preferences
        state_goal = 100,
        state_lane = 1.0,
        state_orientation = 2.0,
    ),
    x_lane_center = nothing,
    y_lane_center = nothing,
    target_speed = 0,
    target_orientation = if !isnothing(x_lane_center)
        0
    elseif !isnothing(y_lane_center)
        pi / 2
    else
        nothing
    end,
    prox_min_regularization = 0.1,
    T_activate_goalcost = T,
)
    state_indices = TestDynamics.state_indices(control_system, player_idx)
    input_indices = TestDynamics.input_indices(control_system, player_idx)

    opponent_position_indices = let
        opponent_indices = filter(!=(player_idx), eachindex(control_system.subsystems))
        map(opponent_indices) do jj
            TestDynamics.state_indices(control_system, jj)[1:2]
        end
    end

    function add_regularized_squared_distance!(opt_model, x_sub_ego, pos_opponent)
        @NLexpression(
            opt_model,
            [t = 1:T],
            (x_sub_ego[1, t] - pos_opponent[1, t])^2 +
            (x_sub_ego[2, t] - pos_opponent[2, t])^2 +
            prox_min_regularization
        )
    end

    function add_objective!(opt_model, x, u; weights)
        T = size(x, 2)
        @views x_sub_ego = x[state_indices, :]
        @views u_sub_ego = u[input_indices, :]
        @views opponent_positions = map(opponent_position_indices) do opp_position_idxs
            x[opp_position_idxs, :]
        end

        J̃ = (;
            state_goal = isnothing(goal_position) ? 0 :
                         sum(el -> el^2, x_sub_ego[1:2, T_activate_goalcost:T] .- goal_position),
            state_lane = let
                # Note the switched indices. This is on purpose. The lane cost for the lane along y acts
                # on the x position and vice versa.
                x_lane_cost =
                    isnothing(x_lane_center) ? 0 :
                    sum(el -> el^2, x_sub_ego[2, :] .- x_lane_center)
                y_lane_cost =
                    isnothing(y_lane_center) ? 0 :
                    sum(el -> el^2, x_sub_ego[1, :] .- y_lane_center)
                x_lane_cost + y_lane_cost
            end,
            state_proximity = let
                prox_cost = sum(opponent_positions) do pos_opponent
                    d_sq = add_regularized_squared_distance!(opt_model, x_sub_ego, pos_opponent)
                    prox_cost = @variable(opt_model, [t = 1:T])
                    @NLconstraint(opt_model, [t = 1:T], prox_cost[t] == -log(d_sq[t]))
                    prox_cost
                end
                sum(prox_cost)
            end,
            state_velocity = sum(el -> el^2, x_sub_ego[3, :] .- target_speed),
            state_orientation = isnothing(target_orientation) ? 0 :
                                sum(el -> el^2, x_sub_ego[4, :] .- target_orientation),
            control_Δv = sum(el -> el^2, u_sub_ego[1, :]),
            control_Δθ = sum(el -> el^2, u_sub_ego[2, :]),
        )
        @objective(
            opt_model,
            Min,
            sum(weights[k] * cost_prescaling[k] * J̃[k] for k in keys(weights)) +
            sum(fix_costs[k] * J̃[k] for k in keys(fix_costs)) * sum(weights) / length(weights)
        )
    end

    function add_objective_gradients!(opt_model, x, u; weights)
        n_states, T = size(x)
        n_controls = size(u, 1)
        @views x_sub_ego = x[state_indices, :]
        @views u_sub_ego = u[input_indices, :]
        @views opponent_positions = map(opponent_position_indices) do opp_position_idxs
            x[opp_position_idxs, :]
        end

        dprox_dxy = sum(opponent_positions) do pos_opponent
            d_sq = add_regularized_squared_distance!(opt_model, x_sub_ego, pos_opponent)
            dproxdx = @variable(opt_model, [t = 1:T])
            @NLconstraint(
                opt_model,
                [t = 1:T],
                dproxdx[t] == -2 * (x_sub_ego[1, t] - pos_opponent[1, t]) / d_sq[t]
            )
            dproxdy = @variable(opt_model, [t = 1:T])
            @NLconstraint(
                opt_model,
                [t = 1:T],
                dproxdy[t] == -2 * (x_sub_ego[2, t] - pos_opponent[2, t]) / d_sq[t]
            )
            [dproxdx'; dproxdy']
        end

        dgoal_dxy =
            isnothing(goal_position) ? zeros(2, T) :
            hcat(
                zeros(2, T_activate_goalcost - 1),
                2 * (x_sub_ego[1:2, T_activate_goalcost:T] .- goal_position),
            )

        # Note: x-lane acts on y gradient and vice versa.
        dlane_dxy = let
            dlane_dx =
                isnothing(y_lane_center) ? zeros(T) : 2 * (x_sub_ego[1, :] .- y_lane_center)
            dlane_dy = isnothing(x_lane_center) ? zeros(T) : 2 * (x_sub_ego[2, :] .- x_lane_center)

            [dlane_dx'; dlane_dy']
        end

        dorientation_dθ =
            isnothing(target_orientation) ? zeros(T)' : 2 * (x_sub_ego[4, :] .- target_orientation)'

        # TODO: Technically this is missing the negative gradient on the opponents state but we
        # can't control that anyway (certainly not in OL Nash). Must be fixed for non-decoupled
        # systems and potentially FB Nash.
        dJdx = let
            dJ̃dx_sub = (;
                state_goal = [dgoal_dxy; zeros(2, T)],
                state_lane = [dlane_dxy; zeros(2, T)],
                state_proximity = [dprox_dxy; zeros(2, T)],
                state_velocity = [zeros(2, T); 2 * (x_sub_ego[3, :] .- target_speed)'; zeros(1, T)],
                state_orientation = [zeros(3, T); dorientation_dθ],
                control_Δv = zeros(size(x_sub_ego)),
                control_Δθ = zeros(size(x_sub_ego)),
            )
            dJdx_sub =
                sum(
                    weights[k] * cost_prescaling[symbol(k)] * dJ̃dx_sub[symbol(k)]
                    for k in keys(weights)
                ) +
                sum(fix_costs[k] * dJ̃dx_sub[k] for k in keys(fix_costs)) * sum(weights) /
                length(weights)
            [
                zeros(first(state_indices) - 1, T)
                dJdx_sub
                zeros(n_states - last(state_indices), T)
            ]
        end

        dJdu = let
            dJdu_sub =
                2 * [weights[:control_Δv], weights[:control_Δθ]] .*
                [cost_prescaling[:control_Δv], cost_prescaling[:control_Δθ]] .* u_sub_ego
            [
                zeros(first(input_indices) - 1, T)
                dJdu_sub
                zeros(n_controls - last(input_indices), T)
            ]
        end

        (; dx = dJdx, du = dJdu)
    end

    (; player_inputs = input_indices, weights, add_objective!, add_objective_gradients!)
end

function generate_integrator_cost(;
    player_idx,
    control_system::TestDynamics.ProductSystem,
    T,
    goal_position,
    weights = (; state_proximity = 1, state_velocity = 1, control_Δv = 1),
    cost_prescaling = (;
        state_proximity = 0.1,
        state_goal = 1,
        control_Δv = 10,
    ),
    prox_min_regularization = 0.1,
    T_activate_goalcost = T,
)
    state_indices = TestDynamics.state_indices(control_system, player_idx)
    input_indices = TestDynamics.input_indices(control_system, player_idx)

    opponent_position_indices = let
        opponent_indices = filter(!=(player_idx), eachindex(control_system.subsystems))
        map(opponent_indices) do jj
            TestDynamics.state_indices(control_system, jj)[1:2]
        end
    end

    function add_regularized_squared_distance!(opt_model, x_sub_ego, pos_opponent)
        @NLexpression(
            opt_model,
            [t = 1:T],
            (x_sub_ego[1, t] - pos_opponent[1, t])^2 +
            (x_sub_ego[2, t] - pos_opponent[2, t])^2 +
            prox_min_regularization
        )
    end

    function add_objective!(opt_model, x, u; weights)
        T = size(x, 2)
        @views x_sub_ego = x[state_indices, :]
        @views u_sub_ego = u[input_indices, :]
        @views opponent_positions = map(opponent_position_indices) do opp_position_idxs
            x[opp_position_idxs, :]
        end

        J̃ = (;
            state_goal = isnothing(goal_position) ? 0 :
                         sum(el -> el^2, x_sub_ego[1:2, T_activate_goalcost:T] .- goal_position),
            state_proximity = let
                prox_cost = sum(opponent_positions) do pos_opponent
                    d_sq = add_regularized_squared_distance!(opt_model, x_sub_ego, pos_opponent)
                    prox_cost = @variable(opt_model, [t = 1:T])
                    @NLconstraint(opt_model, [t = 1:T], prox_cost[t] == -log(d_sq[t]))
                    prox_cost
                end
                sum(prox_cost)
            end,
            control_Δv = sum(el -> el^2, u_sub_ego[1:2, :]),
        )
        @objective(
            opt_model,
            Min,
            sum(weights[k] * cost_prescaling[k] * J̃[k] for k in keys(weights))
        )
    end

    function add_objective_gradients!(opt_model, x, u; weights)
        n_states, T = size(x)
        n_controls = size(u, 1)
        @views x_sub_ego = x[state_indices, :]
        @views u_sub_ego = u[input_indices, :]
        @views opponent_positions = map(opponent_position_indices) do opp_position_idxs
            x[opp_position_idxs, :]
        end

        dprox_dxy = sum(opponent_positions) do pos_opponent
            d_sq = add_regularized_squared_distance!(opt_model, x_sub_ego, pos_opponent)
            dproxdx = @variable(opt_model, [t = 1:T])
            @NLconstraint(
                opt_model,
                [t = 1:T],
                dproxdx[t] == -2 * (x_sub_ego[1, t] - pos_opponent[1, t]) / d_sq[t]
            )
            dproxdy = @variable(opt_model, [t = 1:T])
            @NLconstraint(
                opt_model,
                [t = 1:T],
                dproxdy[t] == -2 * (x_sub_ego[2, t] - pos_opponent[2, t]) / d_sq[t]
            )
            [dproxdx'; dproxdy']
        end

        dgoal_dxy =
            isnothing(goal_position) ? zeros(2, T) :
            hcat(
                zeros(2, T_activate_goalcost - 1),
                2 * (x_sub_ego[1:2, T_activate_goalcost:T] .- goal_position),
            )

        # TODO: Technically this is missing the negative gradient on the opponents state but we
        # can't control that anyway (certainly not in OL Nash). Must be fixed for non-decoupled
        # systems and potentially FB Nash.
        dJdx = let
            dJ̃dx_sub = (;
                state_goal = [dgoal_dxy; zeros(2, T)],
                state_proximity = [dprox_dxy; zeros(2, T)],
                control_Δv = zeros(size(x_sub_ego))
            )
            dJdx_sub =
                sum(
                    weights[k] * cost_prescaling[symbol(k)] * dJ̃dx_sub[symbol(k)]
                    for k in keys(weights)
                )
            [
                zeros(first(state_indices) - 1, T)
                dJdx_sub
                zeros(n_states - last(state_indices), T)
            ]
        end

        dJdu = let
            dJdu_sub =
                2 * [weights[:control_Δv], weights[:control_Δv]] .* 
                [cost_prescaling[:control_Δv], cost_prescaling[:control_Δv]] .* 
                u_sub_ego
            [
                zeros(first(input_indices) - 1, T)
                dJdu_sub
                zeros(n_controls - last(input_indices), T)
            ]
        end

        (; dx = dJdx, du = dJdu)
    end

    (; player_inputs = input_indices, weights, add_objective!, add_objective_gradients!, goal_position)
end

function generate_hyperintegrator_cost(;
    player_idx,
    control_system::TestDynamics.ProductSystem,
    T,
    goal_position,
    weights = (; state_velocity = 1, control_Δv = 1),
    cost_prescaling = (;
        state_goal = 1,
        control_Δv = 10,
    ),
    T_activate_goalcost = T,
)
    state_indices = TestDynamics.state_indices(control_system, player_idx)
    input_indices = TestDynamics.input_indices(control_system, player_idx)

    opponent_position_indices = let
        opponent_indices = filter(!=(player_idx), eachindex(control_system.subsystems))
        map(opponent_indices) do jj
            TestDynamics.state_indices(control_system, jj)[1:2]
        end
    end

    function add_objective!(opt_model, x, u; weights)
        T = size(x, 2)
        @views x_sub_ego = x[state_indices, :]
        @views u_sub_ego = u[input_indices, :]
        @views opponent_positions = map(opponent_position_indices) do opp_position_idxs
            x[opp_position_idxs, :]
        end

        J̃ = (;
            state_goal = isnothing(goal_position) ? 0 :
                         sum(el -> el^2, x_sub_ego[1:2, T_activate_goalcost:T] .- goal_position),
            control_Δv = sum(el -> el^2, u_sub_ego[1:2, :]),
        )
        @objective(
            opt_model,
            Min,
            sum(weights[k] * cost_prescaling[k] * J̃[k] for k in keys(weights))
        )
    end

    function evaluate_objective(x, u)
        T = size(x, 2)
        @views x_sub_ego = x[state_indices, :]
        @views u_sub_ego = u[input_indices, :]
        J̃ = (;
            state_goal = isnothing(goal_position) ? 0 :
                         sum(el -> el^2, x_sub_ego[1:2, T_activate_goalcost:T] .- goal_position),
            control_Δv = sum(el -> el^2, u_sub_ego[1:2, :]),
        )

        sum(weights[k] * cost_prescaling[k] * J̃[k] for k in keys(weights))
    end

    function add_objective_gradients!(opt_model, x, u; weights)
        n_states, T = size(x)
        n_controls = size(u, 1)
        @views x_sub_ego = x[state_indices, :]
        @views u_sub_ego = u[input_indices, :]
        @views opponent_positions = map(opponent_position_indices) do opp_position_idxs
            x[opp_position_idxs, :]
        end

        dgoal_dxy =
            isnothing(goal_position) ? zeros(2, T) :
            hcat(
                zeros(2, T_activate_goalcost - 1),
                2 * (x_sub_ego[1:2, T_activate_goalcost:T] .- goal_position),
            )

        # TODO: Technically this is missing the negative gradient on the opponents state but we
        # can't control that anyway (certainly not in OL Nash). Must be fixed for non-decoupled
        # systems and potentially FB Nash.
        dJdx = let
            dJ̃dx_sub = (;
                state_goal = [dgoal_dxy; zeros(2, T)],
                control_Δv = zeros(size(x_sub_ego))
            )
            dJdx_sub =
                sum(
                    weights[k] * cost_prescaling[symbol(k)] * dJ̃dx_sub[symbol(k)]
                    for k in keys(weights)
                )
            [
                zeros(first(state_indices) - 1, T)
                dJdx_sub
                zeros(n_states - last(state_indices), T)
            ]
        end

        dJdu = let
            dJdu_sub =
                2 * [weights[:control_Δv], weights[:control_Δv]] .* 
                [cost_prescaling[:control_Δv], cost_prescaling[:control_Δv]] .* 
                u_sub_ego
            [
                zeros(first(input_indices) - 1, T)
                dJdu_sub
                zeros(n_controls - last(input_indices), T)
            ]
        end

        (; dx = dJdx, du = dJdu)
    end

    (; player_inputs = input_indices, weights, add_objective!, add_objective_gradients!, goal_position, evaluate_objective)
end

function generate_player_cost_model_simple(;
    player_idx,
    control_system::TestDynamics.ProductSystem,
    T,
    goal_position,
    weights = (; control_Δv = 1, control_Δθ = 1, state_goal = 1),
    T_activate_goalcost = T,
)
    state_indices = TestDynamics.state_indices(control_system, player_idx)
    input_indices = TestDynamics.input_indices(control_system, player_idx)

    opponent_position_indices = let
        opponent_indices = filter(!=(player_idx), eachindex(control_system.subsystems))
        map(opponent_indices) do jj
            TestDynamics.state_indices(control_system, jj)[1:2]
        end
    end

    function add_objective!(opt_model, x, u; weights)
        T = size(x, 2)
        @views x_sub_ego = x[state_indices, :]
        @views u_sub_ego = u[input_indices, :]
        @views opponent_positions = map(opponent_position_indices) do opp_position_idxs
            x[opp_position_idxs, :]
        end

        J̃ = (;
            state_goal = isnothing(goal_position) ? 0 :
                         sum(el -> el^2, x_sub_ego[1:2, T_activate_goalcost:T] .- goal_position),
            control_Δv = sum(el -> el^2, u_sub_ego[1, :]),
            control_Δθ = sum(el -> el^2, u_sub_ego[2, :]),
        )
        @objective(
            opt_model,
            Min,
            sum(weights[k] * J̃[k] for k in keys(weights))
        )
    end 

    function add_objective_gradients!(opt_model, x, u; weights) 
        n_states, T = size(x)
        n_controls = size(u, 1)
        @views x_sub_ego = x[state_indices, :]
        @views u_sub_ego = u[input_indices, :]
        @views opponent_positions = map(opponent_position_indices) do opp_position_idxs
            x[opp_position_idxs, :]
        end
    
        dgoal_dxy =
            isnothing(goal_position) ? zeros(2, T) :
            hcat(
                zeros(2, T_activate_goalcost - 1),
                2 * (x_sub_ego[1:2, T_activate_goalcost:T] .- goal_position),
            )

        dJdx = 
        [
            zeros(first(state_indices) - 1, T)
            [dgoal_dxy; zeros(2, T)]
            zeros(n_states - last(state_indices), T)
        ]
    
        # TODO: Technically this is missing the negative gradient on the opponents state but we
        # can't control that anyway (certainly not in OL Nash). Must be fixed for non-decoupled
        # systems and potentially FB Nash.
        # dJdx = let
        #     dJ̃dx_sub = (;
        #         state_goal = [dgoal_dxy; zeros(2, T)],
        #         control_Δv = zeros(size(x_sub_ego)),
        #         control_Δθ = zeros(size(x_sub_ego)),
        #     )
        #     dJdx_sub =
        #         sum(weights[k] * dJ̃dx_sub[symbol(k)] for k in keys(weights))
        #     [
        #         zeros(first(state_indices) - 1, T)
        #         dJdx_sub
        #         zeros(n_states - last(state_indices), T)
        #     ]
        # end
    
        dJdu = let
            dJdu_sub =
                2 * [weights[:control_Δv], weights[:control_Δθ]] .* u_sub_ego
            [
                zeros(first(input_indices) - 1, T)
                dJdu_sub
                zeros(n_controls - last(input_indices), T)
            ]
        end
    
        (; dx = dJdx, du = dJdu)
    end 

    (; player_inputs = input_indices, weights, add_objective!, add_objective_gradients!, goal_position)
end

function generate_3dintegrator_cost(;
    player_idx,
    control_system::TestDynamics.ProductSystem,
    T,
    goal_position,
    weights = (; state_proximity = 1, state_velocity = 1, control_Δv = 1),
    cost_prescaling = (;
        state_proximity = 0.1,
        state_goal = 1,
        control_Δv = 10,
    ),
    prox_min_regularization = 0.1,
    T_activate_goalcost = T,
)
    state_indices = TestDynamics.state_indices(control_system, player_idx)
    input_indices = TestDynamics.input_indices(control_system, player_idx)

    opponent_position_indices = let
        opponent_indices = filter(!=(player_idx), eachindex(control_system.subsystems))
        map(opponent_indices) do jj
            TestDynamics.state_indices(control_system, jj)[1:3]
        end
    end

    function add_regularized_squared_distance!(opt_model, x_sub_ego, pos_opponent)
        @NLexpression(
            opt_model,
            [t = 1:T],
            (x_sub_ego[1, t] - pos_opponent[1, t])^2 +
            (x_sub_ego[2, t] - pos_opponent[2, t])^2 +
            (x_sub_ego[3, t] - pos_opponent[3, t])^2 +
            prox_min_regularization
        )
    end

    function add_objective!(opt_model, x, u; weights)
        T = size(x, 2)
        @views x_sub_ego = x[state_indices, :]
        @views u_sub_ego = u[input_indices, :]
        @views opponent_positions = map(opponent_position_indices) do opp_position_idxs
            x[opp_position_idxs, :]
        end

        J̃ = (;
            state_goal = isnothing(goal_position) ? 0 :
                         sum(el -> el^2, x_sub_ego[1:3, T_activate_goalcost:T] .- goal_position),
            state_proximity = let
                prox_cost = sum(opponent_positions) do pos_opponent
                    d_sq = add_regularized_squared_distance!(opt_model, x_sub_ego, pos_opponent)
                    prox_cost = @variable(opt_model, [t = 1:T])
                    @NLconstraint(opt_model, [t = 1:T], prox_cost[t] == -log(d_sq[t]))
                    prox_cost
                end
                sum(prox_cost)
            end,
            control_Δv = sum(el -> el^2, u_sub_ego[1:3, :]),
        )
        @objective(
            opt_model,
            Min,
            sum(weights[k] * cost_prescaling[k] * J̃[k] for k in keys(weights))
        )
    end

    function add_objective_gradients!(opt_model, x, u; weights)
        n_states, T = size(x)
        n_controls = size(u, 1)
        @views x_sub_ego = x[state_indices, :]
        @views u_sub_ego = u[input_indices, :]
        @views opponent_positions = map(opponent_position_indices) do opp_position_idxs
            x[opp_position_idxs, :]
        end
        
        dprox_dxyz = sum(opponent_positions) do pos_opponent
            d_sq = add_regularized_squared_distance!(opt_model, x_sub_ego, pos_opponent)
            dproxdx = @variable(opt_model, [t = 1:T])
            @NLconstraint(
                opt_model,
                [t = 1:T],
                dproxdx[t] == -2 * (x_sub_ego[1, t] - pos_opponent[1, t]) / d_sq[t]
            )
            dproxdy = @variable(opt_model, [t = 1:T])
            @NLconstraint(
                opt_model,
                [t = 1:T],
                dproxdy[t] == -2 * (x_sub_ego[2, t] - pos_opponent[2, t]) / d_sq[t]
            )
            dproxdz = @variable(opt_model, [t = 1:T])
            @NLconstraint(
                opt_model,
                [t = 1:T],
                dproxdz[t] == -2 * (x_sub_ego[3, t] - pos_opponent[3, t]) / d_sq[t]
            )
            [dproxdx'; dproxdy'; dproxdz']
        end

        dgoal_dxyz =
            isnothing(goal_position) ? zeros(3, T) :
            hcat(
                zeros(3, T_activate_goalcost - 1),
                2 * (x_sub_ego[1:3, T_activate_goalcost:T] .- goal_position),
            )

        # TODO: Technically this is missing the negative gradient on the opponents state but we
        # can't control that anyway (certainly not in OL Nash). Must be fixed for non-decoupled
        # systems and potentially FB Nash.
        dJdx = let
            dJ̃dx_sub = (;
                state_goal = [dgoal_dxyz; zeros(3, T)],
                state_proximity = [dprox_dxyz; zeros(3, T)],
                control_Δv = zeros(size(x_sub_ego))
            )
            dJdx_sub =
                sum(
                    weights[k] * cost_prescaling[symbol(k)] * dJ̃dx_sub[symbol(k)]
                    for k in keys(weights)
                )
            [
                zeros(first(state_indices) - 1, T)
                dJdx_sub
                zeros(n_states - last(state_indices), T)
            ]
        end

        dJdu = let
            dJdu_sub =
                2 * 
                [weights[:control_Δv], weights[:control_Δv], weights[:control_Δv]] .* 
                [cost_prescaling[:control_Δv], cost_prescaling[:control_Δv], cost_prescaling[:control_Δv]] .* 
                u_sub_ego
            [
                zeros(first(input_indices) - 1, T)
                dJdu_sub
                zeros(n_controls - last(input_indices), T)
            ]
        end

        (; dx = dJdx, du = dJdu)
    end

    (; player_inputs = input_indices, weights, add_objective!, add_objective_gradients!, goal_position)
end

end
