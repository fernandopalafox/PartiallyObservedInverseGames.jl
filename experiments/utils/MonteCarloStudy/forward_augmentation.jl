function augment_with_forward_solution(
    estimates;
    solver,
    control_system,
    player_cost_models_gt,
    x0,
    T,
    verbose = false,
    kwargs...,
)
    @showprogress pmap(enumerate(estimates)) do (ii, estimate)
        # overwrite the weights of the ground truth model with the weights of the estimate.
        player_cost_models_est =
            map(player_cost_models_gt, estimate.estimate.player_weights) do cost_model_gt, weights
                merge(cost_model_gt, (; weights))
            end

        match_equilibrium = (; estimate.ground_truth.x)
        init = (; estimate.ground_truth.x, estimate.ground_truth.u)

        # solve the forward game at this point
        converged, forward_solution, forward_opt_model = solve_game(
            solver,
            control_system,
            player_cost_models_est,
            x0,
            T;
            match_equilibrium,
            init,
            kwargs...,
        )

        converged ||
            verbose && @warn "Forward solution augmentation did not converge on observation $ii."

        estimate = @set estimate.converged = converged
        estimate = @set estimate.estimate =
            (; estimate.estimate..., forward_solution.x, forward_solution.u)

        estimate
    end
end
