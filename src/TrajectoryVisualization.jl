module TrajectoryVisualization
using VegaLite: VegaLite
using Plots

export visualize_trajectory, visualize_trajectory_batch

#=========================== Visualization Interface for Dynamics  Models ==========================#

function trajectory_data end
"Converts the state matrix x into a tabular format of `trajectory_data` that can be handed to
visualize_trajectory"
trajectory_data(control_system, x, player)

#===================================== generic implementations =====================================#

function visualize_trajectory(control_system, x; kwargs...)
    td = trajectory_data(control_system, x)
    visualize_trajectory(td; kwargs...)
end

"""
Visualizes a single trajectory `x` given as a matrix of states for a preset `control_system`.
"""
function visualize_trajectory(
    trajectory_data;
    canvas = VegaLite.@vlplot(),
    x_position_domain = extrema(s.px for s in trajectory_data) .+ (-0.01, 0.01),
    y_position_domain = extrema(s.py for s in trajectory_data) .+ (-0.01, 0.01),
    draw_line = true,
    legend = false,
    group = "",
    opacity = 1.0,
    color_scale = VegaLite.@vlfrag(),
)
    legend_viz = if legend
        VegaLite.@vlfrag(orient = "top", title = "Method", padding = 5, symbolOpacity = 1.0)
    else
        false
    end

    trajectory_visualizer =
        VegaLite.@vlplot(
            encoding = {
                x = {"px:q", scale = {domain = x_position_domain}, title = "Position x [m]"},
                y = {"py:q", scale = {domain = y_position_domain}, title = "Position y [m]"},
                opacity = {value = opacity},
                order = "t:q",
                color =
                    VegaLite.@vlfrag(datum = group, legend = legend_viz, scale = color_scale),
                detail = {"player:n"},
            }
        ) + VegaLite.@vlplot(mark = {"point", size = 25, clip = true, filled = true})

    if draw_line
        trajectory_visualizer += VegaLite.@vlplot(mark = {"line", clip = true})
    end

    canvas + (trajectory_data |> trajectory_visualizer)
end

"""
Generic visualization of a single tarjectory `trajectory_data`. Here, the `trajectory_data` is
provided in a tabular format and is self-contained. That is, the table-liek format must have columns
have positions `px` and `py`, time `t`, and player identifier `player`.
"""
function visualize_trajectory_batch(
    control_system,
    trajectory_batch;
    canvas = VegaLite.@vlplot(opacity = {value = 0.2}, width = 200, height = 200),
    kwargs...,
)
    mapreduce(+, trajectory_batch; init = canvas) do x
        visualize_trajectory(control_system, x; kwargs...)
    end
end

function visualize_trajectory_batch(
    trajectory_data_batch;
    canvas = VegaLite.@vlplot(opacity = {value = 0.2}, width = 200, height = 200),
    kwargs...,
)
    mapreduce(+, trajectory_data_batch; init = canvas) do trajectory_data
        visualize_trajectory(trajectory_data; kwargs...)
    end
end

"""
Animation of a two-player collision avoidance game where player 1 is using a rotating hyperplane to 
compute it's control input. 
"""
function visualize_rotating_hyperplane(states, params) 
    ρs = params.ρs # KoZ radius
    ωs = params.ωs # Angular velocity of hyperplane
    αs = params.αs # Initial angle of hyperplane

    # Breakout states
    states_1 = states[1:4,:]
    states_2 = states[5:8,:]
    T = size(states_1,2)

    # Calculate plot limits
    x_domain = extrema(states[[1,5],:]) .+ (-0.01, 0.01)
    y_domain = extrema(states[[2,6],:]) .+ (-0.01, 0.01)
    domain  = [minimum([x_domain[1],y_domain[1]]),maximum([x_domain[2],y_domain[2]])]

    # Define useful vectors
    function n(t)
        [cos(αs[1,2] + ωs[1,2] * (t-1)), sin(αs[1,2] + ωs[1,2] * (t-1))]
    end

    # n0_full = states_2[1:2] - states_1[1:2]
    # α = atan(n0_full[2],n0_full[1])

    # Animation of trajectory 
    anim = @animate for i = 1:T
        # Plot trajectories
        plot(
            [states_1[1,1:i], states_2[1,1:i]], [states_1[2,1:i], states_2[2,1:i]], 
            legend = true, 
            title = params.title * "\nt = $i\nω = " * string(round(params.ωs[1,2], digits = 5)),
            xlabel = "x", ylabel = "y", 
            size = (500,500),
            xlims = domain,
            ylims = domain,
        )

        # Plot positions  
        scatter!(
            [states_1[1,i]], [states_1[2,i]], 
            color = :blue,
            markersize = 5,
            label = "P1",
        )
        scatter!(
            [states_2[1,i]], [states_2[2,i]], 
            color = :red,
            markersize = 5,
            label = "P2",
        )

        # Plot KoZ around player 2
        plot!(
            [states_2[1,i] + ρs[1,2] * cos(θ) for θ in range(0,stop=2π,length=100)], 
            [states_2[2,i] + ρs[1,2] * sin(θ) for θ in range(0,stop=2π,length=100)], 
            color = :blue, 
            legend = false,
            fillalpha = 0.1,
            fill = true,
        )
        # Plot line from player 2 to hyperplane 
        ni = ρs[1,2]*n(i)
        plot!([states_2[1,i],states_2[1,i] + ni[1]],
              [states_2[2,i],states_2[2,i] + ni[2]],
              arrow = true,
              color = :blue)   
              
        # Plot hyperplane
        hyperplane_domain = 10*range(domain[1],domain[2],100)
        p_for_p2 = states_2[1:2,i] + ni
        plot!(hyperplane_domain .+ p_for_p2[1],
            [-ni[1]/ni[2]*x + p_for_p2[2] for x in hyperplane_domain],
            color = :blue,
        )

    end
    gif(anim, fps = 5, "rotating_hyperplane_"*params.title*".gif")
end    

function visualize_rotating_hyperplanes(states, params; koz = true, fps = 5) 

    # Useful stuff
    pos_idx = vcat(
        [[1 2] .+ (player - 1) * params.n_states_per_player for player in 1:(params.n_players)]...,
    )
    couples = findall(params.adj_mat)
    colors = palette(:default)[1:(params.n_players)]
    T = size(states,2)

    # Breakout states
    x_domain = extrema(states[pos_idx[:, 1], :]) .+ (-0.01, 0.01)
    y_domain = extrema(states[pos_idx[:, 2], :]) .+ (-0.01, 0.01)
    domain = [minimum([x_domain[1], y_domain[1]]), maximum([x_domain[2], y_domain[2]])]    

    # Define useful vectors
    function n(t, α, ω)
        [cos(α + ω * (t-1)), sin(α + ω * (t-1))]
    end

    function n0(couple, states, pos_idx)
        states[pos_idx[couple[1], 1:2], 1] - states[pos_idx[couple[2], 1:2], 1] 
    end

    # Animation of trajectory 
    anim = @animate for i = 1:T
        # Plot trajectories
        plot(; legend = false, title = params.title, xlabel = "x", ylabel = "y", size = (500, 500))
        plot!(
            [states[pos_idx[player, 1], 1:i] for player in 1:(params.n_players)],
            [states[pos_idx[player, 2], 1:i] for player in 1:(params.n_players)]
        )
        scatter!(
            [states[pos_idx[player, 1], i] for player in 1:(params.n_players)],
            [states[pos_idx[player, 2], i] for player in 1:(params.n_players)],
            markersize = 5,
            color = colors,
        )

        # Plot KoZs
        
        for couple in couples
            if koz
                # Plot KoZ around hyperplane owner
                plot!(
                    [states[pos_idx[couple[2], 1], i] + params.ρs[couple[1], couple[2]] * cos(θ) for θ in range(0,stop=2π,length=100)], 
                    [states[pos_idx[couple[2], 2], i] + params.ρs[couple[1], couple[2]] * sin(θ) for θ in range(0,stop=2π,length=100)], 
                    color = colors[couple[1]], 
                    legend = false,
                    fillalpha = 0.1,
                    fill = true,
                )
            end
            # Plot hyperplane normal
            ni =
                params.ρs[couple[1], couple[2]] *
                n(i, params.αs[couple[1], couple[2]], params.ωs[couple[1], couple[2]])
            plot!(
                [states[pos_idx[couple[2], 1], i], states[pos_idx[couple[2], 1], i] + ni[1]],
                [states[pos_idx[couple[2], 2], i], states[pos_idx[couple[2], 2], i] + ni[2]],
                arrow = true,
                color = colors[couple[1]],
            )
            # Plot hyperplane 
            hyperplane_domain = 10*range(domain[1],domain[2],100)
            p = states[pos_idx[couple[2], 1:2], i] + ni
            plot!(hyperplane_domain .+ p[1],
                [-ni[1]/ni[2]*x + p[2] for x in hyperplane_domain],
                color = colors[couple[1]],
            )
        end

        # Set domain
        plot!(xlims = domain,
              ylims = domain)
    end
    gif(anim, fps = fps, "rotating_hyperplanes_"*params.title*".gif")
end    

function visualize_obs_pred(states, T_obs, params; koz = true, fps = 5) 

    # Useful stuff
    pos_idx = vcat(
        [[1 2] .+ (player - 1) * params.n_states_per_player for player in 1:(params.n_players)]...,
    )
    couples = findall(params.adj_mat)
    colors = palette(:default)[1:(params.n_players)]
    T = size(states,2)

    # Domain
    x_domain = extrema(states[pos_idx[:, 1], :]) .+ (-0.01, 0.01)
    y_domain = extrema(states[pos_idx[:, 2], :]) .+ (-0.01, 0.01)
    domain = [minimum([x_domain[1], y_domain[1]]), maximum([x_domain[2], y_domain[2]])]    

    # Define useful vectors
    function n(t, α, ω)
        [cos(α + ω * (t-1)), sin(α + ω * (t-1))]
    end

    function n0(couple, states, pos_idx)
        states[pos_idx[couple[1], 1:2], 1] - states[pos_idx[couple[2], 1:2], 1] 
    end

    # Animation of trajectory 
    anim = @animate for i = 1:T

        if i <= T_obs
            title = "$(params.title)\n t = $i/$T\nmode: observing"
        else
            title = "$(params.title)\n t = $i/$T\nmode: predicting"
        end

        # Plot trajectories
        plot(; legend = false, title = title, xlabel = "x", ylabel = "y", size = (500, 500), palette = colors)
        
        if i <= T_obs 
            plot!(
                [states[pos_idx[player, 1], 1:i] for player in 1:(params.n_players)],
                [states[pos_idx[player, 2], 1:i] for player in 1:(params.n_players)]
            )
            scatter!(
                [states[pos_idx[player, 1], i] for player in 1:(params.n_players)],
                [states[pos_idx[player, 2], i] for player in 1:(params.n_players)],
                markersize = 5,
                color = colors,
            )
        else
            plot!(
                [states[pos_idx[player, 1], 1:T_obs] for player in 1:(params.n_players)],
                [states[pos_idx[player, 2], 1:T_obs] for player in 1:(params.n_players)], 
                linestyle = :solid,
            )
            plot!(
                [states[pos_idx[player, 1], T_obs:i] for player in 1:(params.n_players)],
                [states[pos_idx[player, 2], T_obs:i] for player in 1:(params.n_players)], 
                linestyle = :dash,
            )
            scatter!(
                [states[pos_idx[player, 1], T_obs] for player in 1:(params.n_players)],
                [states[pos_idx[player, 2], T_obs] for player in 1:(params.n_players)],
                markersize = 5,
                color = colors,
            )
            scatter!(
                [states[pos_idx[player, 1], i] for player in 1:(params.n_players)],
                [states[pos_idx[player, 2], i] for player in 1:(params.n_players)],
                markersize = 5,
                markerstrokealpha = 1.0,
                markeralpha = 0.5,
                color = colors,
            )
        end

        # Plot KoZs
        opacity = clamp((i/T_obs)^4, 0, 1)
        for couple in couples
            if koz
                # Plot KoZ around hyperplane owner
                plot!(
                    [states[pos_idx[couple[2], 1], i] + params.ρs[couple[1], couple[2]] * cos(θ) for θ in range(0,stop=2π,length=100)], 
                    [states[pos_idx[couple[2], 2], i] + params.ρs[couple[1], couple[2]] * sin(θ) for θ in range(0,stop=2π,length=100)], 
                    color = colors[couple[1]], 
                    legend = false,
                    linealpha = 0.0,
                    fillalpha = 0.5 * opacity,
                    fill = true,
                )
            end
            # Plot hyperplane normal
            ni =
                params.ρs[couple[1], couple[2]] *
                n(i, params.αs[couple[1], couple[2]], params.ωs[couple[1], couple[2]])
            # plot!(
            #     [states[pos_idx[couple[2], 1], i], states[pos_idx[couple[2], 1], i] + ni[1]],
            #     [states[pos_idx[couple[2], 2], i], states[pos_idx[couple[2], 2], i] + ni[2]],
            #     arrow = true,
            #     color = colors[couple[1]],
            #     linealpha = opacity,
            # )
            # Plot hyperplane 
            hyperplane_domain = 10*range(domain[1],domain[2],100)
            p = states[pos_idx[couple[2], 1:2], i] + ni
            plot!(hyperplane_domain .+ p[1],
                [-ni[1]/ni[2]*x + p[2] for x in hyperplane_domain],
                color = colors[couple[1]],
                linealpha = 0.5 * opacity,
            )
        end

        # Set domain
        plot!(xlims = domain,
              ylims = domain)
    end
    gif(anim, fps = fps, "obs_pred_"*params.title*".gif")
end    

function animate_trajectory(states, params)
    pos_idx = vcat(
        [[1 2] .+ (player - 1) * params.n_states_per_player for player in 1:(params.n_players)]...,
    )

    # Plot limits
    x_domain = extrema(states[pos_idx[:, 1], :]) .+ (-0.01, 0.01)
    y_domain = extrema(states[pos_idx[:, 2], :]) .+ (-0.01, 0.01)
    domain = [minimum([x_domain[1], y_domain[1]]), maximum([x_domain[2], y_domain[2]])]

    T = size(states, 2)
    colors = palette(:default)[1:(params.n_players)]

    # Animation of trajectory 
    anim = @animate for i in 1:T

        # Plot trajectories
        plot(
            [states[pos_idx[player, 1], 1:i] for player in 1:(params.n_players)],
            [states[pos_idx[player, 2], 1:i] for player in 1:(params.n_players)],
            legend = false,
            title = params.title * "\nt = $i/$T",
            xlabel = "x",
            ylabel = "y",
            size = (500, 500),
            xlims = domain,
            ylims = domain,
        )
        scatter!(
            [states[pos_idx[player, 1], i] for player in 1:(params.n_players)],
            [states[pos_idx[player, 2], i] for player in 1:(params.n_players)],
            markersize = 5,
            color = colors,
        )
        # Set domain
        plot!(xlims = domain, ylims = domain)
    end
    gif(anim, fps = 5, params.title * ".gif")
end

end
