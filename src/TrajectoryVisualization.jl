module TrajectoryVisualization
using VegaLite: VegaLite
using Plots
# using GLMakie: lines, Axis3, Point3f, record, set_theme!, theme_black, Observable, Figure, hidespines!, mesh!, hidezdecorations!, notify, scatter!, lines!, Makie.RGBA
using GLMakie
using GeometryBasics: Cylinder, Circle, Polygon, mesh
import GeometryBasics
using DataStructures: CircularBuffer
using LinearAlgebra: norm

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
    ρs = params.ρs # KoZs radius
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

        # Plot KoZs around player 2
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

function visualize_rotating_hyperplanes(states, params; title = "", koz = true, fps = 5) 

    # Useful stuff
    position_indices = vcat(
        [[1 2] .+ (player - 1) * params.n_states_per_player for player in 1:(params.n_players)]...,
    )
    couples = findall(params.adjacency_matrix)
    colors = palette(:default)[1:(params.n_players)]
    T = size(states,2)

    # Breakout states
    x_domain = extrema(states[position_indices[:, 1], :]) .+ (-0.01, 0.01)
    y_domain = extrema(states[position_indices[:, 2], :]) .+ (-0.01, 0.01)
    domain = [minimum([x_domain[1], y_domain[1]]), maximum([x_domain[2], y_domain[2]])]    

    θs = zeros(length(couples))
    player_couples = [findall(couple -> couple[1] == player_idx, couples) for player_idx in 1:params.n_players] 
    for player_idx in 1:params.n_players
        for (couple_idx, couple) in enumerate(couples[player_couples[player_idx]])
            parameter_idx = player_couples[player_idx][couple_idx]
            idx_ego   = (1:2) .+ (couple[1] - 1)*params.n_states_per_player
            idx_other = (1:2) .+ (couple[2] - 1)*params.n_states_per_player
            x_ego   = states[idx_ego,1]
            x_other = states[idx_other,1]
            x_diff  = x_ego - x_other
            θ = atan(x_diff[2], x_diff[1])

            θs[parameter_idx] = θ
        end
    end

    # Define useful vectors
    function n(t, θ, α, ω)
        [cos(θ + α + ω * (t - 1) * params.ΔT), sin(θ + α + ω * (t - 1) * params.ΔT)]
    end

    function n0(couple, states, position_indices)
        states[position_indices[couple[1], 1:2], 1] - states[position_indices[couple[2], 1:2], 1] 
    end

    # Animation of trajectory 
    anim = @animate for i = 1:T
        # Plot trajectories
        Plots.plot(; legend = false, title = title, xlabel = "x", ylabel = "y", size = (500, 500))
        Plots.plot!(
            [states[position_indices[player, 1], 1:i] for player in 1:(params.n_players)],
            [states[position_indices[player, 2], 1:i] for player in 1:(params.n_players)]
        )
        Plots.scatter!(
            [states[position_indices[player, 1], i] for player in 1:(params.n_players)],
            [states[position_indices[player, 2], i] for player in 1:(params.n_players)],
            markersize = 5,
            color = colors,
        )
        # plot goals from params info with an x
        Plots.scatter!(
            [params.goals[player][1] for player in 1:(params.n_players)],
            [params.goals[player][2] for player in 1:(params.n_players)],
            markersize = 5,
            marker = :x,
            color = colors,
        )

        # Plot KoZs
        for (couple_idx, couple)  in enumerate(couples)
            if koz
                # Plot KoZs around hyperplane owner
                Plots.plot!(
                    [states[position_indices[couple[2], 1], i] + params.ρs[couple_idx] * cos(θ) for θ in range(0,stop=2π,length=100)], 
                    [states[position_indices[couple[2], 2], i] + params.ρs[couple_idx] * sin(θ) for θ in range(0,stop=2π,length=100)], 
                    color = colors[couple[1]], 
                    legend = false,
                    fillalpha = 0.1,
                    fill = true,
                )
            end
            # Plot hyperplane normal
            ni =
                params.ρs[couple_idx] *
                n(i, θs[couple_idx], params.αs[couple_idx], params.ωs[couple_idx])
            Plots.plot!(
                [states[position_indices[couple[2], 1], i], states[position_indices[couple[2], 1], i] + ni[1]],
                [states[position_indices[couple[2], 2], i], states[position_indices[couple[2], 2], i] + ni[2]],
                arrow = true,
                color = colors[couple[1]],
            )
            # Plot hyperplane 
            hyperplane_domain = 10*range(domain[1],domain[2],100)
            p = states[position_indices[couple[2], 1:2], i] + ni
            Plots.plot!(hyperplane_domain .+ p[1],
                [-ni[1]/ni[2]*x + p[2] for x in hyperplane_domain],
                color = colors[couple[1]],
            )
        end

        # Set domain
        Plots.plot!(xlims = domain,
              ylims = domain)
    end
    gif(anim, fps = fps, "rotating_hyperplanes_"*title*".gif")
end    

function visualize_obs_pred(states, T_obs, params; koz = true, fps = 5) 

    # Useful stuff
    position_indices = vcat(
        [[1 2] .+ (player - 1) * params.n_states_per_player for player in 1:(params.n_players)]...,
    )
    couples = findall(params.adj_mat)
    colors = palette(:default)[1:(params.n_players)]
    T = size(states,2)

    # Domain
    x_domain = extrema(states[position_indices[:, 1], :]) .+ (-0.01, 0.01)
    y_domain = extrema(states[position_indices[:, 2], :]) .+ (-0.01, 0.01)
    domain = [minimum([x_domain[1], y_domain[1]]), maximum([x_domain[2], y_domain[2]])]    

    # Define useful vectors
    function n(t, α, ω)
        [cos(α + ω * (t-1)), sin(α + ω * (t-1))]
    end

    function n0(couple, states, position_indices)
        states[position_indices[couple[1], 1:2], 1] - states[position_indices[couple[2], 1:2], 1] 
    end

    # Animation of trajectory 
    anim = @animate for i = 1:T

        if i <= T_obs
            title = "$(params.title)\n t = " *string(round(i*params.ΔT; digits = 2)) * "/" * string(round(params.ΔT*T; digits = 2)) * "s" *"\nmode: observing"
        else
            title = "$(params.title)\n t = " *string(round(i*params.ΔT; digits = 2)) * "/" * string(round(params.ΔT*T; digits = 2)) * "s" *"\nmode: predicting"
        end

        # Plot trajectories
        plot(; legend = false, title = title, xlabel = "x", ylabel = "y", size = (500, 500), palette = colors)
        
        if i <= T_obs 
            plot!(
                [states[position_indices[player, 1], 1:i] for player in 1:(params.n_players)],
                [states[position_indices[player, 2], 1:i] for player in 1:(params.n_players)]
            )
            scatter!(
                [states[position_indices[player, 1], i] for player in 1:(params.n_players)],
                [states[position_indices[player, 2], i] for player in 1:(params.n_players)],
                markersize = 5,
                color = colors,
            )
        else
            plot!(
                [states[position_indices[player, 1], 1:T_obs] for player in 1:(params.n_players)],
                [states[position_indices[player, 2], 1:T_obs] for player in 1:(params.n_players)], 
                linestyle = :solid,
            )
            plot!(
                [states[position_indices[player, 1], T_obs:i] for player in 1:(params.n_players)],
                [states[position_indices[player, 2], T_obs:i] for player in 1:(params.n_players)], 
                linestyle = :dash,
            )
            scatter!(
                [states[position_indices[player, 1], T_obs] for player in 1:(params.n_players)],
                [states[position_indices[player, 2], T_obs] for player in 1:(params.n_players)],
                markersize = 5,
                color = colors,
            )
            scatter!(
                [states[position_indices[player, 1], i] for player in 1:(params.n_players)],
                [states[position_indices[player, 2], i] for player in 1:(params.n_players)],
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
                # Plot KoZs around hyperplane owner
                plot!(
                    [states[position_indices[couple[2], 1], i] + params.ρs[couple[1], couple[2]] * cos(θ) for θ in range(0,stop=2π,length=100)], 
                    [states[position_indices[couple[2], 2], i] + params.ρs[couple[1], couple[2]] * sin(θ) for θ in range(0,stop=2π,length=100)], 
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
            #     [states[position_indices[couple[2], 1], i], states[position_indices[couple[2], 1], i] + ni[1]],
            #     [states[position_indices[couple[2], 2], i], states[position_indices[couple[2], 2], i] + ni[2]],
            #     arrow = true,
            #     color = colors[couple[1]],
            #     linealpha = opacity,
            # )
            # Plot hyperplane 
            hyperplane_domain = 10*range(domain[1],domain[2],100)
            p = states[position_indices[couple[2], 1:2], i] + ni
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

function animate_trajectory(states, params; fps = 5)
    position_indices = vcat(
        [[1 2] .+ (player - 1) * params.n_states_per_player for player in 1:(params.n_players)]...,
    )

    # Plot limits
    extra = 0.1 
    x_domain = extrema(states[position_indices[:, 1], :]) .+ (-extra, extra)
    y_domain = extrema(states[position_indices[:, 2], :]) .+ (-extra, extra)
    domain = [minimum([x_domain[1], y_domain[1]]), maximum([x_domain[2], y_domain[2]])]

    T = size(states, 2)
    colors = palette(:default)[1:(params.n_players)]

    # Animation of trajectory 
    anim = Plots.@animate for i in 1:T

        # Plot trajectories
        Plots.plot(
            [states[position_indices[player, 1], 1:i] for player in 1:(params.n_players)],
            [states[position_indices[player, 2], 1:i] for player in 1:(params.n_players)],
            legend = false,
            title = params.title * "\nt = " *string(round(i*params.ΔT; digits = 2)) * "/" * string(params.ΔT*T) * "s",
            xlabel = "x",
            ylabel = "y",
            size = (500, 500),
            xlims = domain,
            ylims = domain,
        )
        Plots.scatter!(
            [states[position_indices[player, 1], i] for player in 1:(params.n_players)],
            [states[position_indices[player, 2], i] for player in 1:(params.n_players)],
            markersize = 5,
            color = colors,
        )
        # Plot goals    
        Plots.scatter!(
            [params.goals[player][1] for player in 1:(params.n_players)],
            [params.goals[player][2] for player in 1:(params.n_players)],
            markersize = 10,
            marker = :star4,
            color = colors,
        )

        # Set domain
        Plots.plot!(xlims = domain, ylims = domain)
    end
    Plots.gif(anim, fps = fps, params.title * ".gif")
end

"Plot trajectory comparison in two-dimensions, since this is where hyperplanes live"
function trajectory_comparison(states_1, states_2, parameters; title = "Forward Game Trajectory Comparison", filename = "comparison.png")

    timesteps = size(states_1, 2)
    position_indices = vcat(
            [[1 2 3] .+ (player - 1) * parameters.n_states_per_player for player in 1:(parameters.n_players)]...,
        )

    extra = maximum(parameters.ρs) + 0.05
    x_domain = extrema(states_1[position_indices[:, 1], :]) .+ (-extra, extra)
    y_domain = extrema(states_1[position_indices[:, 2], :]) .+ (-extra, extra)
    # xy_domain = [minimum([x_domain[1], y_domain[1]]), maximum([x_domain[2], y_domain[2]])]

    set_theme!()
    fig = Figure(); 

    # Setup grids 
    panel_1 = fig[1, 1] = GridLayout()
    panel_2 = fig[1, 2] = GridLayout()
    axis_1 = Axis(
        panel_1[1, 1],
        limits = (x_domain, y_domain),
        xlabel = "Position x [m]",
        ylabel = "Position y [m]",
        title = "KKT",
        xgridvisible = false,
        ygridvisible = false,
    )
    axis_2 = Axis(
        panel_2[1, 1],
        limits = (x_domain, y_domain),
        xlabel = "Position x [m]",
        title = "Our method",
        xgridvisible = false,
        ygridvisible = false,
    )
    hideydecorations!(axis_2)
    
    

    # For every player, at every point, measure the closest approach to any other player
    # Every player should be left with a vector of length horizon corresponding to the closest approach to every other player
    # Color is inversely proportional to this distance
    
    function compute_player_colors(states)
        # Create an empty vector of vectors to store the colors
        player_colors = Vector{Vector{Float64}}(undef, 0) 

        for idx_ego in 1:parameters.n_players
            min_distance_to_other = ones(timesteps) .* Inf
            pos_ego = states[position_indices[idx_ego, 1:2], :]
            for idx_other in setdiff(1:parameters.n_players, idx_ego)
                pos_other = states[position_indices[idx_other, 1:2], :]
                distance_to_other = map(norm, eachcol(pos_ego .- pos_other))
                min_distance_to_other = vec(minimum(hcat(min_distance_to_other, distance_to_other), dims = 2)) # In Julia 1.9 norm has dim arg 
            end
            # Print player i minimum distance to others
            println("   Player $idx_ego min distance to others: ", minimum(min_distance_to_other))
            # push!(player_colors, 1 ./ min_distance_to_other) 
            push!(player_colors, min_distance_to_other) 
        end
        player_colors
    end
    player_colors_1 = compute_player_colors(states_1)
    player_colors_2 = compute_player_colors(states_2)

    min_player_colors_1 = minimum([minimum(player_colors_1[player]) for player in 1:(parameters.n_players)])
    min_player_colors_2 = minimum([minimum(player_colors_2[player]) for player in 1:(parameters.n_players)])
    max_player_colors_1 = maximum([maximum(player_colors_1[player]) for player in 1:(parameters.n_players)])
    max_player_colors_2 = maximum([maximum(player_colors_2[player]) for player in 1:(parameters.n_players)])

    # Normalize colors by the maximum value between the two data sets 
    # max_color_1 = maximum([maximum(player_colors_1[player]) for player in 1:(parameters.n_players)])
    # max_color_2 = maximum([maximum(player_colors_2[player]) for player in 1:(parameters.n_players)])
    # max_color = maximum([max_color_1, max_color_2])
    # player_colors_1 = [player_colors_1[player] ./ max_color for player in 1:(parameters.n_players)]
    # player_colors_2 = [player_colors_2[player] ./ max_color for player in 1:(parameters.n_players)]

    # Plot first set of trajectories 
    # :cool 
    # :RdYlGn_11
    track_color_symbol = :RdYlGn_11
    track_color_gradient = cgrad(track_color_symbol)
    track_width = 5
    marker_colors = palette(:default)[1:parameters.n_players]
    for player in 1:parameters.n_players 
        lines!(
            axis_1,
            states_1[position_indices[player, 1:2], :];
            linewidth = track_width,
            color = [track_color_gradient[z] for z in player_colors_1[player]],
            # label = "KKT trajectory"
        )

        # Plot player goals 
        Makie.scatter!(
            axis_1,
            Point2f(parameters.goals[player]);
            markersize = 20,
            marker = :star4,
            color = marker_colors[player],
            label = "Player goal"
        )

        # Plot player start
        Makie.scatter!(
            axis_1,
            Point2f(states_1[position_indices[player, 1:2], 1]);
            markersize = 20,
            marker = :circle,
            color = marker_colors[player],
            label = "Player start"
        )
    end   

    for player in 1:parameters.n_players 
        lines!(
            axis_2,
                states_2[position_indices[player, 1:2], :];
                linewidth = track_width,
                color = [track_color_gradient[z] for z in player_colors_2[player]],
                # label = "Hyperplane trajectory"
            )

        # Plot player goals 
        Makie.scatter!(
            axis_2,
            Point2f(parameters.goals[player]);
            markersize = 20,
            marker = :star4,
            color = marker_colors[player],
            label = "Player goal"
        )

        # Plot player start
        Makie.scatter!(
            axis_2,
            Point2f(states_1[position_indices[player, 1:2], 1]);
            markersize = 20,
            marker = :circle,
            color = marker_colors[player],
            label = "Player start"
        )
    end   

    # Colorbar
    Colorbar(
        fig[1, 3],
        limits = (
            minimum([min_player_colors_1, min_player_colors_2]),
            maximum([max_player_colors_1, max_player_colors_2]),
        ),
        colormap = track_color_symbol,
        label = "closest approach",
    )

    # legend
    Legend(
        fig[2, 1:2],
        axis_1,
        framevisible = false,
        merge = true,
        unique = true,
        orientation = :horizontal,
        valign = :top,
        tellwidth = false, 
        tellheight = true
    )

    save(filename, fig)
    display(fig)
end

function display_3D_trajectory(states, parameters; title = "title", filename = "3D_trajectory.gif", hyperplane = false)

    position_indices = vcat(
        [[1 2 3] .+ (player - 1) * parameters.n_states_per_player for player in 1:(parameters.n_players)]...,
    )

    if hyperplane
        extra = maximum(parameters.ρs) + 0.05
    else
        extra = 0.1
    end
    x_domain = extrema(states[position_indices[:, 1], :]) .+ (-extra, extra)
    y_domain = extrema(states[position_indices[:, 2], :]) .+ (-extra, extra)
    z_domain = extrema(states[position_indices[:, 3], :]) .+ (-extra, extra)
    xy_domain = [minimum([x_domain[1], y_domain[1]]), maximum([x_domain[2], y_domain[2]])]

    # Setup observables
    # Trajectory
    tail = 50
    player_points = [Observable(Point3f(states[position_indices[i,:], 1])) for i in 1:parameters.n_players]
    # player_tracks = [Observable([Point3f(states[position_indices[i,:], 1])]) for i in 1:parameters.n_players]
    player_tracks = [
        Observable(
            fill!(CircularBuffer{Point3f}(tail), Point3f(states[position_indices[i, :], 1])),
        ) for i in 1:(parameters.n_players)
    ]

    if hyperplane
        # KoZ
        KoZs = [
            Observable(
                Cylinder(
                    Point3f([states[position_indices[couple[2], 1:2], 1]; z_domain[1]]),
                    Point3f([states[position_indices[couple[2], 1:2], 1]; z_domain[2]]),
                    Float32(ρ),
                ),
            ) for (couple, ρ) in zip(parameters.couples, parameters.ρs)
        ]
        # KoZ projection 
        function draw_koz_projection(center, radius)
            hcat(
                [center[1] + radius * cos(θ) for θ in range(0, stop = 2π, length = 100)],
                [center[2] + radius * sin(θ) for θ in range(0, stop = 2π, length = 100)],
                z_domain[1] .* ones(100),
            )
        end
        KoZ_projections = [
            Observable(draw_koz_projection(states[position_indices[couple[2], 1:2], 1], ρ))
            for (couple, ρ) in zip(parameters.couples, parameters.ρs)
        ]

        # Hyperplane 
        θs = zeros(length(parameters.couples))
        for (couple_idx, couple) in enumerate(parameters.couples)
            idx_ego   = (1:2) .+ (couple[1] - 1)*parameters.n_states_per_player
            idx_other = (1:2) .+ (couple[2] - 1)*parameters.n_states_per_player
            x_ego   = states[idx_ego,1]
            x_other = states[idx_other,1]
            x_diff  = x_ego - x_other
            θ = atan(x_diff[2], x_diff[1])

            θs[couple_idx] = θ
        end
        hyperplane_x = collect(range(xy_domain[1], xy_domain[2], 1000))
        function n(t, θ, α, ω)
            [cos(θ + α + ω * (t - 1) * parameters.ΔT), sin(θ + α + ω * (t - 1) * parameters.ΔT)]
        end
        function draw_hyperplane_projection(i, couple, couple_idx)
            ni = parameters.ρs[couple_idx] * n(i, θs[couple_idx], parameters.αs[couple_idx], parameters.ωs[couple_idx])
            p = states[position_indices[couple[2], 1:2], i] + ni
            hyperplane_y = [-ni[1] / ni[2] * (x - p[1]) + p[2] for x in hyperplane_x]
            
            idx = findall((hyperplane_y .> xy_domain[1]) .* (hyperplane_y .< xy_domain[2]))
            if length(idx) < 2 # Case where ni[2] is close to zero
                hcat(p[1] .* ones(length(hyperplane_x)), 
                    hyperplane_x, 
                    z_domain[1] .* ones(length(hyperplane_x))
                    )
            else
                hcat(
                    hyperplane_x[idx], 
                    hyperplane_y[idx],
                    z_domain[1] .* ones(length(idx)),
                )
            end
        end
        hyperplanes_xy = [
            Observable(draw_hyperplane_projection(1, couple, couple_idx))
            for (couple_idx, couple) in enumerate(parameters.couples)
        ]

        function draw_hyperplane(i, couple, couple_idx)
            # Hyperplane projection 
            hyperplane_xy = draw_hyperplane_projection(i, couple, couple_idx)
    
            # Hyperplane normal 
            ni = n(i, θs[couple_idx], parameters.αs[couple_idx], parameters.ωs[couple_idx])
    
            # Find min and max x
            _, x_min_idx = findmin(hyperplane_xy[:,1])
            _, x_max_idx = findmax(hyperplane_xy[:,1])
            if x_min_idx == x_max_idx
                _, x_max_idx = findmax(hyperplane_xy[:,2])
            end
    
            thickness = 0.01
            # Create hyperplane vertices
            # with thickness 
            face_vertices = 
            [
                hyperplane_xy[x_min_idx, 1] hyperplane_xy[x_min_idx, 2] z_domain[1];
                hyperplane_xy[x_max_idx, 1] hyperplane_xy[x_max_idx, 2] z_domain[1];
                hyperplane_xy[x_max_idx, 1] hyperplane_xy[x_max_idx, 2] z_domain[2];
                hyperplane_xy[x_min_idx, 1] hyperplane_xy[x_min_idx, 2] z_domain[2];
            ]
            [
                face_vertices
                face_vertices .+ thickness .* [ni[1] ni[2] 0]
            ]
        end
        hyperplanes = [
            Observable(draw_hyperplane(1, couple, couple_idx))
            for (couple_idx, couple) in enumerate(parameters.couples)
        ]
    end
    
    set_theme!(theme_light())
    fig = Figure(); display(fig)
    ax = Axis3(
        fig[1, 1],
        aspect = :equal,
        limits = (xy_domain, xy_domain, z_domain),
        viewmode = :fit,
        title = title,
        xypanelvisible = false,
        perspectiveness = 0.5
    )
    # hidedecorations!(ax; top = false)
    hidespines!(ax)
    colors = palette(:default)[1:parameters.n_players]

    for player in 1:parameters.n_players
        # Player points
        Makie.scatter!(
            ax,
            player_points[player];
            color = colors[player],
            markersize = 15,
            marker = :utriangle
        )

        # Player track
        track_color = [Makie.RGBA(colors[player],(i/tail)^2) for i in 1:length(player_tracks[player][])]
        lines!(ax, player_tracks[player]; linewidth = 2, color = track_color, transparency = true)

        # Player goals 
        Makie.scatter!(
            ax,
            Point3f(parameters.goals[player]);
            markersize = 15,
            marker = :star4,
            color = colors[player],
        )

        # Projections on xy plane (if hyperplane)
        if hyperplane
            # Player points
            Makie.scatter!(
                ax,
                @lift(Point3f([$(player_points[player])[1:2]; z_domain[1]]));
                markersize = 12.5,
                marker = :utriangle,
                color = Makie.RGBA(colors[player], 0.3),
            )

            # Player track
            track_color = [Makie.RGBA(colors[player],0.3*(i/tail)^2) for i in 1:length(player_tracks[player][])]
            lines!(
                ax,
                @lift([
                    Point3f([$(player_tracks[player])[i][1:2]; z_domain[1]]) for
                    i in 1:length($(player_tracks[player]))
                ]);
                linewidth = 1.5,
                color = track_color,
            )

            # Player goals 
            Makie.scatter!(
                ax,
                Point3f([parameters.goals[player][1:2]; z_domain[1]]);
                markersize = 12.5,
                marker = :star4,
                color = Makie.RGBA(colors[player], 0.3),
            )
        end
    end
    # Hyperplane 
    if hyperplane
        for (couple_idx, couple) in enumerate(parameters.couples)
            # KoZs
            mesh!(
                ax,
                KoZs[couple_idx];
                color = Makie.RGBA(colors[couple[2]], 0.1),
                transparency = true,
                shading = true,
            )

            # Projection of KoZs
            lines!(ax, KoZ_projections[couple_idx], color = Makie.RGBA(colors[couple[2]], 0.25))

            # Hyperplanes on xy
            lines!(ax, hyperplanes_xy[couple_idx], color = Makie.RGBA(colors[couple[2]], 0.25))

            # Hyperplanes in 3D
            mesh!(
                ax,
                hyperplanes[couple_idx],
                [
                    1 2 3 # front bottom 
                    1 3 4 # front top
                    5 6 7 # back bottom
                    5 7 8 # back top
                    1 2 6 # left bottom
                    1 5 6 # left top
                    2 3 7 # right bottom
                    2 6 7 # right top
                    3 4 8 # top bottom
                    3 7 8 # top top
                    4 1 5 # bottom bottom
                    4 5 8 # bottom top
                ];
                color = Makie.RGBA(colors[couple[2]], 0.15),
                transparency = true,
                shading = true,
            )
        end
    end

    record(fig, filename, 1:length(states[1, :])) do frame

            for player in 1:parameters.n_players
                # Update player points
                player_points[player][] = Point3f(states[position_indices[player, :], frame])
                notify(player_points[player])

                # Update player track
                push!(player_tracks[player][], Point3f(states[position_indices[player, :], frame]))
                notify(player_tracks[player])
            end

            if hyperplane
                for (couple_idx, (couple, ρ)) in enumerate(zip(parameters.couples, parameters.ρs))
                    # Update KoZs 
                    KoZs[couple_idx][] = Cylinder(
                        Point3f([states[position_indices[couple[2], 1:2], frame]; z_domain[1]]),
                        Point3f([states[position_indices[couple[2], 1:2], frame]; z_domain[2]]),
                        Float32(ρ),
                    )
                    notify(KoZs[couple_idx])

                    # Update KoZs projection
                    KoZ_projections[couple_idx][] = draw_koz_projection(states[position_indices[couple[2], 1:2], frame], ρ)
                    notify(KoZ_projections[couple_idx])

                    # Update hyperplanes on xy
                    hyperplanes_xy[couple_idx][] = draw_hyperplane_projection(frame, couple, couple_idx)
                    notify(hyperplanes_xy[couple_idx])

                    # Update hyperplanes in 3D
                    hyperplanes[couple_idx][] = draw_hyperplane(frame, couple, couple_idx)
                    notify(hyperplanes[couple_idx])
                end
            end

            ax.azimuth[] = pi/3 - (pi/4 - 0.01) * sin(2pi * frame / 380) # ROtate 
            # ax.elevation[] = pi/6 + 0.9 * sin(2pi * frame / 400)
            # ax.azimuth[] = 1.7*pi + pi * (frame - 1) / length(states[1, :])
    end

    # Display last frame 
    display(fig)

end

end
