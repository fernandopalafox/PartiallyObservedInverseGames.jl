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
    ρ = params.ρ # KoZ radius
    ω = params.ω # Angular velocity of hyperplane

    # Breakout states
    states_1 = states[1:4,:]
    states_2 = states[5:8,:]
    T = size(states_1,2)

    # Calculate plot limits
    x_domain = extrema(states[[1,5],:]) .+ (-0.01, 0.01)
    y_domain = extrema(states[[2,6],:]) .+ (-0.01, 0.01)
    domain  = [minimum([x_domain[1],y_domain[1]]),maximum([x_domain[2],y_domain[2]])]

    # Define useful vectors
    function n_for_p1(t)
        [cos(α + ω * t), sin(α + ω * t)]
    end
    function n(t)
        -[cos(α + ω * t), sin(α + ω * t)]
    end

    n0_full = states_2[1:2] - states_1[1:2]
    α = atan(n0_full[2],n0_full[1])

    # Animation of trajectory 
    anim = @animate for i = 1:T
        # Plot trajectories
        plot(
            [states_1[1,1:i], states_2[1,1:i]], [states_1[2,1:i], states_2[2,1:i]], 
            legend = true, 
            title = params.title * "\nt = $i\nω = " * string(round(params.ω, digits = 5)),
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
            [states_2[1,i] + ρ * cos(θ) for θ in range(0,stop=2π,length=100)], 
            [states_2[2,i] + ρ * sin(θ) for θ in range(0,stop=2π,length=100)], 
            color = :black, 
            legend = false,
            fillalpha = 0.1,
            fill = true,
        )
        # Plot line from player 2 to hyperplane 
        ni = ρ*n(i-1)
        plot!([states_2[1,i],states_2[1,i] + ni[1]],
              [states_2[2,i],states_2[2,i] + ni[2]],
              arrow = true,
              color = :black)   
              
        # Plot hyperplane
        hyperplane_domain = 10*range(domain[1],domain[2],100)
        p_for_p2 = states_2[1:2,i] + ni
        plot!(hyperplane_domain .+ p_for_p2[1],
            [-ni[1]/ni[2]*x + p_for_p2[2] for x in hyperplane_domain],
            color = :black,
        )

    end
    gif(anim, fps = 5, "rotating_hyperplane_"*params.title*".gif")
end    

function visualize_rotating_hyperplanes(states, params) 

    # Parameters
    ρs = params.ρs # KoZ radius
    ωs = params.ωs # Angular velocity of hyperplane

    # Breakout states
    states_1 = states[1:4,:]
    states_2 = states[5:8,:]
    states_3 = states[9:12,:]
    T = size(states_1,2)

    # Calculate plot limits
    x_domain = extrema(states[[1,5,9],:]) .+ (-0.01, 0.01)
    y_domain = extrema(states[[2,6,10],:]) .+ (-0.01, 0.01)
    domain  = [minimum([x_domain[1],y_domain[1]]),maximum([x_domain[2],y_domain[2]])]

    Main.@infiltrate

    # Define useful vectors
    function n(t, α, ω)
        -[cos(α + ω * t), sin(α + ω * t)]
    end

    n0_1_2 = states_2[1:2] - states_1[1:2]
    n0_1_3 = states_3[1:2] - states_1[1:2]
    n0_2_3 = states_3[1:2] - states_2[1:2]
    αs = [atan(n0_1_2[2],n0_1_2[1]), atan(n0_1_3[2],n0_1_3[1]), atan(n0_2_3[2],n0_2_3[1])]

    # Animation of trajectory 
    anim = @animate for i = 1:T
        # Plot trajectories
        # plot(
        #     [states_1[1,1:i], states_2[1,1:i], states_3[1,1:i]], 
        #     [states_1[2,1:i], states_2[2,1:i], states_3[2,1:i]], 
        #     linecolor = [:blue :red :red],
        #     legend = true, 
        #     title = params.title * "\nt = $i\nω = " * string(params.ωs,),
        #     xlabel = "x", ylabel = "y", 
        #     size = (500,500),
        #     xlims = domain,
        #     ylims = domain,
        # )
        plot(
            states_1[1,1:i], 
            states_1[2,1:i], 
            legend = true, 
            title = params.title * "\nt = $i\nω = " * string(params.ωs),
            xlabel = "x", ylabel = "y", 
            size = (500,500),
            xlims = domain,
            ylims = domain
        )

        # Plot player positions  
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
        scatter!(
            [states_3[1,i]], [states_3[2,i]], 
            color = :red,
            markersize = 5,
            label = "P3",
        )

        # Plot KoZ around player 2
        plot!(
            [states_2[1,i] + ρs[1] * cos(θ) for θ in range(0,stop=2π,length=100)], 
            [states_2[2,i] + ρs[1] * sin(θ) for θ in range(0,stop=2π,length=100)], 
            color = :black, 
            legend = false,
            fillalpha = 0.1,
            fill = true,
        )
        # Plot KoZ around player 3
        plot!(
            [states_3[1,i] + ρs[2] * cos(θ) for θ in range(0,stop=2π,length=100)], 
            [states_3[2,i] + ρs[2] * sin(θ) for θ in range(0,stop=2π,length=100)], 
            color = :black, 
            legend = false,
            fillalpha = 0.1,
            fill = true,
        )
        # Plot line from player 2 to hyperplane 
        ni = ρs[1]*n(i, αs[1], ωs[1])
        plot!([states_2[1,i],states_2[1,i] + ni[1]],
              [states_2[2,i],states_2[2,i] + ni[2]],
              arrow = true,
              color = :black)   
        # Plot hyperplane
        hyperplane_domain = 10*range(domain[1],domain[2],100)
        p_for_p2 = states_2[1:2,i] + ni
        plot!(hyperplane_domain .+ p_for_p2[1],
            [-ni[1]/ni[2]*x + p_for_p2[2] for x in hyperplane_domain],
            color = :blue,
        )
        # Plot line from player 3 to hyperplane 1 
        ni = ρs[2]*n(i, αs[2], ωs[2])
        plot!([states_3[1,i],states_3[1,i] + ni[1]],
              [states_3[2,i],states_3[2,i] + ni[2]],
              arrow = true,
              color = :black)   
        # Plot hyperplane 1
        hyperplane_domain = 10*range(domain[1],domain[2],100)
        p_for_p3 = states_3[1:2,i] + ni
        plot!(hyperplane_domain .+ p_for_p3[1],
            [-ni[1]/ni[2]*x + p_for_p3[2] for x in hyperplane_domain],
            color = :blue,
        )
        # Plot line from player 3 to hyperplane 2
        ni = ρs[3]*n(i, αs[3], ωs[3])
        plot!([states_3[1,i],states_3[1,i] + ni[1]],
              [states_3[2,i],states_3[2,i] + ni[2]],
              arrow = true,
              color = :red)   
        # Plot hyperplane 3
        hyperplane_domain = 10*range(domain[1],domain[2],100)
        p_for_p3 = states_3[1:2,i] + ni
        plot!(hyperplane_domain .+ p_for_p3[1],
            [-ni[1]/ni[2]*x + p_for_p3[2] for x in hyperplane_domain],
            color = :red,
        )

        # Set domain
        plot!(xlims = domain,
              ylims = domain)
    end
    gif(anim, fps = 5, "rotating_hyperplane_"*params.title*".gif")
end    

end
