using Plots

function n(θ)
    [1 - (θ)^2/2, θ]
end
 
T = 100

θs = range(0, 2π, length = T)


anim = @animate for i = 1:T
    # Plot trajectories
    plot(; legend = false, xlabel = "x", ylabel = "y", size = (500, 500))
    scatter!([n(θs[i])[1]], [n(θs[i])[2]], arrow = true)
    plot!(xlims = 5, ylims = 5)
    plot!(xlims = [-5,5],
          ylims = [-5,5])
end
gif(anim, fps = 10, "small_angle.gif")