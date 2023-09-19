using GLMakie

"This code is used to generate the robustness plot. I should probably automate this, but for now it's fine."
let 
    set_theme!()
    text_size = 23
    convergence_vec = [1.0, 0.96, 0.92 ,0.93, 0.95, 0.96, 0.91, 0.91, 0.82, 0.66, 0.60, 0.66]*100
    x_vec = [0.90, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    fig_convergence = Figure(resolution = (800, 230), fontsize = text_size)
    ax_convergence = Axis(
        fig_convergence[1, 1],
        xlabel = "Input bound [N]",
        ylabel = "Convergence %",
        limits = (nothing, (0, 100)),

    )
    Makie.barplot!(
        ax_convergence,
        x_vec,
        convergence_vec,
    )
    rowsize!(fig_convergence.layout, 1, Aspect(1,0.2))

    save("figures/robustness.jpg", fig_convergence)
end