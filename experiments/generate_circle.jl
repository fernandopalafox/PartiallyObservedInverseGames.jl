
function generate_circle()
    # user parameters
    n_players = 3
    n_states_per_player = 4

    ΔT = 0.1
    t_real = 10.0
    scale = 100
    
    os_init = pi/4 # init. angle offset

    # Initial positions
    as = [-pi/2 + os_init*(i - 1) for i in 1:n_players] # angles

    # Generate trajectory
    T = Int(t_real / ΔT)
    x = hcat(
        [
            vcat(
                [
                    vcat(-scale * [cos(a + (t - 1) / T * pi), sin(a + (t - 1) / T * pi)], [0, 0]) for a in as
                ]...,
            ) for t in 1:T
        ]...,
    )

    # Plot it just to be sure
    visualize_rotating_hyperplanes(
        x,
        (;
            ΔT = 0.1,
            adjacency_matrix = zeros(Bool, n_players, n_players),
            n_players,
            n_states_per_player,
            goals = [scale*unitvector(a) for a in as],
        );
        title = "circle",
        koz = true,
        fps = 10.0,
    )

    # Save trajectory 
    CSV.write("data/circle_" * string(n_players) * "p.csv", DataFrame(x, :auto), header = false)

    return nothing
end
