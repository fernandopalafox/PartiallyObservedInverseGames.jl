using Plots

function test_dynamics()
    # Satellite parameters
    T = 1000
    ΔT = 0.1
    m  = 100.0 # kg
    r₀ = (400 + 6378.137) # km
    μ  = 398600.4418 # km^3/s^2
    n = sqrt(μ/(r₀^3)) # rad/s

    A_d = [ 4-3*cos(n*ΔT)      0  1/n*sin(n*ΔT)     2/n*(1-cos(n*ΔT));
            6*(sin(n*ΔT)-n*ΔT) 1 -2/n*(1-cos(n*ΔT)) 1/n*(4*sin(n*ΔT)-3*n*ΔT);
            3*n*sin(n*ΔT)      0  cos(n*ΔT)         2*sin(n*ΔT);
            -6*n*(1-cos(n*ΔT))  0 -2*sin(n*ΔT)       4*cos(n*ΔT)-3];

    B_d = 1/m*[ 1/n^2(1-cos(n*ΔT))   2/n^2*(n*ΔT-sin(n*ΔT));
                -2/n*(n*ΔT-sin(n*ΔT)) 4/n^2*(1-cos(n*ΔT))-3/2*ΔT^2;
                1/n*sin(n*ΔT)        2/n*(1-cos(n*ΔT));
                -2/n*(1-cos(n*ΔT))    4/n*sin(n*ΔT)-3*ΔT]

    # Run sim for a satellite starting slightly offset from origin and stationary and NO input
    x0 = [1.0, 0.0, 0.0, 0.0]
    x = zeros(4, T)
    x[:, 1] = x0

    for i in 2:T
        x[:, i] = A_d*x[:, i-1] + B_d*[0.0, 0.0]
    end

    # Plot trajectory in 2d 
    plot(x[1, :], x[2, :], label="trajectory")

    Main.@infiltrate
end