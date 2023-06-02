using CSV, DataFrames

# This script solves for the rotating hyperplane obstacle avoidance scenario 
# Assumptions
# - The number of players is 2
# - Data is formatted as a CSV file where 
#   - Every column corresponds to a time step
#   - Every row corresponds to a state variable
#   - States for player 1 are listed first, appended by states for player 2
#   - We know the players' objectives

player_angles = [0,pi/2]

# Load data 
data = Matrix(CSV.read("data/hyperplane_trajectory.csv", DataFrame, header = false))


# Define objective
function objective_1(x)
end

# Define equality constraints
    # Dynamics

# Define inequality constraints
    # Obstacle avoidance

# Turn into log-barrier problem

# Solve optimization problem