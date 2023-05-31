using CSV, DataFrames

# This script solves the inverse game for the rotating hyperplane obstacle avoidance scenario 
# Assumptions
# - The number of players is 2
# - Data is formatted as a CSV file where 
#   - Every column corresponds to a time step
#   - Every row corresponds to a state variable
#   - States for player 1 are listed first, appeneded by states for player 2

# Load data 
data = Matrix(CSV.read("data/hyperplane_trajectory.csv", DataFrame, header = false))

# Define objective

# Define objective

# Define equality constraints

# Solve optimization problem 