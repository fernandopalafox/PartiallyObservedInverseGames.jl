unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "experiments/utils/MonteCarloStudy"))))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "experiments/utils"))))
unique!(push!(LOAD_PATH, realpath(joinpath(project_root_dir, "test/utils"))))

using Distributed: Distributed

Distributed.@everywhere begin
    using Pkg: Pkg
    Pkg.activate($project_root_dir)
    union!(LOAD_PATH, $LOAD_PATH)

    using MonteCarloStudy: MonteCarloStudy
    using CollisionAvoidanceGame: CollisionAvoidanceGame
    using TestDynamics: TestDynamics
    using PartiallyObservedInverseGames.ForwardGame: ForwardGame, IBRGameSolver, KKTGameSolver
    using PartiallyObservedInverseGames.InverseGames:
        InverseKKTConstraintSolver,
        InverseKKTResidualSolver,
        AugmentedInverseKKTResidualSolver,
        solve_inverse_game
    using PartiallyObservedInverseGames.InversePreSolve: InversePreSolve
end

using VegaLite: VegaLite
using Random: Random
using Distributor: Distributor
using CostHeatmapVisualizer: CostHeatmapVisualizer
using PartiallyObservedInverseGames.TrajectoryVisualization:
    TrajectoryVisualization, visualize_trajectory

# Utils
include("misc.jl")
include("simple_caching.jl")
