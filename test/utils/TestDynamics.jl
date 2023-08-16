module TestDynamics

import Plots
import VegaLite
import PartiallyObservedInverseGames.DynamicsModelInterface
import PartiallyObservedInverseGames.TrajectoryVisualization
import JuMP
using JuMP: @variable, @constraint, @NLconstraint

include("unicycle.jl")
include("hyperunicycle.jl")
include("product_system.jl")
include("doubleintegrator.jl")
include("satellite2d.jl")
include("satellite3d.jl")

end
