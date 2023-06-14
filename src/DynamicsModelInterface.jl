module DynamicsModelInterface

export add_dynamics_constraints!, add_dynamics_jacobians!,
       add_inequality_constraints!, add_inequality_jacobians!, next_x

function add_dynamics_constraints! end
function add_dynamics_jacobians! end
function add_inequality_constraints! end
function add_inequality_jacobians! end
function next_x end

end
