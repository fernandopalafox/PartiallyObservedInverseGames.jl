module JuMPUtils

import JuMP
export get_values, set_solver_attributes!, init_if_hasproperty!, isconverged, init_model_if_hasproperty!

get_values(; jump_vars...) = (; map(((k, v),) -> k => JuMP.value.(v), collect(jump_vars))...)

function set_solver_attributes!(opt_model; solver_attributes...)
    foreach(
        ((k, v),) -> JuMP.set_optimizer_attribute(opt_model, string(k), v),
        pairs(solver_attributes),
    )
end

function init_if_hasproperty!(v, init, sym; default = nothing)
    init_value = hasproperty(init, sym) ? getproperty(init, sym) : default
    if !isnothing(init_value)
        JuMP.set_start_value.(v, init_value)
    end
end

function init_model_if_hasproperty!(opt_model, init, sym)
    if hasproperty(init, sym)

        # Extract refs from model
        variable_refs_init = JuMP.all_variables(init.model)
        constraint_refs_init = JuMP.all_constraints(init.model; include_variable_in_set_constraints = true)
        nonlinear_constraint_refs_init = JuMP.all_nonlinear_constraints(init.model)

        variable_refs = JuMP.all_variables(opt_model)
        constraint_refs = JuMP.all_constraints(opt_model; include_variable_in_set_constraints = true)
        nonlinear_constraint_refs = JuMP.all_nonlinear_constraints(opt_model)

        # Create dicts mapping opt_model refs to init_model values
        dict_variables = Dict(zip(variable_refs, JuMP.value.(variable_refs_init)))
        dict_constraints = Dict(zip(constraint_refs, JuMP.value.(constraint_refs_init)))
        dict_duals = Dict(zip(constraint_refs, JuMP.dual.(constraint_refs_init)))
        dict_duals_nonlinear = Dict(zip(nonlinear_constraint_refs, JuMP.dual.(nonlinear_constraint_refs_init)))

        # Set values 
        JuMP.set_start_values(
            opt_model;
            variable_primal_start = x -> dict_variables[x],
            constraint_primal_start = x -> dict_constraints[x],
            constraint_dual_start = x -> dict_duals[x],
            # nonlinear_dual_start = x -> dict_duals_nonlinear[x], # ipopt does not support this
        )
    end
end

function isconverged(opt_model)
    JuMP.termination_status(opt_model) in (JuMP.MOI.LOCALLY_SOLVED, JuMP.MOI.OPTIMAL)
end

end
