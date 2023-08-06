### TO-DO

- [x] Implement planning ahead 
- [x] addobjectivegradients should return a row vector, not a col vector. And KKT conditions in inverse game should reflect this. 
- [x] Try slowing player's down so that they interact for longer
- [x] Work on stuff relevant to Anegi 
- [x] Re-introduce pre-scaling of objectives
- [x] Annealing of \mu in inverse game
- [x] Remove extra variables (data_states, init, etc...) from inverse hyperplane solver
- [x] Setup rho upper bound for inverse hyperplane solver
- [x] New cost model that doesn't have obstacle avoidance
  [x] Angle offsets should be relative to initial state. 
- [x] Regularize all the hyperplane parameters. Will probably see more active constraints. Make weight small enough so it doesn't dominate objective. 
- [ ] Talks that don't suck 
- [x] New initial conditions and solve forward game (with obstacle avoidance)
- [ ] Test whether inverse solver converges in one step if I initialize it with the solution. I suspect something weird is happening here.
- [ ] 
  
