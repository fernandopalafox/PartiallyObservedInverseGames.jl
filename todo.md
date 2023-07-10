### TO-DO

- [x] Implement planning ahead 
- [x] addobjectivegradients should return a row vector, not a col vector. And KKT conditions in inverse game should reflect this. 
- [x] Try slowing player's down so that they interact for longer
- [ ] Talks that don't suck 
- [ ] Parameters should be fed into constraints as a vector (not a matrix) and indexed into using the couple numbers (not the couple itself)
- [ ] Add observation noise 
- [ ] Work on stuff relevant to Anegi 
- [ ] Heuristic to initialize slack. Perhaps scale of the problem? Max distance between players? For a scale of 1 problem, initializing to 1 increased performance ENORMOUSLY (42s vs. 0.93s). Perhaps have it do with initialization. 
- [ ] Fix weight inference

