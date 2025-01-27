# Unscanted Kalman Filter applied to Richards equation

Goal is to predict deep infiltration.
Test problem: 1D 5m depth, Dirichlet on bottom, variable flux on the top.
Measurement: 5 moisture sensors at 0.1, 0.2, 0.4, 0.6, 1.0 m depth.
Test variables: 
- moisture at 0.5, 0.8, 2.0, 4.0 m depth.
- flux at 1m, 2m, 5m 

State variables:
- pressure head
- moisture values
- van Genuchten parameters (assuming constant over whole domain)
Resource[https://www.researchgate.net/figure/The-van-Genuchten-Mualem-model-parameters-for-simulated-soils_tbl1_329050595]- 
taking all sandy soil variants not complete sand
Initial state, independent:

theta_r=N[0.05, 0.9] @ 95
theta_s=N[0.35, 0.45] @ 95
n=N[1.5, 2.3] @ 95
alpha=N[0.02, 0.12] @ 95    [1/cm]
Ks=LN[0.01, 0.3] @ 95      [cm/min

Parameters for the synthetic measurements:
- van Gnuchten parameters for sandy loam:
- theta_r=0.065
- theta_s=0.41
- alpha=0.75 1/cm
- n=1.89
- Ks=0.0737 cm/min

TODO:
- add support for test velocity
- model parameters syntax:
  value (fixed no variance)
  [value1, value2, N95] (normal, 95 percent interval)
  [value1, value2, LN95] (lognormal, 95 percent interval)
- "synthetic_data_params" 
  updates the params dict, nonfixed will be fixed to 
  mean
- Plots for: test values, model parameters vs. prior and syntetic model data
  (seems partly done)
- synthetic rain procedure

Then we have to play with, R, Q, Kalman time step, and length of simulation
- How to get cov matrices for model and measurements (Q, R)?
- Adding a process noise to the synthetic data, assuming 
  a smooth gaussian external flux (roots, lateral flow, soil inhomogeneity).
  White noise in time.
  Could try to incorporate some of those into stat in future.

Questions:
- How Kalman time step affects convergence, how to choose it?
  How the simulation time compares to the Kalman processing time?
  Simpler Q choice is justifable for small time steps.