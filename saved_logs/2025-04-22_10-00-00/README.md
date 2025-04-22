# Experiment 2025-04-22_10-00-00

This introduced a more aggressive curriculum that made it learn faster, but also made it less precise.  
It also made changes to the configuration of the PPO algorithm

## Notes

- Changed entropy and initial noise in the PPO configuration to make it more likely to explore and learn to move the arm earlier
- Added a curriculum that would increase catchy_points, catchy_points_tanh and end_effector_orientation_tracking.
- The curriculum would kick in 8500 timesteps after less than 2% of the environments would terminate, giving it some time to learn to stand before learning to move the arm