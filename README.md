## Fish Models

The repository contains:

- `Code:` The directory contains notebooks for training and evaluation of the model. It acts 
as basic reference for training the model on the `Couzin Torous Data` and `Live Female Female Data`.
- `Debugging:` The directory contains notebooks for debugging `Preprocessing_and_Training.ipynb` and  `Evaluation_Phase_1.ipynb`.
- `Evaluation_Phase_1:` The directory contains notebooks for evaluating the model on the `Couzin Torous Data` and 
`Live Female Female Data`.
- `Resources:` The directory contains links for papers and links used for the project. 
- `Saved_Models:` The directory contains the models saved after training.
- `Training:` The directory contains notebooks that were used for training on the `Couzin Torous Data` and 
`Live Female Female Data`.

### Links/ References:

---

The links can be found below:
- The abstract of the project can be found in this [link.](https://docs.google.com/document/d/1mjzMosknZmGbL_73ENnhrcIAFOByGMEAJ0Lvd_snE5Y/edit?usp=sharing)  
- Mr. Maxeiner's thesis can be found in this [link.](https://www.mi.fu-berlin.de/inf/groups/ag-ki/Theses/Completed-theses/Master_Diploma-theses/2019/Maxeiner/MA-Maxeiner.pdf)
Most of the functions and the methodology for the starting part of the project can be found in the thesis.
- The data used for the project can be found in this [link.](https://zenodo.org/record/3457834#.XtK4XcBS_IU)


### Documentation for the Preprocessing and Training Notebook

---

The documentation of the functions that are currently being used in the notebook are given below:

- __`transform_coordinates(state):`__ The function is used to changes the coordinate system from one where the origin is the 
 top left of the fish tank and the point (100, 100) is the bottom right of the tank and (dx, dy) = (1, 0) is the fish 
 looking right and (dx, dy) = (0, 1) is looking down, to a normal system where the origin is the center of the tank and 
 (dx, dy) = (0, 1) points upwards.
- __`return_view_of_the_agents(socf, soof, max_dist, no_of_bins, total_angle):`__ The function returns an single array that 
represents how close the other agents (fish) to the fish under consideration and what radial bins do they belong to. 
The way the function works is that angle bisectors are created for each bin and then a vector is taken from the fish 
under consideration to the other fish. Then a dot product is taken between all the angle bisectors and the vector. The 
other fish would be placed in the bin which is the argmax of the dot product. Moreover, if the angle between the 
orientation of the fish under consideration and the other fish is greater than the field of view of the fish, then the 
array will not be updated. The value that will be placed in the bin is: 1 - (distance_between_the_fish) / max_distance
- __`return_view_of_the_walls(socf, wall_boundaries, max_dist, no_of_bins, total_angle):`__ The function returns a single 
array that represents how close the walls are to the fish under consideration and what radial bin do they belong to. 
The way the function works is that angle bisectors are created for each bin and then the intersection point is taken for
each angle bisector and the walls. The points that is within the wall limits are considered and then a dot product is 
taken between these points and the angle bisectors. The point whose dot product is maximum is considered to be in that 
particular bin (corresponding to the angle bisector). The value that is placed in that angle bisector is: 1 - 
distance_between_the_fish_and_the_point / max_dist
- __`return_speeds(current_state, previous_state):`__ The function returns the speed as well as the angular velocity of the 
fish derived from the positions and the pose.
- __`return_binned_speeds(current_state, previous_state, max_speed, no_speed_bins, max_ang_velo, ang_vel_bins):`__ The 
function returns the bin that the speed as well as the angular velocity of the fish belong to.
- __`data_to_model(type_ ,path, random, no_of_trajectories, max_dist, no_of_bins_voa, total_angle_voa, wall_boundaries, 
no_of_bins_vow, total_angle_vow, max_speed, speed_bins, max_ang_velo, ang_vel_bins):`__ The function processes the position
and pose data and returns `X: The binned input to the model`, `Y1: The speed output labels to the model`, `Y2: The angular
velocity output labels to the model`. This function uses the above three functions while preprocessing. 
- __` Moritz_Ladder_Network_Live_Female(shape_, output_bins_speed, output_bins_av):`__ The function returns the ladder network
as described in the paper 'Learning Recurrent Representations for Hierarchial Behaviour Modelling'. 
- __`Moritz_Couzin_Tourus(shape_, output_bins_speed, output_bins_av):`__ The function returns a simple LSTM model specifically 
for the `Couzin Tourus Dataset`

### Documentation for the Evaluation Phase 1 Notebook:

---

The documentation of the functions that are currently being used in the notebook are given below:

- __`import_data(path, random, no_of_trajectories, transform):`__ The function returns a particular trajectory of both 
position and pose data of one or more agents. 
- __`animate(AgentAnimation, data, title):`__ The function returns an in browser video of fish interactions  
from position and pose trajectory data.
__Credits for creating the AgentAnimation class: Christopher Mühl, Patrick Winterstein__
- __`return_binned_to_speeds(speed_binned, angular_velocity_binned, max_speed, no_speed_bins, max_ang_velo, ang_vel_bins):`__ 
The function returns the speed and the angular velocity from the binned speed and angular velocity outputs from the model
to actual speed and angular velocity values. A particular bin is selected according to the softmax probability output of 
the model and then a uniform sampling is done in that bin.
- __`velocity_to_current_position(speed, ang_vel, prev_position):`__ The function returns the position as well as the pose
of the fish given the speed, angular velocity and the previous position.
- __`simulate_trajectory(path, no_of_simulated_agents, no_of_control_agents, control_agents_data, time_steps, limit, 
max_dist, wall_boundaries, no_of_bins_voa, no_of_bins_vow, total_angle_vow, total_angle_voa, max_speed, no_speed_bins, 
max_ang_velo, ang_vel_bins):`__ The function returns the trajectory of the agents as predicted by the Network in the presence
of other control agents. The function first predicts the speed and position for a particular agent using the model and 
then that speed and velocity is used to predict the next position and pose. A similar process is followed for all other 
agents other than the control agents (whose trajectory is already known).
- __`follow(X,Y):`__ : The function returns the dot product of X’s velocity and the normalized direction from X’s 
position to Y’s position. 





