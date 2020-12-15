#!/usr/bin/env python
# coding: utf-8

# ### Importing the Libraries

# In[19]:


import numpy as np
import math
import glob
import h5py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from matplotlib import pyplot as plt
import pandas as pd
import json
print("Version:", tf.__version__)


#   ### Preprocessing Functions

# In[3]:


def transform_coordinates(state):
  # state has x , y, dx ,dy 
    state_ = np.zeros_like(state)
    state_[:, :, 0] = state[:, :, 0] - 50
    state_[:, :, 1] = - state[:, :, 1] + 50
    state_[:, :, 2] = state[:, :, 2]
    state_[:, :, 3] = - state[:, :, 3]
    return state_


# In[4]:


def return_view_of_the_agents(socf, soof, max_dist, no_of_bins, total_angle):
    # socf: state of fish under considerdation
    # soof: states of other fish (can be a batch)
    # total angle is in radian (angle whos bins are to be taken)
    # The logic for this method is that we look at at how close the other fish are to the the angle biscetors using a dot product. 
    orientation_socf = np.arctan2(socf[3], socf[2])
    angle_bisectors = np.array(range(int(math.floor(no_of_bins/2)), -int(math.floor(no_of_bins/2)) -1 , -1)) * total_angle / no_of_bins + orientation_socf
    shape = np.shape(soof)
    socf_vector_cos = np.cos(angle_bisectors)
    socf_vector_sin = np.sin(angle_bisectors)
    view_of_the_agents = np.zeros(no_of_bins)
    for i in range(shape[0]):
        soof_X = soof[i, 0] - socf[0]
        soof_Y = soof[i,1] - socf[1]
        soof_vector_cos = np.cos(np.arctan2(soof_Y, soof_X))
        soof_vector_sin = np.sin(np.arctan2(soof_Y, soof_X))
        angle = np.arccos(np.clip(soof_vector_cos * socf_vector_cos[int(math.floor(no_of_bins/2))] + soof_vector_sin * socf_vector_sin[int(math.floor(no_of_bins/2))], -1, 1))
        if angle > - total_angle/2 and angle < total_angle/2:
            max_bin = np.argmax(socf_vector_cos * soof_vector_cos + socf_vector_sin * soof_vector_sin)
            distance = np.sqrt(soof_X ** 2 + soof_Y ** 2)
            if view_of_the_agents[max_bin] < 1 - distance / max_dist :
                view_of_the_agents[max_bin] =  1 - distance / max_dist
    return view_of_the_agents


# In[5]:


def return_view_of_the_walls(socf, wall_boundaries, max_dist, no_of_bins, total_angle):
  # wall_boundaries contain +x, -x, +y and -y of the walls (Basically equations of the lines that describe the walls)
    orientation_socf = np.arctan2(socf[3], socf[2])
    angles = np.array(range(int(math.floor(no_of_bins/2)), -int(math.floor(no_of_bins/2)) -1 , -1)) * total_angle / no_of_bins + orientation_socf
    slopes = np.tan(angles)
    socf_cos = np.cos(angles)
    socf_sin = np.sin(angles)
#     print(slopes)
#     print(angles)
    view_of_walls = np.zeros(no_of_bins)
    for i in range(len(slopes)):
        for j in range(len(wall_boundaries)):
            if (j < 2):
                if (slopes[i] > np.tan(0.99 * np.pi/2)) or (slopes[i] < np.tan(-0.99 * np.pi/2)) :
                # Condition to not evaluate the lines paralell to lines +x and - x
                    continue
                else: 
                    y = slopes[i] * (wall_boundaries[j]*1.0 - socf[0]) + socf[1]
                    angle = np.arctan2((y - socf[1]), (wall_boundaries[j]*1.0 - socf[0]))
                    vector_cos = np.cos(angle)
                    vector_sin = np.sin(angle)
                    index = np.argmax(vector_cos * socf_cos + vector_sin * socf_sin)
#                     print(i, index, wall_boundaries[j], y)
                    if (y < wall_boundaries[2]) and (y > wall_boundaries[3]) and (index == i):
#                         print('Selected:', y)
                        view_of_walls[i] = 1 - np.sqrt((socf[0] - wall_boundaries[j])**2 + (socf[1] - y)**2) / max_dist
                        break
            if (j>=2 and j < 4):
                if slopes[i] < np.tan(0.001 * np.pi/2) and slopes[i] > np.tan(-0.001 * np.pi/2) :
                    # Condition to not evaluate the lines parallel or almost parallel to + y and - y
                    continue
                else: 
                    x = (wall_boundaries[j] - socf[1]) / slopes[i] + socf[0]
                    angle = np.arctan2((wall_boundaries[j] - socf[1]), (x - socf[0]))
                    vector_cos = np.cos(angle)
                    vector_sin = np.sin(angle)
                    index = np.argmax(vector_cos * socf_cos + vector_sin * socf_sin)
#                     print(i, index, x, wall_boundaries[j])
                    if  (x < wall_boundaries[0]) and (x > wall_boundaries[1]) and (index == i):
#                         print('Selected', x)
                        view_of_walls[i] = 1 - np.sqrt((socf[0] - x)**2 + (socf[1] - wall_boundaries[j])**2) / max_dist
                        break

    return view_of_walls


# In[6]:


def return_binned_velocities(current_state, previous_state, max_linear_velo, lin_vel_bins, max_ang_velo, ang_vel_bins):
    vel_lin_bins = np.array(range(0, lin_vel_bins + 1 , 1)) * max_linear_velo / lin_vel_bins - max_linear_velo / 2 
    vel_ang_bins = np.array(range(0, ang_vel_bins + 1 , 1)) * max_ang_velo / ang_vel_bins - max_ang_velo / 2 

    vel_x = current_state[0] - previous_state[0]
    vel_y = current_state[1] - previous_state[1]
    vel_ang = np.arctan2(current_state[3], current_state[2]) - np.arctan2(previous_state[3], previous_state[2])
    vel_x_binned = np.zeros(lin_vel_bins)
    vel_y_binned = np.zeros(lin_vel_bins)
    vel_ang_binned = np.zeros(ang_vel_bins)

    for i in range(lin_vel_bins):
        if vel_x > vel_lin_bins[i] and vel_x < vel_lin_bins[i + 1]:
            vel_x_binned[i] = 1
        if vel_y > vel_lin_bins[i] and vel_y < vel_lin_bins[i + 1]:
            vel_y_binned[i] = 1

        for i in range(ang_vel_bins):
            if vel_ang > vel_ang_bins[i] and vel_ang < vel_ang_bins[i + 1]:
                vel_ang_binned[i] = 1

    return vel_x_binned, vel_y_binned, vel_ang_binned


# In[7]:


def return_speeds(current_state, previous_state, max_speed, max_ang_velo):
    speed = np.sqrt((current_state[0] - previous_state[0])**2 + (current_state[1] - previous_state[1])**2)
    speed = min(speed, max_speed)
    angular_velocity = np.arctan2(current_state[3], current_state[2]) - np.arctan2(previous_state[3], previous_state[2])
    if angular_velocity > np.pi:
        angular_velocity = angular_velocity - 2 * np.pi
    if angular_velocity < -np.pi:
        angular_velocity = angular_velocity + 2 * np.pi
    
    angular_velocity = min(angular_velocity, max_ang_velo) * (angular_velocity >= 0) + max(angular_velocity, -max_ang_velo) * (angular_velocity < 0)
    return speed, angular_velocity


# In[8]:


def return_binned_speeds(current_state, previous_state, max_speed, no_speed_bins, max_ang_velo, ang_vel_bins):
    speed = np.sqrt((current_state[0] - previous_state[0])**2 + (current_state[1] - previous_state[1])**2)
    speed = min(speed, max_speed)
    speed_bins = np.array(range(-no_speed_bins, no_speed_bins + 1 , 2)) * max_speed / (no_speed_bins - 1) 
    vel_ang_bins = np.array(range(-ang_vel_bins, ang_vel_bins + 1, 2)) * max_ang_velo / (ang_vel_bins - 1) 
    vel_ang = np.arctan2(current_state[3], current_state[2]) - np.arctan2(previous_state[3], previous_state[2])
    
    if vel_ang > np.pi:
        vel_ang = vel_ang - 2 * np.pi
    if vel_ang < -np.pi:
        vel_ang = vel_ang + 2 * np.pi
    
    vel_ang = min(vel_ang, max_ang_velo) * (vel_ang >= 0) + max(vel_ang, -max_ang_velo) * (vel_ang < 0)
    vel_ang_binned = np.zeros(ang_vel_bins)
    speed_binned = np.zeros(no_speed_bins)
    
#     print(speed_bins)
#     print(vel_ang_bins)
    for i in range(no_speed_bins):
        if speed > speed_bins[i] and speed < speed_bins[i + 1]:
            speed_binned[i] = 1
            
    for i in range(ang_vel_bins):
        if vel_ang > vel_ang_bins[i] and vel_ang < vel_ang_bins[i + 1]:
            vel_ang_binned[i] = 1    
    
    return speed_binned, vel_ang_binned


# In[9]:


def data_to_model(type_ ,path, random, no_of_trajectories, max_dist, no_of_bins_voa, total_angle_voa, wall_boundaries, no_of_bins_vow, total_angle_vow, max_speed, speed_bins, max_ang_velo, ang_vel_bins, include_y_labels):
    
    if random == 1:
        random_selection = np.random.randint(0,np.shape(path)[0], no_of_trajectories)
    else:
        random_selection = np.array(range(len(path)))
    
    agents = []
    actual_no_of_agents = 0
    
    if include_y_labels == 0:
        for j in random_selection:
            agents.append([])

        for j, p in zip(random_selection, range(len(random_selection))):

            hf = h5py.File(path[j], 'r')
            keys = list(hf.keys())

            for k in range(len(keys)):

                agents[p].append(np.asarray(hf.get(keys[k])))
                actual_no_of_agents += 1

            agents[p] = transform_coordinates(np.asarray(agents[p]))
    else:
        p = 0
        for j in random_selection:
            hf = h5py.File(path[j], 'r')
            keys = list(hf.keys())
            if len(keys) == 2:
                agents.append([])
                
                for k in range(len(keys)):
                    agents[p].append(np.asarray(hf.get(keys[k])))
                agents[p] = transform_coordinates(np.asarray(agents[p]))
                p = p + 1
                    
    
    agents_voa_vow_sp_av = []
    agents_sp_binned = []
    agents_va_binned = []
    agents_y_label = []
    
    for j in range(len(agents)):
        
        shape = np.shape(agents[j])
        agents_voa_vow_sp_av.append(np.zeros((shape[0], shape[1], no_of_bins_voa + no_of_bins_vow + 2)))
        agents_sp_binned.append(np.zeros((shape[0], shape[1]-1 , speed_bins)))
        agents_va_binned.append(np.zeros((shape[0], shape[1]-1, ang_vel_bins)))
        agents_y_label.append(np.zeros((shape[0], shape[1]-1, 1)))
    
    
    for j in range(len(agents)):
            
            shape = np.shape(agents[j])
            
            for k in range(shape[1]):
                
                for l in range(shape[0]):
                    if shape[0] > 0:
                        other_agents = np.append(agents[j][0:l,k,:], agents[j][l+1:, k, :], axis = 0)
                        agents_voa_vow_sp_av[j][l,k,0:no_of_bins_voa] = return_view_of_the_agents(agents[j][l,k,:], other_agents, max_dist, no_of_bins_voa, total_angle_voa)
                        
                        
                    agents_voa_vow_sp_av[j][l,k,no_of_bins_voa : no_of_bins_voa + no_of_bins_vow] = return_view_of_the_walls(agents[j][l,k,:], wall_boundaries, max_dist, no_of_bins_vow, total_angle_vow)
                    
                    if k > 0:
                        speed, ang_vel = return_speeds(agents[j][l, k, :], agents[j][l,k-1,:], max_speed, max_ang_velo)
                        agents_voa_vow_sp_av[j][l,k, no_of_bins_voa + no_of_bins_vow], agents_voa_vow_sp_av[j][l,k, no_of_bins_voa + no_of_bins_vow + 1] = speed, ang_vel
                        agents_sp_binned[j][l,k-1,:], agents_va_binned[j][l,k-1,:] = return_binned_speeds(agents[j][l, k, :], agents[j][l,k-1,:], max_speed, speed_bins, max_ang_velo, ang_vel_bins)
                        
                        if shape[0] == 2:
                            other_agents = np.append(agents[j][0:l,k-1,:], agents[j][l+1:, k-1, :], axis = 0)
                            cos_theta = np.cos(ang_vel + np.arctan2(agents[j][l,k-1,3], agents[j][l,k-1,2]))
                            sin_theta = np.sin(ang_vel + np.arctan2(agents[j][l,k-1,3], agents[j][l,k-1,2]))
                            cos_agent = other_agents[0,0] - agents[j][l,k-1,0]
                            sin_agent = other_agents[0,1] - agents[j][l,k-1,1]
                            agents_y_label[j][l, k-1, :] = speed * (cos_theta * cos_agent + sin_theta * sin_agent)/ (max_speed * max_dist) 
                
    
    
    for j in range(len(agents) - 1):
        agents_voa_vow_sp_av[0] = np.append(agents_voa_vow_sp_av[0],agents_voa_vow_sp_av[j+1], axis = 0)
        agents_sp_binned[0] = np.append(agents_sp_binned[0], agents_sp_binned[j+1], axis = 0)
        agents_va_binned[0] = np.append(agents_va_binned[0], agents_va_binned[j+1], axis = 0)
        agents_y_label[0] = np.append(agents_y_label[0], agents_y_label[j+1], axis = 0)
    
    X = agents_voa_vow_sp_av[0][:,0:np.shape(agents_voa_vow_sp_av[0])[1] - 1]
    Y1 = agents_sp_binned[0]
    Y2 = agents_va_binned[0]
    Y3 = agents_y_label[0]
    
    if type_ == 1:
        X = np.reshape(X, (np.shape(X)[0], np.shape(X)[1], 1, 2, no_of_bins_voa + 1))
    
    return X, Y1, Y2, Y3


# In[10]:


def data_to_model_csv_rf(type_ , path_rf, random, no_of_trajectories_rf, time_steps, max_dist, no_of_bins_voa, total_angle_voa, wall_boundaries, no_of_bins_vow, total_angle_vow, max_speed, speed_bins, max_ang_velo, ang_vel_bins):

    agents_rf = []

    if random == 1:
        random_selection_rf = np.random.randint(0, np.shape(path_rf)[0], no_of_trajectories_csv)
    else:
        random_selection_rf = np.array(range(len(path_rf)))

    for j in random_selection_rf:
        agents_rf.append([])

    for j, p in zip(random_selection_rf, range(len(random_selection_rf))):
        rf_data = np.asarray(pd.read_csv(path_rf[j]))
        start_list = 0 
        for l, m in (rf_data[:, 15], range(len(rf_data[:, 15]))):
            if l != 'Milling':
                start_list = m
                break
        rf_proc = np.zeros((2, np.shape(rf_data)[0] - start_list, 4))
        rf_proc[0, :, 0:2] = rf_data[start_list:, 11:13].astype(np.float) 
        rf_proc[1, :, 0:2] = rf_data[start_list:, 5:7].astype(np.float) 
        rf_proc[1, :, 2], rf_proc[1, :, 3] = np.cos(rf_data[start_list:, 8].astype(np.float)), np.sin(rf_data[start_list:, 8].astype(np.float))
        agents_rf[p] = rf_proc

    agents_voa_vow_sp_av_rf = []
    agents_sp_binned_rf = []
    agents_va_binned_rf = []
    agents_y_label = []

    for j in range(len(agents_rf)):
        shape = np.shape(agents_rf[j])
        agents_voa_vow_sp_av_rf.append(np.zeros((1, shape[1], no_of_bins_voa + no_of_bins_vow + 2)))
        agents_sp_binned_rf.append(np.zeros((1, shape[1]-1 , speed_bins)))
        agents_va_binned_rf.append(np.zeros((1, shape[1]-1, ang_vel_bins)))
        agents_y_label.append(np.zeros((1, shape[1]-1, 1)))
        
    for j in range(len(agents_rf)):

            shape = np.shape(agents_rf[j])

            for k in range(shape[1]):

                l = 0
                if shape[0] > 0:
                    other_agents = np.append(agents_rf[j][0:l,k,:], agents_rf[j][l+1:, k, :], axis = 0)
                    agents_voa_vow_sp_av_rf[j][l,k,0:no_of_bins_voa] = return_view_of_the_agents(agents_rf[j][l,k,:], other_agents, max_dist, no_of_bins_voa, total_angle_voa)

                agents_voa_vow_sp_av_rf[j][l,k,no_of_bins_voa : no_of_bins_voa + no_of_bins_vow] = return_view_of_the_walls(agents_rf[j][l,k,:], wall_boundaries, max_dist, no_of_bins_vow, total_angle_vow)

                if k > 0:
                    speed, ang_vel = return_speeds(agents_rf[j][l, k, :], agents_rf[j][l,k-1,:], max_speed, max_ang_velo)
                    agents_voa_vow_sp_av_rf[j][l,k, no_of_bins_voa + no_of_bins_vow], agents_voa_vow_sp_av_rf[j][l,k, no_of_bins_voa + no_of_bins_vow + 1] = speed, ang_vel
                    agents_sp_binned_rf[j][l,k-1,:], agents_va_binned_rf[j][l,k-1,:] = return_binned_speeds(agents_rf[j][l, k, :], agents_rf[j][l,k-1,:], max_speed, speed_bins, max_ang_velo, ang_vel_bins)

                    if shape[0] == 2:
                        other_agents = np.append(agents_rf[j][0:l,k-1,:], agents_rf[j][l+1:, k-1, :], axis = 0)
                        cos_theta = np.cos(ang_vel + np.arctan2(agents_rf[j][l,k-1,3], agents_rf[j][l,k-1,2]))
                        sin_theta = np.sin(ang_vel + np.arctan2(agents_rf[j][l,k-1,3], agents_rf[j][l,k-1,2]))
                        cos_agent = other_agents[0,0] - agents_rf[j][l,k-1,0]
                        sin_agent = other_agents[0,1] - agents_rf[j][l,k-1,1]
                        agents_y_label[j][l, k-1, :] = speed * (cos_theta * cos_agent + sin_theta * sin_agent)/ (max_speed * max_dist) 
                            
    for j in range(len(agents_rf)):
        print(np.shape(agents_voa_vow_sp_av_rf[j]), np.shape(agents_sp_binned_rf[j]), np.shape(agents_va_binned_rf[j]))


    for j in range(len(agents_rf)):
        shape = np.shape(agents_voa_vow_sp_av_rf[j])
        agents_voa_vow_sp_av_rf[j] = np.reshape(agents_voa_vow_sp_av_rf[j][:, 0: (shape[1] - shape[1]%time_steps), :], (shape[0] * int((shape[1] - shape[1]%time_steps)/time_steps), time_steps, shape[2]))

        shape = np.shape(agents_sp_binned_rf[j])
        agents_sp_binned_rf[j] = np.reshape(agents_sp_binned_rf[j][:, 0: (shape[1] - shape[1]%time_steps), :], (shape[0] * int((shape[1] - shape[1]%time_steps)/time_steps), time_steps, shape[2]))

        shape = np.shape(agents_va_binned_rf[j])
        agents_va_binned_rf[j] = np.reshape(agents_va_binned_rf[j][:, 0: (shape[1] - shape[1]%time_steps), :], (shape[0] * int((shape[1] - shape[1]%time_steps)/time_steps), time_steps, shape[2]))
        
        shape = np.shape(agents_y_label[j])
        agents_y_label[j] = np.reshape(agents_y_label[j][:, 0: (shape[1] - shape[1]%time_steps), :], (shape[0] * int((shape[1] - shape[1]%time_steps)/time_steps), time_steps, shape[2]))

    for j in range(len(agents_rf) - 1):
        agents_voa_vow_sp_av_rf[0] = np.append(agents_voa_vow_sp_av_rf[0],agents_voa_vow_sp_av_rf[j+1], axis = 0)
        agents_sp_binned_rf[0] = np.append(agents_sp_binned_rf[0], agents_sp_binned_rf[j+1], axis = 0)
        agents_va_binned_rf[0] = np.append(agents_va_binned_rf[0], agents_va_binned_rf[j+1], axis = 0)
        agents_y_label[0] = np.append(agents_y_label[0], agents_y_label[j+1], axis = 0)
        
    X = agents_voa_vow_sp_av_rf[0]
    Y1 = agents_sp_binned_rf[0]
    Y2 = agents_va_binned_rf[0]
    Y3 = agents_y_label[0]
    
    if type_ == 1:
        X = np.reshape(X, (np.shape(X)[0], np.shape(X)[1], 1, 2, no_of_bins_voa + 1))

    return X, Y1, Y2, Y3

# In[11]:


def data_to_model_csv_gg(type_ ,path, random, no_of_trajectories, max_dist, no_of_bins_voa, total_angle_voa, wall_boundaries, no_of_bins_vow, total_angle_vow, max_speed, speed_bins, max_ang_velo, ang_vel_bins):
    
    if random == 1:
        random_selection = np.random.randint(0,np.shape(path)[0], no_of_trajectories)
    else:
        random_selection = np.array(range(len(path)))

    agents = []
    
    for j, p in zip(random_selection, range(len(random_selection))):

        data = np.asarray(pd.read_csv(path[random_selection[p]]))
        data_proc = np.zeros((int(np.shape(data)[1] / 4), np.shape(data)[0], 4))

        for k in range(np.shape(data_proc)[0]):
            data_proc[k, :, :] = data[:, (4*k):(4*(k+1))]

        agents.append(data_proc)
    
    agents_voa_vow_sp_av = []
    agents_sp_binned = []
    agents_va_binned = []
    agents_y_label = []
    
    for j in range(len(agents)):
        
        shape = np.shape(agents[j])
        agents_voa_vow_sp_av.append(np.zeros((shape[0], shape[1], no_of_bins_voa + no_of_bins_vow + 2)))
        agents_sp_binned.append(np.zeros((shape[0], shape[1]-1 , speed_bins)))
        agents_va_binned.append(np.zeros((shape[0], shape[1]-1, ang_vel_bins)))
        agents_y_label.append(np.zeros((shape[0], shape[1]-1, 1)))
    
    
    for j in range(len(agents)):
            
            shape = np.shape(agents[j])
            
            for k in range(shape[1]):
                
                for l in range(shape[0]):
                    if shape[0] > 0:
                        other_agents = np.append(agents[j][0:l,k,:], agents[j][l+1:, k, :], axis = 0)
                        agents_voa_vow_sp_av[j][l,k,0:no_of_bins_voa] = return_view_of_the_agents(agents[j][l,k,:], other_agents, max_dist, no_of_bins_voa, total_angle_voa)
                        
                    agents_voa_vow_sp_av[j][l,k,no_of_bins_voa : no_of_bins_voa + no_of_bins_vow] = return_view_of_the_walls(agents[j][l,k,:], wall_boundaries, max_dist, no_of_bins_vow, total_angle_vow)
                    
                    if k > 0:
                        speed, ang_vel = return_speeds(agents[j][l, k, :], agents[j][l,k-1,:], max_speed, max_ang_velo)
                        agents_voa_vow_sp_av[j][l,k, no_of_bins_voa + no_of_bins_vow], agents_voa_vow_sp_av[j][l,k, no_of_bins_voa + no_of_bins_vow + 1] = speed, ang_vel
                        agents_sp_binned[j][l,k-1,:], agents_va_binned[j][l,k-1,:] = return_binned_speeds(agents[j][l, k, :], agents[j][l,k-1,:], max_speed, speed_bins, max_ang_velo, ang_vel_bins)
                        
                        if shape[0] == 2:
                            other_agents = np.append(agents[j][0:l,k-1,:], agents[j][l+1:, k-1, :], axis = 0)
                            cos_theta = np.cos(ang_vel + np.arctan2(agents[j][l,k-1,3], agents[j][l,k-1,2]))
                            sin_theta = np.sin(ang_vel + np.arctan2(agents[j][l,k-1,3], agents[j][l,k-1,2]))
                            cos_agent = other_agents[0,0] - agents[j][l,k-1,0]
                            sin_agent = other_agents[0,1] - agents[j][l,k-1,1]
                            agents_y_label[j][l, k-1, :] = speed * (cos_theta * cos_agent + sin_theta * sin_agent)/ (max_speed * max_dist) 
                
    
    
    for j in range(len(agents) - 1):
        agents_voa_vow_sp_av[0] = np.append(agents_voa_vow_sp_av[0],agents_voa_vow_sp_av[j+1], axis = 0)
        agents_sp_binned[0] = np.append(agents_sp_binned[0], agents_sp_binned[j+1], axis = 0)
        agents_va_binned[0] = np.append(agents_va_binned[0], agents_va_binned[j+1], axis = 0)
        agents_y_label[0] = np.append(agents_y_label[0], agents_y_label[j+1], axis = 0)
    
    X = agents_voa_vow_sp_av[0][:,0:np.shape(agents_voa_vow_sp_av[0])[1] - 1]
    Y1 = agents_sp_binned[0]
    Y2 = agents_va_binned[0]
    Y3 = agents_y_label[0]
    
    if type_ == 1:
        X = np.reshape(X, (np.shape(X)[0], np.shape(X)[1], 1, 2, no_of_bins_voa + 1))
    
    return X, Y1, Y2, Y3

# In[12]:


def import_data(path, random, no_of_trajectories, transform):
    
    if random == 1:
        random_selection = np.random.randint(0,np.shape(path)[0], no_of_trajectories)
    else:
        random_selection = np.array(range(len(path)))
    
    agents = []
    actual_no_of_agents = 0
    
    for j in random_selection:
        agents.append([])
        
    for j, p in zip(random_selection, range(len(random_selection))):
        
        hf = h5py.File(path[j], 'r')
        keys = list(hf.keys())
        
        for k in range(len(keys)):
            
            agents[p].append(np.asarray(hf.get(keys[k])))
            actual_no_of_agents += 1
        
        if transform == 1:
            agents[p] = transform_coordinates(np.asarray(agents[p]))
        else:
            agents[p] = np.asarray(agents[p])
            
    for j in range(len(agents) - 1):
        agents[0] = np.append(agents[0], agents[j+1], axis = 0)
        
    return agents[0]


# In[13]:


def import_data_csv_rf(path_rf, random, no_of_trajectories_csv):
    agents_rf = []

    if random == 1:
        random_selection_rf = np.random.randint(0, np.shape(path_rf)[0], no_of_trajectories_csv)
    else:
        random_selection_rf = np.array(range(len(path_rf)))

    for j in random_selection_rf:
        agents_rf.append([])

    for j, p in zip(random_selection_rf, range(len(random_selection_rf))):
        rf_data = np.asarray(pd.read_csv(path_rf[j]))
        start_list = 0 
        for l, m in (rf_data[:, 15], range(len(rf_data[:, 15]))):
            if l != 'Milling':
                start_list = m
                break
        rf_proc = np.zeros((2, np.shape(rf_data)[0] - start_list, 4))
        rf_proc[0, start_list:, 0:2] = rf_data[start_list:, 11:13].astype(np.float) 
        rf_proc[1, start_list:, 0:2] = rf_data[start_list:, 5:7].astype(np.float) 
        rf_proc[1, start_list:, 2], rf_proc[1, start_list:, 3] = np.cos(rf_data[start_list:, 8].astype(np.float)), np.sin(rf_data[start_list:, 8].astype(np.float))
        agents_rf[p] = rf_proc
    
    return agents_rf


# In[14]:


def import_data_csv_gg(path, random, no_of_trajectories):
    
    if random == 1:
        random_selection = np.random.randint(0,np.shape(path)[0], no_of_trajectories)
    else:
        random_selection = np.array(range(len(path)))

    agents = []
    
    for j, p in zip(random_selection, range(len(random_selection))):

        data = np.asarray(pd.read_csv(train_gg[random_selection[p]]))
        data_proc = np.zeros((int(np.shape(data)[1] / 4), np.shape(data)[0], 4))

        for k in range(np.shape(data_proc)[0]):
            data_proc[k, :, :] = data[:, (4*k):(4*(k+1))]

        agents.append(data_proc)
        
    for j in range(no_of_trajectories - 1):
        agents[0] = np.append(agents[0], agents[j+1], axis = 0)
        
    return agents[0]


# ### Network

# In[15]:

def Ladder_Net(shape_, shape_2, output_bins_speed, output_bins_av):
    
    input_ = layers.Input(shape = shape_)
    
    enc1 = layers.LSTM(200, return_sequences=True)(input_)
    norm_enc1 = layers.LayerNormalization()(enc1)
    enc2 = layers.LSTM(200, return_sequences= True)(norm_enc1)
    norm_enc2 = layers.LayerNormalization()(enc2)
    enc3 = layers.LSTM(200, return_sequences= True)(norm_enc2)
    norm_enc3 = layers.LayerNormalization()(enc3)
    
    dec3 = layers.LSTM(200, return_sequences=True)(norm_enc3)
    norm_dec3 = layers.LayerNormalization()(dec3)
    comb_nenc2_ndec3 = layers.concatenate([norm_enc2, norm_dec3])
    dec2 = layers.LSTM(200, return_sequences=True)(comb_nenc2_ndec3)
    norm_dec2 = layers.LayerNormalization()(dec2)
    comb_nenc1_ndec2 = layers.concatenate([norm_enc1, norm_dec2])
    dec1 = layers.LSTM(200, return_sequences=True)(comb_nenc2_ndec3)
    norm_dec1 = layers.LayerNormalization()(dec1)
    
    speed_binned = layers.TimeDistributed(layers.Dense(output_bins_speed, activation = 'softmax', name = 'speed_binned'))(norm_dec1)
    angular_velocity_binned = layers.TimeDistributed(layers.Dense(output_bins_av, activation = 'softmax', name = 'angular_velocity_binned'))(norm_dec1)
    opt = optimizers.Adam(lr=0.001)
    model = models.Model(inputs = input_, outputs = [speed_binned, angular_velocity_binned], name = 'ladder_net')
    model.compile(loss = [losses.categorical_crossentropy, losses.categorical_crossentropy], optimizer= opt, metrics = ['categorical_accuracy'])

    return model

def Ladder_Net_With_Y_Labels(shape_, shape_2, output_bins_speed, output_bins_av):
    
    input_ = layers.Input(shape = shape_)
    
    enc1 = layers.LSTM(200, return_sequences=True)(input_)
    norm_enc1 = layers.LayerNormalization()(enc1)
    enc2 = layers.LSTM(200, return_sequences= True)(norm_enc1)
    norm_enc2 = layers.LayerNormalization()(enc2)
    enc3 = layers.LSTM(200, return_sequences= True)(norm_enc2)
    norm_enc3 = layers.LayerNormalization()(enc3)
    
    dec3 = layers.LSTM(200, return_sequences=True)(norm_enc3)
    norm_dec3 = layers.LayerNormalization()(dec3)
    comb_nenc2_ndec3 = layers.concatenate([norm_enc2, norm_dec3])
    dec2 = layers.LSTM(200, return_sequences=True)(comb_nenc2_ndec3)
    norm_dec2 = layers.LayerNormalization()(dec2)
    comb_nenc1_ndec2 = layers.concatenate([norm_enc1, norm_dec2])
    dec1 = layers.LSTM(200, return_sequences=True)(comb_nenc2_ndec3)
    norm_dec1 = layers.LayerNormalization()(dec1)
    
    y_label_output = layers.TimeDistributed(layers.Dense(1, activation = 'tanh', name = 'y_label'))(norm_enc3)
    speed_binned = layers.TimeDistributed(layers.Dense(output_bins_speed, activation = 'softmax', name = 'speed_binned'))(norm_dec1)
    angular_velocity_binned = layers.TimeDistributed(layers.Dense(output_bins_av, activation = 'softmax', name = 'angular_velocity_binned'))(norm_dec1)
    opt = optimizers.Adam(lr=0.001)
    
    
    model = models.Model(inputs = input_, outputs = [speed_binned, angular_velocity_binned, y_label_output], name = 'ladder_net_with_y_labels')
    model.compile(loss = [losses.categorical_crossentropy, losses.categorical_crossentropy, losses.MSE], optimizer= opt, metrics = ['categorical_accuracy', 'accuracy'])

    return model


def Moritz_Ladder_Network_Live_Female(shape_, output_bins_speed, output_bins_av):

    input_ = layers.Input(shape = shape_)
    convLSTM_1 = layers.ConvLSTM2D(filters= 8, kernel_size=(9, 9), data_format='channels_first', padding='same', return_sequences=True)(input_)
    norm_1 = layers.LayerNormalization()(convLSTM_1)
    flatten = layers.TimeDistributed(layers.Flatten())(norm_1)
    norm_2 = layers.LayerNormalization()(flatten)
    h1 = layers.LSTM(100, return_sequences= True)(norm_2)
    norm_3 = layers.LayerNormalization()(h1)
    h2 = layers.LSTM(100, return_sequences= True)(norm_3)
    norm_4 = layers.LayerNormalization()(h2)
    norm_5 = layers.concatenate([norm_3, norm_4])
    speed_binned = layers.TimeDistributed(layers.Dense(output_bins_speed, activation = 'softmax', name = 'speed_binned'))(norm_5)
    angular_velocity_binned = layers.TimeDistributed(layers.Dense(output_bins_av, activation = 'softmax', name = 'angular_velocity_binned'))(norm_5)
    opt = optimizers.Adam(lr=0.0002)
    model = models.Model(inputs = input_, outputs = [speed_binned, angular_velocity_binned], name = 'moritzs_ladder_network')
    model.compile(loss = [losses.categorical_crossentropy, losses.categorical_crossentropy], optimizer= opt, metrics = ['categorical_accuracy'])

    return model


# In[16]:


def Moritz_Couzin_Tourus(shape_, output_bins_speed, output_bins_av):
    
    input_ = layers.Input(shape= shape_)
    h1 = layers.LSTM(10, return_sequences=True)(input_)
    norm_1 = layers.LayerNormalization()(h1)
    speed_binned = layers.TimeDistributed(layers.Dense(output_bins_speed, activation = 'softmax', name = 'speed_binned'))(norm_1)
    angular_velocity_binned = layers.TimeDistributed(layers.Dense(output_bins_av, activation = 'softmax', name = 'angular_velocity_binned'))(norm_1)
    model = models.Model(inputs = input_, outputs = [speed_binned, angular_velocity_binned], name = 'moritz_couzin_tourus')
    opt = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss = [losses.categorical_crossentropy, losses.categorical_crossentropy], optimizer= opt, metrics = ['categorical_accuracy'])

    return model


# ### Training

# In[18]:


data_type = int(input("Enter 0 for Couzin Data, 1 for Adaptive Guppy Gym Data and 2 for Live Fish Fish:"))
training_data_type = int(input("Enter 0 for a normal LSTM Network and 1 for the Ladder Network With Convnets and 2 for a Ladder Network:"))
choose_bin_hyperparameters = int(input("Enter 0 for Couzin Torus Parameters and 1 for fish-fish and robot-fish parameters:"))
include_y_labels = int(input("Enter 1 for using the Y labels and 0 otherwise:"))

train_path_couzin_torus = glob.glob("/home/mi/chanakya/learn_fish_models/Datasets/datasets/couzin_torus/train/*.hdf5")
test_path_couzin_torus = glob.glob("/home/mi/chanakya/learn_fish_models/Datasets/datasets/couzin_torus/test/*.hdf5")
validation_path_couzin_torus = glob.glob("/home/mi/chanakya/learn_fish_models/Datasets/datasets/couzin_torus/validation/*.hdf5")

train_path_live_female_female = glob.glob("/home/mi/chanakya/learn_fish_models/Datasets/datasets/live_female_female/train/*.hdf5")
test_path_live_female_female = glob.glob("/home/mi/chanakya/learn_fish_models/Datasets/datasets/live_female_female/test/*.hdf5")
validation_path_live_female_female = glob.glob("/home/mi/chanakya/learn_fish_models/Datasets/datasets/live_female_female/validation/*.hdf5")

train_path_guppy_gym = glob.glob('/home/mi/chanakya/gym-guppy/data/train/*.csv')
test_path_guppy_gym = glob.glob('/home/mi/chanakya/gym-guppy/data/test/*.csv')
validation_path_guppy_gym = glob.glob('/home/mi/chanakya/gym-guppy/data/validate/*.csv')

include_data_rf = int(input("Enter 1 for using the Robot Fish Interaction Data:"))

if include_data_rf == 1:
    path_csv_rf = glob.glob("/home/mi/chanakya/input_data/2018/P_Trials/Adaptive/*.csv")
    train_path_rf = path_csv_rf[0:17]
    test_path_rf = path_csv_rf[17:20]
    validation_path_rf = path_csv_rf[20:23]

if data_type == 0:
    train_path = train_path_couzin_torus
    test_path = test_path_couzin_torus
    validation_path = validation_path_couzin_torus
elif data_type == 1:
    train_path = train_path_guppy_gym
    test_path = test_path_guppy_gym
    validation_path = validation_path_guppy_gym
elif data_type == 2:
    train_path = train_path_live_female_female
    test_path = test_path_live_female_female
    validation_path = validation_path_live_female_female

for i in range(len(train_path)):
    train_path[i] = train_path[i].replace("\\", "/")
for i in range(len(test_path)):
    test_path[i] = test_path[i].replace("\\", "/")
for i in range(len(validation_path)):
    validation_path[i] = validation_path[i].replace("\\", "/")
    
# Wall, raycast parameters
if choose_bin_hyperparameters == 0:
    
    max_dist = np.sqrt(100*100*2)    

    no_of_bins_voa = 21
    total_angle_voa = np.pi

    no_of_bins_vow = 15
    total_angle_vow = np.pi
    wall_boundaries = np.array([50, -50, 50, -50])

    max_speed = 0.32
    speed_bins = 41
    max_ang_velo = 0.75
    ang_vel_bins = 217

else:
    
    max_dist = np.sqrt(100*100*2)    

    no_of_bins_voa = 151
    total_angle_voa = np.pi * 300 / 180

    no_of_bins_vow = 151
    total_angle_vow = np.pi * 298 / 180
    wall_boundaries = np.array([50, -50, 50, -50])

    max_speed = 0.8
    speed_bins = 215
    max_ang_velo = 0.8
    ang_vel_bins = 215

model_time_steps = 75
no_of_data_trajectories = 0 # Dummy Variable gets updated later    

resume_training = int(input("Enter 1 for resuming from some time step:"))
# resume_training = 0

saved_model_path = '/home/mi/chanakya/fish-models/Saved_Models/'
saved_model_name = input('Enter the name of the file you want to save or load:')
saved_model_path = saved_model_path + saved_model_name
starting_time_step = int(input("Enter the time step from which the training should start:"))
starting_time_step = starting_time_step - 1
if resume_training == 1:
    model = models.load_model(saved_model_path)

if training_data_type == 0 and resume_training == 0:
    model = Moritz_Couzin_Tourus((model_time_steps, no_of_bins_voa + no_of_bins_vow + 2), (model_time_steps, no_of_bins_voa + 1), speed_bins, ang_vel_bins)
if training_data_type == 1 and resume_training == 0:
    model = Moritz_Ladder_Network_Live_Female((model_time_steps, 1 , 2, no_of_bins_voa + 1), speed_bins, ang_vel_bins)
if training_data_type == 2 and resume_training == 0:
    if include_y_labels == 0:
        model = Ladder_Net((model_time_steps, no_of_bins_voa + no_of_bins_vow + 2), (model_time_steps, no_of_bins_voa + 1),speed_bins, ang_vel_bins)
    else:
        model = Ladder_Net_With_Y_Labels((model_time_steps, no_of_bins_voa + no_of_bins_vow + 2), (model_time_steps, no_of_bins_voa + 1),speed_bins, ang_vel_bins)

validation_history = {'loss':[], 'speed_loss':[], 'ang_vel_loss':[], 'speed_acc':[],  'ang_vel_acc':[]}
train_history = {'loss':[], 'speed_loss':[], 'ang_vel_loss':[], 'speed_acc':[],  'ang_vel_acc':[]}

if data_type == 0 or data_type == 2:
    if include_y_labels == 0:
        X_train_f, Y1_train_f, Y2_train_f, _ = data_to_model(training_data_type, train_path, 0, no_of_data_trajectories, max_dist, no_of_bins_voa, total_angle_voa, wall_boundaries, no_of_bins_vow, total_angle_vow, max_speed, speed_bins, max_ang_velo, ang_vel_bins, include_y_labels)
        X_val_f, Y1_val_f, Y2_val_f, _ = data_to_model(training_data_type, validation_path, 0, no_of_data_trajectories, max_dist, no_of_bins_voa, total_angle_voa, wall_boundaries, no_of_bins_vow, total_angle_vow, max_speed, speed_bins, max_ang_velo, ang_vel_bins, include_y_labels)
    else:
        X_train_f, Y1_train_f, Y2_train_f, Y_label_train_f = data_to_model(training_data_type, train_path, 0, no_of_data_trajectories, max_dist, no_of_bins_voa, total_angle_voa, wall_boundaries, no_of_bins_vow, total_angle_vow, max_speed, speed_bins, max_ang_velo, ang_vel_bins, include_y_labels)
        X_val_f, Y1_val_f, Y2_val_f, Y_label_val_f = data_to_model(training_data_type, validation_path, 0, no_of_data_trajectories, max_dist, no_of_bins_voa, total_angle_voa, wall_boundaries, no_of_bins_vow, total_angle_vow, max_speed, speed_bins, max_ang_velo, ang_vel_bins, include_y_labels)

else:
    if include_y_labels == 0:
        X_train_f, Y1_train_f, Y2_train_f, _ = data_to_model_csv_gg(training_data_type, train_path, 0, no_of_data_trajectories, max_dist, no_of_bins_voa, total_angle_voa, wall_boundaries, no_of_bins_vow, total_angle_vow, max_speed, speed_bins, max_ang_velo, ang_vel_bins) 
        X_val_f, Y1_val_f, Y2_val_f, _ = data_to_model_csv_gg(training_data_type, validation_path, 0, no_of_data_trajectories, max_dist, no_of_bins_voa, total_angle_voa, wall_boundaries, no_of_bins_vow, total_angle_vow, max_speed, speed_bins, max_ang_velo, ang_vel_bins)
    else:
        X_train_f, Y1_train_f, Y2_train_f, Y_label_train_f = data_to_model_csv_gg(training_data_type, train_path, 0, no_of_data_trajectories, max_dist, no_of_bins_voa, total_angle_voa, wall_boundaries, no_of_bins_vow, total_angle_vow, max_speed, speed_bins, max_ang_velo, ang_vel_bins) 
        X_val_f, Y1_val_f, Y2_val_f, Y_label_val_f = data_to_model_csv_gg(training_data_type, validation_path, 0, no_of_data_trajectories, max_dist, no_of_bins_voa, total_angle_voa, wall_boundaries, no_of_bins_vow, total_angle_vow, max_speed, speed_bins, max_ang_velo, ang_vel_bins)

X_train_shape, Y1_train_shape, Y2_train_shape = np.shape(X_train_f), np.shape(Y1_train_f), np.shape(Y2_train_f)
X_val_shape, Y1_val_shape, Y2_val_shape = np.shape(X_val_f), np.shape(Y1_val_f), np.shape(Y2_val_f)

if include_y_labels == 1:
    Y_label_train_f_shape = np.shape(Y_label_train_f)
    Y_label_val_f_shape = np.shape(Y_label_val_f)

if training_data_type == 0 or training_data_type == 2:
    X_train = np.reshape(X_train_f[:,0: (X_train_shape[1] - X_train_shape[1]%model_time_steps),:], (X_train_shape[0] * int((X_train_shape[1] - X_train_shape[1]%model_time_steps)/model_time_steps), model_time_steps, X_train_shape[2])) 
    Y1_train = np.reshape(Y1_train_f[:,0: (Y1_train_shape[1] - Y1_train_shape[1]%model_time_steps),:], (Y1_train_shape[0] * int((Y1_train_shape[1] - Y1_train_shape[1]%model_time_steps)/model_time_steps), model_time_steps, Y1_train_shape[2])) 
    Y2_train = np.reshape(Y2_train_f[:,0: (Y2_train_shape[1] - Y2_train_shape[1]%model_time_steps),:], (Y2_train_shape[0] * int((Y2_train_shape[1] - Y2_train_shape[1]%model_time_steps)/model_time_steps), model_time_steps, Y2_train_shape[2]))
    X_val = np.reshape(X_val_f[:,0: (X_val_shape[1] - X_val_shape[1]%model_time_steps),:], (X_val_shape[0] * int((X_val_shape[1] - X_val_shape[1]%model_time_steps)/model_time_steps), model_time_steps, X_val_shape[2])) 
    Y1_val = np.reshape(Y1_val_f[:,0: (Y1_val_shape[1] - Y1_val_shape[1]%model_time_steps),:], (Y1_val_shape[0] * int((Y1_val_shape[1] - Y1_val_shape[1]%model_time_steps)/model_time_steps), model_time_steps, Y1_val_shape[2])) 
    Y2_val = np.reshape(Y2_val_f[:,0: (Y2_val_shape[1] - Y2_val_shape[1]%model_time_steps),:], (Y2_val_shape[0] * int((Y2_val_shape[1] - Y2_val_shape[1]%model_time_steps)/model_time_steps), model_time_steps, Y2_val_shape[2]))
    
if include_y_labels == 1:
    Y_label_train = np.reshape(Y_label_train_f[:,0: (Y_label_train_f_shape[1] - Y_label_train_f_shape[1]%model_time_steps),:], (Y_label_train_f_shape[0] * int((Y_label_train_f_shape[1] - Y_label_train_f_shape[1]%model_time_steps)/model_time_steps), model_time_steps, Y_label_train_f_shape[2]))
    Y_label_val = np.reshape(Y_label_val_f[:,0: (Y_label_val_f_shape[1] - Y_label_val_f_shape[1]%model_time_steps),:], (Y_label_val_f_shape[0] * int((Y_label_val_f_shape[1] - Y_label_val_f_shape[1]%model_time_steps)/model_time_steps), model_time_steps, Y_label_val_f_shape[2]))

else:
    X_train = np.reshape(X_train_f[:,0: (X_train_shape[1] - X_train_shape[1]%model_time_steps),:], (X_train_shape[0] * int((X_train_shape[1] - X_train_shape[1]%model_time_steps)/model_time_steps), model_time_steps, X_train_shape[2], X_train_shape[3], X_train_shape[4])) 
    Y1_train = np.reshape(Y1_train_f[:,0: (Y1_train_shape[1] - Y1_train_shape[1]%model_time_steps),:], (Y1_train_shape[0] * int((Y1_train_shape[1] - Y1_train_shape[1]%model_time_steps)/model_time_steps), model_time_steps, Y1_train_shape[2])) 
    Y2_train = np.reshape(Y2_train_f[:,0: (Y2_train_shape[1] - Y2_train_shape[1]%model_time_steps),:], (Y2_train_shape[0] * int((Y2_train_shape[1] - Y2_train_shape[1]%model_time_steps)/model_time_steps), model_time_steps, Y2_train_shape[2])) 
    X_val = np.reshape(X_val_f[:,0: (X_val_shape[1] - X_val_shape[1]%model_time_steps),:], (X_val_shape[0] * int((X_val_shape[1] - X_val_shape[1]%model_time_steps)/model_time_steps), model_time_steps, X_val_shape[2], X_val_shape[3], X_val_shape[4])) 
    Y1_val = np.reshape(Y1_val_f[:,0: (Y1_val_shape[1] - Y1_val_shape[1]%model_time_steps),:], (Y1_val_shape[0] * int((Y1_val_shape[1] - Y1_val_shape[1]%model_time_steps)/model_time_steps), model_time_steps, Y1_val_shape[2])) 
    Y2_val = np.reshape(Y2_val_f[:,0: (Y2_val_shape[1] - Y2_val_shape[1]%model_time_steps),:], (Y2_val_shape[0] * int((Y2_val_shape[1] - Y2_val_shape[1]%model_time_steps)/model_time_steps), model_time_steps, Y2_val_shape[2]))  

X_train_1 = X_train[:,:,0:152]
X_val_1 = X_val[:,:,0:152]

# Training Hyperparameters
training_epochs = 150
#synonymous to batches
# no_of_data_trajectories = int(np.shape(X_train_f)[0]/4)
no_of_trajectories_rf = 50

print('Y_label_test')
print(Y_label_train[0, 0:15, 0])

print('Without Robot - Fish Interaction Data:')
print('X_train_shape:', np.shape(X_train))
print('Y1_train_shape:', np.shape(Y1_train))
print('Y2_train_shape:', np.shape(Y2_train))
if include_y_labels == 1:
    print('Y_label_train_shape:', np.shape(Y_label_train))
print()
print('X_val_shape:', np.shape(X_val))
print('Y1_val_shape:', np.shape(Y1_val))
print('Y2_val_shape:', np.shape(Y2_val))
if include_y_labels == 1:
    print('Y_val_train_shape:', np.shape(Y_label_val))
print()
  
if include_data_rf == 1:
    X_train_rf, Y1_train_rf, Y2_train_rf, Y_label_train_rf = data_to_model_csv_rf(training_data_type, train_path_rf, 0, no_of_trajectories_rf, model_time_steps, max_dist, no_of_bins_voa, total_angle_voa, wall_boundaries, no_of_bins_vow, total_angle_vow, max_speed, speed_bins, max_ang_velo, ang_vel_bins)
    X_train = np.append(X_train, X_train_rf, axis = 0)
    Y1_train = np.append(Y1_train, Y1_train_rf, axis = 0)
    Y2_train = np.append(Y2_train,  Y2_train_rf, axis = 0)
    
    X_val_rf, Y1_val_rf, Y2_val_rf, Y_label_val_rf = data_to_model_csv_rf(training_data_type, validation_path_rf , 0, no_of_trajectories_rf, model_time_steps, max_dist, no_of_bins_voa, total_angle_voa, wall_boundaries, no_of_bins_vow, total_angle_vow, max_speed, speed_bins, max_ang_velo, ang_vel_bins)
    X_val = np.append(X_val, X_val_rf, axis = 0)
    Y1_val = np.append(Y1_val, Y1_val_rf, axis = 0)
    Y2_val = np.append(Y2_val,  Y2_val_rf, axis = 0)
    
    if include_y_labels == 1:
        Y_label_train = np.append(Y_label_train, Y_label_train_rf, axis = 0)
        Y_label_val = np.append(Y_label_val, Y_label_val_rf, axis = 0)

# save_callback = tf.keras.callbacks.ModelCheckpoint('live_female_female_6.hdf5', save_best_only=True, monitor='val_loss', mode='min')
# history_train = model.fit([X_train, X_train_1], [Y1_train, Y2_train], epochs=training_epochs, validation_data=([X_val, X_val_1], [Y1_val, Y2_val]),callbacks=[save_callback])

print('Without Robot - Fish Interaction Data:')
print('X_train_shape:', np.shape(X_train))
print('Y1_train_shape:', np.shape(Y1_train))
print('Y2_train_shape:', np.shape(Y2_train))
if include_y_labels == 1:
    print('Y_label_train_shape:', np.shape(Y_label_train))
print()
print('X_val_shape:', np.shape(X_val))
print('Y1_val_shape:', np.shape(Y1_val))
print('Y2_val_shape:', np.shape(Y2_val))
if include_y_labels == 1:
    print('Y_val_train_shape:', np.shape(Y_label_val))
print("=======================================================================================================================")

model.save(saved_model_path)

for i in range(starting_time_step, training_epochs):

    model = models.load_model(saved_model_path)
    
    print("Iteration Number:", i + 1)
    print("Training:")
    if include_y_labels == 0:
        history_train = model.fit(X_train, [Y1_train, Y2_train])
    else:
        history_train = model.fit(X_train, [Y1_train, Y2_train, Y_label_train])
#     for j, k in zip(list(train_history.keys()), list(history_train.history.keys())):
#         train_history[j].append(str(history_train.history[k][0]))

    print("Validation:")
    if include_y_labels == 0:
        history_validation = model.evaluate(X_val, [Y1_val, Y2_val])
    else:
        history_validation = model.evaluate(X_val, [Y1_val, Y2_val, Y_label_val])
    print("===================================================================================================================")
#     for j, k in zip(validation_history.keys(), range(len(history_validation))):
#         validation_history[j].append(str(history_validation[k]))

    model.save(saved_model_path)
    
    del model
    tf.keras.backend.clear_session()

#     with open('training_log_live_female_female_8.json', 'w') as fp:
#         json.dump(train_history, fp)
#     with open('validation_log_live_female_female_8.json', 'w') as fp:
#         json.dump(validation_history, fp)


# In[ ]:




