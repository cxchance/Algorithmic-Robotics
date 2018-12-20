#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def get_next_state(cur_state,inputs):

    next_state = [0,0,0]
    next_state[0] = cur_state[0] + inputs[1]*np.cos(cur_state[2]+inputs[0])
    next_state[1] = cur_state[1] + inputs[1]*np.sin(cur_state[2]+inputs[0])
    next_state[2] = cur_state[2] + inputs[0]+inputs[2]
    return next_state

if __name__ == '__main__':
    
    #initialize plotting        
    
    ground_truth_states = np.load('odometry/path.npy') #(20, 3)
    noisy_measurement = np.load('data/corrected.npy') #(20, 3)
    actions = np.load('odometry/inputs.npy') # (19, 3)
    #we are assuming both the motion and sensor noise is 0 mean
    N = ground_truth_states.shape[0]

    motion_errors = np.zeros((3,N))
    sensor_errors = np.zeros((3,N))

    for i in range(1,N):
        x_t = ground_truth_states[i,:] # np (3, )
        x_tminus1 = ground_truth_states[i-1,:] #np (3, ) 
        u_t = actions[i-1,:] #np (3,) 
        z_t = noisy_measurement[i,:] #np (3, )

        motion_errors[:,i] = x_t - np.array(get_next_state(x_tminus1,u_t)) #change this
        sensor_errors[:,i] = z_t - x_t # C is identity matrix

    
    motion_cov=np.cov(motion_errors)
    sensor_cov=np.cov(sensor_errors)
    np.save('data/covar.npy',[motion_cov,sensor_cov])
    
    print "Motion Covariance:"
    print motion_cov
    print "Measurement Covariance:"
    print sensor_cov

    
