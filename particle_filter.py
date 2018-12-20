#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import openravepy
from matplotlib import pyplot as plt

if not __openravepy_build_doc__:
    from openravepy import *
    import numpy as np

def waitrobot(robot):
    """busy wait for robot completion"""
    while not robot.GetController().IsDone():
        time.sleep(0.01)

def tuckarms(env,robot):
    with env:
        jointnames = ['l_shoulder_lift_joint','l_elbow_flex_joint','l_wrist_flex_joint','r_shoulder_lift_joint','r_elbow_flex_joint','r_wrist_flex_joint']
        robot.SetActiveDOFs([robot.GetJoint(name).GetDOFIndex() for name in jointnames])
        robot.SetActiveDOFValues([1.29023451,-2.32099996,-0.69800004,1.27843491,-2.32100002,-0.69799996]);
        robot.GetController().SetDesired(robot.GetDOFValues());
    waitrobot(robot)

def check_collision(config):
    rot = quatFromAxisAngle([0.,0.,1.],config[2])
    trans = np.array([config[0],config[1],0.05])
    pose = np.concatenate((rot,trans))
    TM = matrixFromPose(pose)

    with env:
        robot.SetTransform(TM)
        return env.CheckCollision(robot)

def detect_distances(config,result_set = []):
	'''this function is very similar to detect_env_noise,
	but that function gives the coordinates in local frame, 
	while this one only outputs the distances'''
	rot = quatFromAxisAngle([0.,0.,1.],config[2])
	trans = np.array([config[0],config[1],0.05])
	pose = np.concatenate((rot,trans))
	TM = matrixFromPose(pose)

	with env:
	    robot.SetTransform(TM)

	olddata = my_laser.GetSensorData(Sensor.Type.Laser)
	while True:
	    cur_data = my_laser.GetSensorData(Sensor.Type.Laser)
	    if cur_data.stamp != olddata.stamp:
	        break

	laser_ranges = cur_data.ranges
	laser_ranges = laser_ranges[0:laser_ranges.shape[0]-1,0:2]
	distances = np.linalg.norm(laser_ranges,axis = -1) # (12,) np.array
	############## Wei Jian
	noise = np.random.normal(0.0,0.2,12)
	distances += noise # (12,) np.array
	############## Wei Jian

	return distances

###variables definition###
###M: number of samples###
###N: number of steps###
###U: process, N-1*3###
###Z: measurement, N*12###
###Xt: list of points, M###
###St: intermediate sample points, M###
###Wt: intermediate sample weights, M###
###variables definition###

def initialize_pf(env, M):
	X0 = np.zeros((M, 3)).tolist() # list of (3,) array
	x_start = -7.25
	y_start = -4.75
	theta_start = -0.15
	###initialize list of points across map###
	for i in range(0, M):
		x = x_start + np.random.normal(0, 0.5)
		y = y_start + np.random.normal(0, 0.5)
		theta = theta_start + np.random.normal(0, 0.5)
		X0[i] = [x, y, theta]
	###initialize list of points across map###
	return X0

###assume small motion, only check obstacle after motion###
def process_pf(xt_prev, ut, epsilon = 0.1):
	### this function is very similar to get_next_step (prediction step) but with additional noise
	###process model###
	x = xt_prev[0]
	y = xt_prev[1]
	theta = xt_prev[2]
	rot1 = ut[0]
	trans = ut[1]
	rot2 = ut[2]

	x = x + trans * np.cos(theta + rot1) + np.random.normal(0, epsilon)
	y = y + trans * np.sin(theta + rot1) + np.random.normal(0, epsilon)
	theta = theta + rot1 + rot2 + np.random.normal(0, epsilon)
	xt = [x, y, theta]
	###process model###
	return xt

def weight_pf(xt, zt, env, delta):
	###weight calculation###
	###wt = p(zt|xt)###
	####### should not excluding the collision cases (Wei)
	distances = []
	distances = detect_distances(xt, distances)
	distances = distances.tolist()
	sensor_number = len(zt)
	wt_log = 0.
	effective_number = 0
	for i in range(0, sensor_number):
		if distances[i] < 5:
			effective_number = effective_number + 1
			wt_log = wt_log + np.log(1./np.sqrt(2.* np.pi * delta**2)) * (-(distances[i]-zt[i])**2/(2. * delta**2))
	if effective_number == 0:
		wt = 0.
	else:
		wt_log = wt_log/effective_number
		wt = np.exp(wt_log)
	###weight calculation###

	return wt

def weight_normalize(Wt):
	###normalize weight###
	Wt_array = np.array(Wt)
	Wt_array = Wt_array / np.sum(Wt_array)
	Wt = Wt_array.tolist()
	###normalize weight###
	return Wt

def cdf(Wt, M):
	###generate cdf###
	CDF = []
	ci = 0
	for m in range(0, M):
		wt = Wt[m]
		ci = ci + wt
		CDF.append(ci)
	###generate cdf###
	return CDF


def algorithm_pf(env, U, epsilon, Z, delta, N, M):
	###env: environment###
	###U: control sequency###
	###epsilon: odeometry Gaussian error###
	###Z: measurement sequence###
	###delta: measurement Gaussian error###
	###M: number of samples###
	###N: number of steps###
	Xt = initialize_pf(env, M) #(M, 3) list of list
	Xm = []
	Xv = []
	for t in range(0, N):
		if t != 0:
			ut = U[t-1]
		zt = Z[t]
		Xt_prev = Xt
		St = []
		Xt = []
		Wt = []
		###process###
		for m in range(0, M):
			xt_prev = Xt_prev[m]
			if t == 0:
				xt = xt_prev
			else:
				xt = process_pf(xt_prev, ut, epsilon)
			###theta issue###
			# xt[2] = path[t][2]
			###theta issue###
			wt = weight_pf(xt, zt, env, delta)
			St.append(xt)
			Wt.append(wt)
		###normalize###
		Wt = weight_normalize(Wt)
		# print Wt

		###resample###
		CDF = cdf(Wt, M)

		##################### XiChen
		# uj = 1./M
		# i = 0
		# particle_kind = 1
		# for m in range(0, M):
		# 	flag = 0
		# 	while uj > CDF[i]:
		# 		i = i + 1
		# 		flag = 1
		# 		# print("i:", i)
		# 		if i >= M-1:
		# 			break
		# 	particle_kind = particle_kind + flag
		# 	Xt.append(St[i])
		# 	uj = uj + 1./M
		##################### XiChen

		##################### WeiJian
		kinds = []
		for m in range(0, M):
			key = np.random.random()
			for i in range(0,M):
				if (key < CDF[i]):
					Xt.append(St[i])
					if (i in kinds) == False:
						kinds.append(i)
					break

			particle_kind = len(kinds)

			Xt.append(St[i])

		##################### WeiJian

		xm = np.mean(np.array(Xt), axis = 0)
		xm = xm.tolist()
		Xm.append(xm)
		xv = np.cov(np.transpose(np.array(Xt)))
		print(xv.shape)
		Xv.append(xv)
		print 'mean:', xm
		# print xv
		print '# of kind:', particle_kind
	return [Xm, Xv]

if __name__ == '__main__':

	env = Environment()
	env.SetViewer('qtcoin')
	collisionChecker = RaveCreateCollisionChecker(env,'ode')
	env.SetCollisionChecker(collisionChecker)

	env.Reset()
	env.Load('data/env/pr2test4.env.xml')
	time.sleep(0.1)

	# 1) get the 1st robot that is inside the loaded scene
	# 2) assign it to the variable named 'robot'
	robot = env.GetRobots()[0]

	my_laser = env.GetSensors()[0]
	my_laser.Configure(Sensor.ConfigureCommand.PowerOn)
	my_laser.Configure(Sensor.ConfigureCommand.RenderDataOn)

	# tuck in the PR2's arms for driving
	tuckarms(env,robot);
	with env:
	    # the active DOF are translation in X and Y and rotation about the Z axis of the base of the robot.
	    robot.SetActiveDOFs([],DOFAffine.X|DOFAffine.Y|DOFAffine.RotationAxis,[0,0,1])


	#### YOUR CODE HERE ####
	U = np.load('data/odometry/new_inputs.npy')
	U = U.tolist()

	epsilon = 0.1

	Z_temp = np.load('data/new_test_noise.npy') #(20,2,12)
	N, dim, L = Z_temp.shape

	Z_temp = Z_temp.tolist()

	Z = np.zeros((N, L)).tolist()
	for i in range(0, N):
		for j in range(0, L):
			Z[i][j] = np.sqrt(Z_temp[i][0][j]**2 + Z_temp[i][1][j]**2)
	# print len(Z), len(Z[0]), len(U), len(U[0])
	###Number of samples###
	M = 5
	path = np.load('data/odometry/new_path.npy')
	path = path.tolist()
	# print path
	delta = 0.1
	[Xm, Xv] = algorithm_pf(env, U, epsilon, Z, delta, N, M)
	np.save("data/pf_mean.npy",Xm)
	np.save("data/pf_var.npy",Xv)
	#### END OF YOUR CODE ###


	raw_input("Press enter to exit...")
	env.Destroy()