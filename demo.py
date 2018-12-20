import time
import numpy as np
import openravepy
from utils2d import *
from matplotlib import pyplot as plt
from copy import deepcopy
from matplotlib.patches import Ellipse

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
		# CDF = cdf(Wt, M)

		##################### XiChen # EECS 568 approach
		particle_kind = 0
		r = np.random.normal(0, 1./M)
		c = Wt[0]
		i = 0
		print Wt
		pos = -1
		b = 0
		for m in range(0, M):
			u = r + float(m) * 1. / M
			while b == 0 and u > c:
				if i >= M - 1:
					b = 1
					break
				i = i + 1
				print i, u, c
				c = c + Wt[i]
			if b == 0:
				if pos != i:
					particle_kind = particle_kind + 1
				pos = i
			print m, pos
			Xt.append(St[pos])
		##################### XiChen
		##################### WeiJian Lecture slide approach
		# kinds = []
		# for m in range(0, M):
		# 	key = np.random.random()
		# 	for i in range(0,M):
		# 		if (key < CDF[i]):
		# 			Xt.append(St[i])
		# 			if (i in kinds) == False:
		# 				kinds.append(i)
		# 			break

		# 	particle_kind = len(kinds)
		# 	Xt.append(St[i])
		##################### WeiJian
		xm = np.mean(np.array(Xt), axis = 0)
		xm = xm.tolist()
		Xm.append(xm)
		xv = np.cov(np.transpose(np.array(Xt)))
		Xv.append(xv)
		print 'mean:', xm
		print 'var', xv
		print '# of kind:', particle_kind
	return [Xm, Xv]

def weight_pf(xt, zt, env, delta):
	###weight calculation###
	###wt = p(zt|xt)###
	####### should not excluding the collision cases (Wei)
	if check_collision(xt):
	# if 0 == 1:	
		wt = 0.
	else:
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

# def cdf(Wt, M):
# 	###generate cdf###
# 	CDF = []
# 	ci = 0
# 	for m in range(0, M):
# 		wt = Wt[m]
# 		ci = ci + wt
# 		CDF.append(ci)
# 	###generate cdf###
# 	return CDF

def initialize_pf(env, M):
	X0 = np.zeros((M, 3)).tolist() # list of (3,) array
	x_start = -7.25
	y_start = -4.75
	theta_start = -0.15
	###initialize list of points across map###
	for i in range(0, M):
		x = x_start + np.random.normal(0, 0.2)
		y = y_start + np.random.normal(0, 0.2)
		theta = theta_start + np.random.normal(0, 0.2)
		X0[i] = [x, y, theta]
	###initialize list of points across map###
	return X0

def check_collision(config):
    rot = quatFromAxisAngle([0.,0.,1.],config[2])
    trans = array([config[0],config[1],0.1])
    pose = concatenate((rot,trans))
    TM = matrixFromPose(pose)
    robot = env.GetRobots()[0]

    with env:
        robot.SetTransform(TM)
        return env.CheckCollision(robot)
#this plots the covariance matrix as an ellipsoid at 2*sigma
def plot_cov_2d(mean,cov,plot_axes):
    lambda_, v = np.linalg.eig(cov)
    lambda_ = lambda_.clip(min=1e-50)
    lambda_ = np.sqrt(lambda_)

    ell = Ellipse(xy=mean[0:2],
              width=lambda_[0]*2, height=lambda_[1]*2,
              angle=np.rad2deg(np.arccos(v[0, 0])))
    ell.set_facecolor('none')
    plot_axes.add_artist(ell)
    plt.scatter(mean[0],mean[1],c='r',marker = '+')
#implement the Kalman filter in this function
def ExtendedKalmanFilter(mu, Sigma, z, u, Q, R):
    ###YOUR CODE HERE###
    #prediction step
    mu_bar = get_next_state(mu,u) # 3, list

    G = get_G(mu,u)
    Sigma_bar = G* Sigma * G.T + R # (3, 3) matrix

    #correction step
    H = get_H(mu,u)
    K = Sigma_bar * H.T*np.linalg.inv(H*Sigma_bar*H.T + Q)

    z = np.array(z)
    mu_bar = np.array(mu_bar)

    mu_new = np.reshape(mu_bar,(3,1)) + K * np.mat((z - mu_bar)).T
    mu_new = np.array(mu_new).tolist()
    mu_new = np.squeeze(mu_new)

    Sigma_new = (np.identity(K.shape[0])- K*H)*Sigma_bar
    ###YOUR CODE HERE###
    return mu_new, Sigma_new

def get_H(cur_state,actions):
    # H = np.mat(np.zeros((3,3)))
    # H[0,0] = np.cos(cur_state[2]+actions[0])
    # H[0,1] = -1.*actions[1]*np.sin(cur_state[2]+actions[0])
    # H[1,0] = np.sin(cur_state[2]+actions[0])
    # H[1,1] = actions[1]*np.cos(cur_state[2]+actions[0])
    # H[2,1] = 1.
    # H[2,2] = 1.
    H = np.identity(3)
    return H

def get_G(cur_state,actions):
    G = np.mat(np.identity(3))
    G[0,2] = -1.*actions[1]*np.sin(cur_state[2]+actions[0])
    G[1,2] = actions[1]*np.cos(cur_state[2]+actions[0])
    return G

def run_ICP(pc_source,pc_target,eps = 1.):
	'''
	list-based approach
	This approach does not convert list to numpy.matrix
	'''
	error_list = []
	pc_current = deepcopy(pc_source)
	T_list = []
	while 1:
		C = []
		for pt_current in pc_current:
			max = 1e10
			close_target = pc_target[0]
			for pt_target in pc_target:
				dist = np.linalg.norm(pt_current - pt_target)
				if(dist < max):
					max = dist
					close_target = pt_target
			C.append([pt_current,close_target])

		# C is a 495-element list, each elem is a list of two 3x1 matrices
		T,R,t = GetTransform(C)

		T_list.append(T)

		error = 0.
		for pair in C:
			error = error + (np.linalg.norm(R * pair[0] + t - pair[1]))**2
		error_list.append(error)

		if len(error_list) > 10 and error_list[-2] - error_list[-1]  < 1e-12:
			eps = 1000

		if error < eps:
			T_total = np.identity(3)
			for j in range(0,len(T_list)):
				T_total = T_list[j] * T_total
			R_total = np.mat(T_total[0:2,0:2])
			t_total = np.mat(T_total[0:2,2])
			return pc_current,R_total,t_total,error_list

		# This does NOT work, pt_current is deep copy of each element
		# for pt_current in pc_current:
		# 	pt_current = R * pt_current + t

		# This work, directly alter pc_current
		for i in range(0,len(pc_current)):
			pc_current[i] = R * pc_current[i] + t

def GetTransform(pair_list):
	'''return T homogeneous transformation 4x4 matrix'''
	n = len(pair_list) # default: n = 495
	S = np.mat(np.zeros([2,2]))
	p_bar = np.mat(np.zeros([2,1]))
	q_bar = np.mat(np.zeros([2,1]))
	for pair in pair_list:
		p_bar = p_bar + pair[0]/n
		q_bar = q_bar + pair[1]/n

	for i in range(0, 2):
		for j in range(0, 2):
			for pair in pair_list:
				# not S[i][j] as in javascript !!
				S[i,j] = S[i,j] + (pair[0][i,0]-p_bar[i,0])*(pair[1][j,0]-q_bar[j,0])

	U,e_values,V_T = np.linalg.svd(S)
	M = np.mat(np.identity(2))
	V = V_T.T
	M[1,1] = np.linalg.det(V*U.T)
	R = V * M * U.T
	t = q_bar - R*p_bar
	T = np.concatenate((np.concatenate((R,t),axis = -1),np.mat([0.,0.,1.])),axis = 0)
	return T,R,t

def trans_pc(pc,guess):
	'''
	:param pc: list of matrices
	:param guess: (3,) array
	:return:
	'''
	R = np.matrix([[np.cos(guess[2]),-np.sin(guess[2])],[np.sin(guess[2]),np.cos(guess[2])]])
	t = np.matrix([[guess[0]],[guess[1]]])
	for i in range(0,len(pc)):
		pc[i] = R * pc[i] + t
	return pc
#### detect_env_noise <~~~> detect_distances
def detect_env_noise(config):
	rot = quatFromAxisAngle([0.,0.,1.],config[2])
	trans = array([config[0],config[1],0.05])
	pose = concatenate((rot,trans))
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
	noise = np.random.normal(0.0,0.2,12)
	distances += noise # (12,) np.array
	local_frame_detection = np.mat(np.zeros((2,12))) # (2,12) np.ndarray
	for i in range(0,12):
	    local_frame_detection[:,i] =np.mat([[distances[i]*np.cos(i * np.pi/6)], [distances[i]*np.sin(i * np.pi/6) ]])

	return local_frame_detection
#### detect_env_noise <~~~> detect_distances   
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
	############## wait until undate 
	olddata = my_laser.GetSensorData(Sensor.Type.Laser)
	while True:
	    cur_data = my_laser.GetSensorData(Sensor.Type.Laser)
	    if cur_data.stamp != olddata.stamp:
	        break
	laser_ranges = cur_data.ranges
	############## wait until undate 
	# waitrobot(robot)
	# laser_ranges = my_laser.GetSensorData(Sensor.Type.Laser).ranges
	laser_ranges = laser_ranges[0:laser_ranges.shape[0]-1,0:2]
	distances = np.linalg.norm(laser_ranges,axis = -1) # (12,) np.array
	############## Wei Jian
	# noise = np.random.normal(0.0,0.2,12)
	# distances += noise # (12,) np.array
	############## Wei Jian

	return distances

def get_input(start,result):
    goal = [
	    result[0]+ np.random.normal(0.0,0.1),
	    result[1]+ np.random.normal(0.0,0.1),
	    result[2]+ np.random.normal(0.0,0.1)]
    input = [0,0,0]
    input[0] = np.arctan2(goal[1] - start[1],goal[0] - start[0]) - start[2]
    input[1] = np.sqrt((goal[1] - start[1])**2 + (goal[0] - start[0])**2)
    input[2] = goal[2] - start[2] - input[0]
    return input
#### get_next_state <~~~> process_pf
def get_next_state(cur_state,actions):
    '''
    :param current_state: (x y theta)
    :param actions: (alpha dist beta)
    :param A:
    :param B:
    :return:
    '''
    # np.random.normal(0.0, 0.1)
    next_state = [0,0,0]
    next_state[0] = cur_state[0] + actions[1]*np.cos(cur_state[2]+actions[0])
    next_state[1] = cur_state[1] + actions[1]*np.sin(cur_state[2]+actions[0])
    next_state[2] = cur_state[2] + actions[0]+actions[2]
    return next_state
#### get_next_state <~~~> process_pf
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

def ConvertPathToTrajectory(robot,path=[]):
	#Path should be of the form path = [q_1, q_2, q_3,...], where q_i = [x_i, y_i, theta_i]

    if not path:
        return None
    # Initialize trajectory
    traj = RaveCreateTrajectory(env,'')
    traj.Init(robot.GetActiveConfigurationSpecification())
    for i in range(0,len(path)):
        traj.Insert(i,numpy.array(path[i]))
    # Move Robot Through Trajectory
    planningutils.RetimeAffineTrajectory(traj,maxvelocities=ones(3),maxaccelerations=5*ones(3))
    return traj

if not __openravepy_build_doc__:
    from openravepy import *
    from numpy import *

if __name__ == "__main__":

	np.random.seed(5000) # collision
	# np.random.seed(10000) # collision
	# np.random.seed(50000) # collision

	############ load the pre-generated hd nmap
	hd_env = np.load('data/HD_map/hd_map.npy') #(1800,2) 2d_array
	############ openrave initialization
	env = Environment()
	env.SetViewer('qtcoin')
	collisionChecker = RaveCreateCollisionChecker(env,'ode')
	env.SetCollisionChecker(collisionChecker)
	env.Reset()
	env.Load('data/env/pr2test4.env.xml')
	time.sleep(0.1)
	robot = env.GetRobots()[0]
	# tuck in the PR2's arms for driving
	tuckarms(env,robot);
	with env:
	    # the active DOF are translation in X and Y and rotation about the Z axis of the base of the robot.
	    robot.SetActiveDOFs([],DOFAffine.X|DOFAffine.Y|DOFAffine.RotationAxis,[0,0,1])
	my_laser = env.GetSensors()[0]
	my_laser.Configure(Sensor.ConfigureCommand.PowerOn)
	my_laser.Configure(Sensor.ConfigureCommand.RenderDataOn)

	############# define path #### generate odometry
	N = 20
	path_0 = [
	    [-7.5, -4.5, 0.0],
	    [-6.5, -4.5, 0.0],
	    [-5.5, -4.5, 0.0],
	    [-4.5, -4.4, 0.15],
	    [-3.5, -4.0, np.pi/3.],
	    [-3.1, -3.0, np.pi/1.9],
	    [-3.0, -2.0, np.pi/2.],
	    [-3.0, -1.0, np.pi/2.],
	    [-3.0, 0.0, np.pi/2.],
	    [-2.8, 1.2, 1.3],
	    [-2.0, 2.5, 1.0],
	    [-1.2, 3.25, 0.5],
	    [-0.2, 3.75, 0.1],
	    [0.8, 3.8, 0.0],
	    [1.8, 3.5, -0.35],
	    [2.75, 3.0, -np.pi/4.],
	    [3.75, 2.0, -1.05],
	    [4.5, 1.0, -1.08],
	    [5.0, 0.0, -1.11],
	    [5.5, -1, -np.pi/2.]]
	path = [
	    [-7.5, -4.5, 0.0],
	    [-6.5, -4.5, 0.0],
	    [-5.5, -4.5, 0.0],
	    [-4.5, -4.4, 0.15],
	    [-3.3, -4.0, 1.05],
	    [-2.7, -3.0, 1.37],
	    [-2.5, -2.0, 1.57],
	    [-2.5, -1.0, 1.57],
	    [-2.5, 0.0, 1.57],
	    [-2.3, 1.2, 1.3],
	    [-1.6, 2.5, 1.0],
	    [-0.7, 3.25, 0.5],
	    [0.3, 3.75, 0.1],
	    [1.3, 3.8, 0.0],
	    [2.3, 3.6, -0.25],
	    [3.4, 3.2, -0.8],
	    [4.4, 2.3, -1.05],
	    [5.1, 1.2, -1.1],
	    [5.6, 0.2, -1.1],
	    [6.0, -0.8, -1.57]]

	######## check obstacle-free
	print('check collision for ground truth states')
	for config in path:
		print(check_collision(config))

	####### show ground truth path (Optional)
	# P = np.array(path)
	# plt.plot(P[:,0],P[:,1],'o')
	# plt.plot(hd_env[:,0],hd_env[:,1],'.k')
	# plt.axis("equal")
	# plt.show()
	# assert(0)

	################## generate inputs (optionally save)
	actions = []
	for i in range(0,len(path)-1):
	    cur_input = get_input(path[i],path[i+1])
	    actions.append(cur_input)
	np.save("data/odometry/new_actions.npy",actions)
	np.save("data/odometry/new_path.npy",path)

	################## sensing (and save)
	sensor_outputs = [] # list of (2,12) ndarray
	for i in range(0,len(path)):
	    sensor_outputs.append(detect_env_noise(path[i]))
	np.save('data/new_test_noise.npy',sensor_outputs)

	############## (Optional) Execute the trajectory on the robot.
	# traj = ConvertPathToTrajectory(robot, path)
	# if traj != None:
	#     robot.GetController().SetPath(traj)
	# waitrobot(robot)
	# assert(0)

	############ ICP process
	pc_target = convert_matrix_to_pc(np.mat(hd_env.T)) # input should be a (2, 1800) matrix
	prediction = [[-7.25, -4.75, -0.15]]
	noisy_measurement = []

	for i in range(0,20): # for 0 <= i < 20
		cur_pc = convert_matrix_to_pc(np.mat(sensor_outputs[i])) # (2,12) matrix
		cur_pc = trans_pc(cur_pc,prediction[i])
		pc_current, R_total, t_total, error_list = run_ICP(cur_pc,pc_target,0.0001)
		print i,'error =',error_list[-1]
		# plot the "bad" case
		# if(error_list[-1] > 1.):
		# 	print("large error:",error_list[-1])
		# 	fig = view_pc([pc_current, pc_target], None, ['r', 'g'], ['o', '.'])
		corrected_pos = R_total*np.matrix([[prediction[i][0]],[prediction[i][1]]]) + t_total
		corrected_angle = np.arcsin(R_total[1,0]) + prediction[i][2]
		noisy_measurement.append([corrected_pos[0,0],corrected_pos[1,0],corrected_angle])
		if i < 19: # no actions[19], no prediction for i = 19
			prediction.append(get_next_state(noisy_measurement[i],actions[i])) # input[i] means u_{k+1}, only 19 elements
	np.save("data/new_predicted.npy",prediction)
	np.save("data/new_corrected.npy",noisy_measurement)

	################ generate covariance matrices
	###### we are assuming both the motion and sensor noise is 0 mean
	N = 20
	motion_errors = np.zeros((3,N))
	sensor_errors = np.zeros((3,N))

	for i in range(1,N):
	    x_t = path[i] # (3, ) list
	    x_tminus1 = path[i-1] # (3, ) list
	    u_t = actions[i-1] #(3, ) list
	    z_t = noisy_measurement[i] #(3, ) list
	    motion_errors[:,i] = np.array(x_t) - np.array(get_next_state(x_tminus1,u_t)) #change this
	    sensor_errors[:,i] = np.array(z_t) - np.array(x_t) # C is identity matrix
	# print(motion_errors)
	# print(sensor_errors)
	R = np.cov(motion_errors) # motion_cov
	Q = np.cov(sensor_errors) # sensor_cov
	np.save('data/new_covar.npy',[R,Q])

	plt.figure(1)
	plt.ion()
	plot_axes = plt.subplot(111, aspect='equal')   

	mu = noisy_measurement[0] # (3, ) list
	Sigma = 0.5**np.eye(3) # (3,3) ndarray
	#go through each measurement and action...
	#and estimate the state using the Kalman filter
	estimated_states = np.zeros((N,3)) 
	estimated_states[0,:] = np.array([-7.25, -4.75, -0.15])
	for i in range(1,N):
		z = noisy_measurement[i] #current x // np (3, )
		u = actions[i-1]         #current u # // np (3, )
		#run the Kalman Filter
		mu, Sigma = ExtendedKalmanFilter(mu, Sigma, z, u, Q, R)
		######### check obstacle-free
		print 'EKF-ICP collision ?', i, check_collision(mu.tolist())
		#store the result
		estimated_states[i,:] = np.squeeze(mu)  
		#draw covariance every 3 steps (drawing every step is too cluttered)
		plot_cov_2d(mu,Sigma,plot_axes)

	np.save("ekf_mean.npy",estimated_states)
	#compute the error between your estimate and ground truth
	state_errors = estimated_states[0:N,:] - np.array(path)[0:N,:]
	# total_error = np.sum(np.linalg.norm(state_errors, axis=1))
	# print "Total Error: %f"%total_error

	####### draw the data and result
	######## convert to matrices, easy for drawing
	path = np.array(path)
	noisy_measurement = np.array(noisy_measurement)

	plt.plot(hd_env[:,0],hd_env[:,1],'.k')
	plt.plot(path[:,0], path[:,1],color ='b',marker = 'x',linewidth=2.0,label='ground truth')
	plt.plot(noisy_measurement[:,0], noisy_measurement[:,1],color ='g',marker = 'x',linewidth=2.0,label='noisy measurement')
	plt.legend()
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('EKF-ICP Method')
	plt.savefig('EKF-ICP.png')
	# plt.show()
	plt.pause(.001)
	print('EKF-ICP done')
	#### YOUR CODE HERE ####
	M = 64 ###Number of samples### Set to 300 for results
	epsilon = 0.1 #### gaussian noise
	delta = 0.15 #### normal distribution (for calculating weights)

	U = actions #### input === actions
	Z_temp = np.load('data/new_test_noise.npy') #(20,2,12)
	N, dim, L = Z_temp.shape
	Z_temp = Z_temp.tolist()
	Z = np.zeros((N, L)).tolist()
	for i in range(0, N):
		for j in range(0, L):
			Z[i][j] = np.sqrt(Z_temp[i][0][j]**2 + Z_temp[i][1][j]**2)

	[Xm, Xv] = algorithm_pf(env, U, epsilon, Z, delta, N, M) ### lists of arrays (not lists)
	np.save("data/pf_mean.npy",Xm)
	np.save("data/pf_var.npy",Xv)

	print('check collision for PF mean')
	for config in Xm: ### config(xm) are 1d array
		print(check_collision(config)) 

	plt.figure(2)
	plt.ion()
	plot_axes = plt.subplot(111, aspect='equal')
	for i in range(0,N):
		plot_cov_2d(np.array(Xm[i]),Xv[i],plot_axes) ####### Xm[i] is a list, Xv[i] is a 2darray

	plt.plot(hd_env[:,0],hd_env[:,1],'.k')
	plt.plot(path[:,0], path[:,1],color ='b',marker = 'x',linewidth=2.0,label='ground truth')
	plt.legend()
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('PF Method')
	plt.savefig('PF.png')
	plt.show()
	#### END OF YOUR CODE ###
	raw_input("Press enter to exit...")
	env.Destroy()