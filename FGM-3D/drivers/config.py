import numpy as np
import torch
torch.set_default_dtype(torch.float64)

# Random number seed
seed = 1234

# interactive job
interactive_job = False

# FEM Data specifications
fem_dir = '..';# By default, the data is chosen to be quasistatic, unless specified in 3rd input argument.
fig_dir = '../drivers/plots';
fem_res = '1k'

# data params
lambda_r = 1.0
filter_value = [] # The number of degrees of freedom we wish to subsample to. Use [] for no filtering

alpha = 45.0* 3.14159/180.0
beta  = np.arccos(1.0/np.sqrt(3.0))

theta_fiber2= -alpha  # Direction of another assumed fiber direction in feature library (In radians)

a1 = np.cos(alpha)*np.sin(beta)
a2 = np.sin(alpha)*np.sin(beta)
a3 = np.cos(beta)

b1 = np.cos(theta_fiber2)
b2 = np.sin(theta_fiber2)


a_mult = torch.zeros(9, 1, dtype=torch.float64)

a_mult[0, 0] = a1*a1   # C11
a_mult[1, 0] = a1*a2   # C12
a_mult[2, 0] = a2*a1   # C21
a_mult[3, 0] = a2*a2   # C22
a_mult[4, 0] = a1*a3   # C13
a_mult[5, 0] = a2*a3   # C23
a_mult[6, 0] = a3*a1   # C31
a_mult[7, 0] = a3*a2   # C32
a_mult[8, 0] = a3*a3   # C33


b_mult = torch.zeros(9,1)
b_mult[0:1,0] = b1**2.0
b_mult[1:2,0] = b1*b2
b_mult[2:3,0] = b1*b2
b_mult[3:4,0] = b2**2.0


def get_loadsteps(fem_material):
	#By default, only loadsteps 10, 20, 30, 40, 50 are considered. If required, different number of loadsteps can be considerd for specific materials by adjusting below.
	loadsteps = [1,2,3,4,5,6]
	return loadsteps

def get_feature_filter(fem_material):
	"""
	Certain features from the 28 features in the library can be suppressed by being excluded
	from the 'feature_filter' list below
	"""
	# feature_filter = range(12); # Features 27 and 28 (True Holzapfel features) are suppressed here.
	feature_filter = [0,1,2,3]
	return feature_filter
