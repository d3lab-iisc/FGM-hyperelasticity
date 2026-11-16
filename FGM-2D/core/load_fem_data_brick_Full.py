import torch
import numpy as np
import pandas as pd
from utilities import *
from features_library import *
from config import *
from data_definitions import *
import pyvista as pv
from paraview_plot import *

def getdNdxi(xi,eta,beta):
	N = np.array([(1+xi)*(1-eta)*(1-beta),(1+xi)*(1+eta)*(1-beta),(1-xi)*(1+eta)*(1-beta),(1-xi)*(1-eta)*(1-beta),(1+xi)*(1-eta)*(1+beta),(1+xi)*(1+eta)*(1+beta),(1-xi)*(1+eta)*(1+beta),(1-xi)*(1-eta)*(1+beta)])/8.0
	dNdxi = np.array([[(1-eta)*(1-beta),(1+eta)*(1-beta),-(1+eta)*(1-beta),-(1-eta)*(1-beta),(1-eta)*(1+beta),(1+eta)*(1+beta),-(1+eta)*(1+beta),-(1-eta)*(1+beta)],\
	[(1+xi)*(-1)*(1-beta),(1+xi)*(1)*(1-beta),(1-xi)*(1)*(1-beta),(1-xi)*(-1)*(1-beta),(1+xi)*(-1)*(1+beta),(1+xi)*(1)*(1+beta),(1-xi)*(1)*(1+beta),(1-xi)*(-1)*(1+beta)],\
	[(1+xi)*(1-eta)*(-1),(1+xi)*(1+eta)*(-1),(1-xi)*(1+eta)*(-1),(1-xi)*(1-eta)*(-1),(1+xi)*(1-eta)*(1),(1+xi)*(1+eta)*(1),(1-xi)*(1+eta)*(1),(1-xi)*(1-eta)*(1)]])/8.0
	dNdxi = dNdxi.transpose()
	return dNdxi, N

def loadFemData(path, AD = True, noiseLevel = 0, noiseType = 'displacement', denoisedDisplacements = None):
	"""
	Load finite element data and add noise (optional).
	Note that the loaded finite element data might already be perturbed by noise.
	In that case, adding additional noise is not necessary.

	_Input Arguments_

	- `path` - path to finite element data

	- `AD` - specify if automatic differention is needed

	- `noiseLevel` - noise level

	- `noiseType` - a string list specifying whether noise should be added to 'displacement','strain','acceleration','velocity'

	- `denoisedDisplacements` - pass denoised displacement data if available

	_Output Arguments_

	- `dataset` - finite element dataset

	---

	"""
	print('\n-----------------------------------------------------')
	print('Loading data: ', path)
	if(path[-1]=='/'):
		path=path[0:-1]
	numNodesPerElement = 8
	quadpoints = np.array([[1,-1,-1],[1,1,-1],[-1,-1,-1],[-1,1,-1],[1,-1,1],[1,1,1],[-1,-1,1],[-1,1,1]])/2.0 # no of integration points
	#nodal data
	df = pd.read_csv(path+'/output_nodes.csv',dtype=np.float64)
	numNodes = df.shape[0]
	x_nodes = torch.tensor(df[['x','y','z']].values)# This is not multiplied by 1000 for avoiding connectivity issues.
	u_nodes = torch.tensor(df[['ux','uy','uz']].values)


	if (('vx' in df.columns) and ('vy' in df.columns)):
		v_nodes = torch.tensor(df[['vx','vy','vz']].values)
	else:
		v_nodes = torch.zeros_like(u_nodes)

	if (('ax' in df.columns) and ('ay' in df.columns)):
		a_nodes = torch.tensor(df[['ax','ay','az']].values)
	else:
		a_nodes = torch.zeros_like(u_nodes)

	if(denoisedDisplacements is not None):
		u_nodes = denoisedDisplacements

	bcs_nodes = torch.tensor(df[['bcx','bcy','bcz']].round().astype(int).values)
	#convert bcs_nodes to booleans
	dirichlet_nodes = (bcs_nodes!=0)

	#noise
	if(('displacement' in noiseType) or ('strain' in noiseType) or ('velocity' in noiseType) or ('acceleration' in noiseType)):
		pass
	else:
		raise ValueError('Incorrect noiseType argument!')


	if('displacement' in noiseType and noiseLevel > 1e-10):
		
		print('Applying noise to displacements:',noiseLevel)
		sig_u = noiseLevel
		lambda_u = 1.0
		alpha = 5.0

		x_min = torch.min(x_nodes[:, 0])
		x_max = torch.max(x_nodes[:, 0])
		y_min = torch.min(x_nodes[:, 1])
		y_max = torch.max(x_nodes[:, 1])
		z_min = torch.min(x_nodes[:, 2])
		z_max = torch.max(x_nodes[:, 2])

		dx = torch.min(torch.stack([x_nodes[:, 0] - x_min, x_max - x_nodes[:, 0]], dim=1), dim=1).values
		dy = torch.min(torch.stack([x_nodes[:, 1] - y_min, y_max - x_nodes[:, 1]], dim=1), dim=1).values
		dz = torch.min(torch.stack([x_nodes[:, 2] - z_min, z_max - x_nodes[:, 2]], dim=1), dim=1).values

		# For 2D Plate
		d = torch.min(torch.stack([dx, dy], dim=1), dim=1).values

		u_true_np = df[['ux', 'uy', 'uz']].values
		
		L_diag = torch.sqrt((1.0 + alpha * torch.exp(-d**2.0 / (2.0 * lambda_u**2.0))) * sig_u**2.0)
		L_matrix = torch.diag(L_diag)

		noise_u_nodes = L_matrix @ torch.randn(x_nodes.shape[0], 3, device=x_nodes.device)
		u_nodes += noise_u_nodes

		runparaview_noise(step_folder=path, num_steps=6, noising_term=noise_u_nodes.cpu().numpy(), denoised_noise_term = None, u_denoised=None)

	# np.savetxt('output_nodes_noised.csv',np.column_stack((x_nodes.numpy(),u_nodes.numpy())),header='x,y,z,ux,uy,uz',comments='',delimiter=",")

	noise_a_nodes = noiseLevel * torch.randn_like(a_nodes)
	noise_a_nodes[dirichlet_nodes] = 0.
	if('acceleration' in noiseType):
		a_nodes += noise_a_nodes
		print('Applying noise to accelerations:',noiseLevel)

	noise_v_nodes = noiseLevel * torch.randn_like(v_nodes)
	noise_v_nodes[dirichlet_nodes] = 0.
	if('velocity' in noiseType):
		v_nodes += noise_v_nodes
		print('Applying noise to velocities:',noiseLevel)

	#reaction forces data
	numReactions = torch.max(bcs_nodes).item()
	df = pd.read_csv(path+'/output_reactions.csv',dtype=np.float64)
	reactions = []
	for i in range(numReactions):
		reactions.append(Reaction(bcs_nodes == (i+1),df['forces'][i])) # These reaction forces are force/thickness in dimension.

	#element data
	df = pd.read_csv(path+'/output_elements.csv',dtype=np.float64)
	connectivity = np.zeros(df.shape,dtype=np.int32)
	nodesperlelem = connectivity.shape[1]
	numElements = connectivity.shape[0]
	for i in range(nodesperlelem):
		connectivity[:,i] = df['node'+str(i+1)].to_numpy()

# F11 = 1+uN1[0]*gradN1[:,0]+uN2[0]*gradN2[:,0]+uN3[0]*gradN3[:,0]+uN4[0]*gradN4[:,0]
# F12 = uN1[0]*gradN1[:,1]+uN2[0]*gradN2[:,1]+uN3[0]*gradN3[:,1]+uN4[0]*gradN4[:,1]
# F13 = uN1[0]*gradN1[:,2]+uN2[0]*gradN2[:,2]+uN3[0]*gradN3[:,2]+uN4[0]*gradN4[:,2]
# F21 = uN1[1]*gradN1[:,0]+uN2[1]*gradN2[:,0]+uN3[1]*gradN3[:,0]+uN4[1]*gradN4[:,0]
# F22 = 1+uN1[1]*gradN1[:,1]+uN2[1]*gradN2[:,1]+uN3[1]*gradN3[:,1]+uN4[1]*gradN4[:,1]
# F23 = uN1[1]*gradN1[:,2]+uN2[1]*gradN2[:,2]+uN3[1]*gradN3[:,2]+uN4[1]*gradN4[:,2]
# F31 = uN1[2]*gradN1[:,0]+uN2[2]*gradN2[:,0]+uN3[2]*gradN3[:,0]+uN4[2]*gradN4[:,0]
# F32 = uN1[2]*gradN1[:,1]+uN2[2]*gradN2[:,1]+uN3[2]*gradN3[:,1]+uN4[2]*gradN4[:,1]
# F33 = 1+uN1[2]*gradN1[:,2]+uN2[2]*gradN2[:,2]+uN3[2]*gradN3[:,2]+uN4[2]*gradN4[:,2]
	#
	# np.savetxt('output_exp_defgrads.csv',np.column_stack((F11,F22,F33,F12,F13,F23,F21,F31,F32)),header='F11,F22,F33,F12,F13,F23,F21,F31,F32', fmt='%20.20e,%20.20e,%20.20e,%20.20e,%20.20e,%20.20e,%20.20e,%20.20e,%20.20e', comments='')

	#--------------------------------------------------
	connectivity = connectivity - 1
	#-----------------------------------------------
	dNdxi = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])/8.0
	Xallels = np.zeros((3*connectivity.shape[0],nodesperlelem))
	dispallels = np.zeros((3*connectivity.shape[0],nodesperlelem))

	Xpoints = x_nodes[connectivity,0].numpy()
	Ypoints = x_nodes[connectivity,1].numpy()
	Zpoints = x_nodes[connectivity,2].numpy()

	uxpoints = u_nodes[connectivity,0].numpy()
	uypoints = u_nodes[connectivity,1].numpy()
	uzpoints = u_nodes[connectivity,2].numpy()

	inds = 3*np.arange(connectivity.shape[0])
	Xallels[inds,:] = Xpoints
	Xallels[inds+1,:] = Ypoints
	Xallels[inds+2,:] = Zpoints

	dispallels[inds,:] = uxpoints
	dispallels[inds+1,:] = uypoints
	dispallels[inds+2,:] = uzpoints

	dispallels = dispallels.reshape((-1,3,8))#now has dims el X 3 X 8

	Ftotal = np.array([])
	gradNaTotal = np.array([])
	NaTotal = np.array([]) # Initialize NaTotal
	connectivityTotal = connectivity
	for quadpoint in range(quadpoints.shape[0]):
		dNdxi,N_val = getdNdxi(quadpoints[quadpoint,0],quadpoints[quadpoint,1],quadpoints[quadpoint,2])
		J = np.matmul(Xallels,dNdxi)
		J = J.reshape((-1,9))
		detJ = J[:,0]*J[:,4]*J[:,8] - J[:,0]*J[:,5]*J[:,7] - J[:,1]*J[:,3]*J[:,8] + J[:,1]*J[:,5]*J[:,6] + J[:,2]*J[:,3]*J[:,7] - J[:,2]*J[:,4]*J[:,6]
		detJ = detJ.reshape((-1,1))
		invJ = np.column_stack((J[:,4]*J[:,8]-J[:,5]*J[:,7],-(J[:,1]*J[:,8]-J[:,2]*J[:,7]),(J[:,1]*J[:,5]-J[:,2]*J[:,4]),-(J[:,3]*J[:,8]-J[:,5]*J[:,6]),(J[:,0]*J[:,8]-J[:,2]*J[:,6]),-(J[:,0]*J[:,5]-J[:,2]*J[:,3]),(J[:,3]*J[:,7]-J[:,4]*J[:,6]),-(J[:,0]*J[:,7]-J[:,1]*J[:,6]),(J[:,0]*J[:,4]-J[:,1]*J[:,3])))/detJ
		invJ = invJ.reshape((-1,3))
		invJ = invJ.reshape((-1,3,3))#now has dims el X 3 X 3
		delta_ij = np.array([[1,0,0],[0,1,0],[0,0,1]])

		F = np.zeros((connectivity.shape[0],3,3))
		gradnanumpy = np.zeros((connectivity.shape[0],8,3))
		for el in range(connectivity.shape[0]):
			gradnanumpy[el,:,:] = np.matmul(dNdxi,invJ[el,:,:])
			F[el,:,:] = delta_ij + np.matmul(dispallels[el,:,:],gradnanumpy[el,:,:])

		# Enforcing Plane Stress Condition and incompressibility

		F[:, 0, 2] = 0.0   # F13
		F[:, 1, 2] = 0.0   # F23
		F[:, 2, 0] = 0.0   # F31
		F[:, 2, 1] = 0.0   # F32

		F11 = F[:, 0, 0]
		F12 = F[:, 0, 1]
		F21 = F[:, 1, 0]
		F22 = F[:, 1, 1]

		F[:, 2, 2] = 1.0 / (F11*F22 - F21*F12 + 1e-12) 

		F = F.reshape(-1,9)
		gradnanumpy = gradnanumpy.reshape(-1,24)
		gradNa = []
		Na = [] # List to hold N for each node
		# Tile N_val for all elements in this batch: shape (numElements, 8)
		N_expanded = np.tile(N_val, (connectivity.shape[0], 1))
		for k in range(nodesperlelem):
			gradNa.append(torch.tensor(gradnanumpy[:,(3*k):(3*k+3)]))
			Na.append(torch.tensor(N_expanded[:, k:k+1])) # Append N values for node k, keeping dimension as (NumElems, 1)
		if quadpoint ==0:
			qpWeightsTotal = detJ
			Ftotal = F
			gradNaTotal = gradNa
			NaTotal = Na # Initialize NaTotal with first quadpoint data
		else:
			Ftotal = np.concatenate((Ftotal,F),axis=0)
			qpWeightsTotal = np.concatenate((qpWeightsTotal,detJ),axis=0)
			for gradindex in range(len(gradNaTotal)):
				gradNaTotal[gradindex] = torch.cat((gradNaTotal[gradindex],gradNa[gradindex]),dim=0)
				NaTotal[gradindex] = torch.cat((NaTotal[gradindex],Na[gradindex]),dim=0) # Concatenate Na
			connectivityTotal = np.concatenate((connectivityTotal,connectivity),axis=0)

	Fall = np.column_stack((Ftotal,Ftotal,Ftotal,Ftotal,Ftotal,Ftotal,Ftotal,Ftotal))
	Fall = np.reshape(Fall,(-1,9))
	mask = ~np.isnan(Fall)
	mask = np.prod(mask,axis=1,dtype=np.bool_)
	elems2 = np.reshape(connectivityTotal,(-1,1))
	Fnodesall = np.zeros((x_nodes.shape[0],9))
	np.add.at(Fnodesall,elems2[mask].squeeze(),Fall[mask.squeeze(),:])

	ons = np.ones(elems2.shape)
	denoms = np.zeros(x_nodes.shape[0])
	# denoms=denoms.squeeze()
	# np.add.at(denoms,elems[mask].squeeze(),ons[mask,:].squeeze())
	np.add.at(denoms,elems2[mask].squeeze(),ons[mask].squeeze())
	# denoms[denoms==0]=1
	denoms=np.reshape(denoms,(-1,1))
	Fnodesall = Fnodesall/denoms
	detF = Fnodesall[:,0]*(Fnodesall[:,4]*Fnodesall[:,8]-Fnodesall[:,7]*Fnodesall[:,5]) - Fnodesall[:,1]*(Fnodesall[:,3]*Fnodesall[:,8]-Fnodesall[:,6]*Fnodesall[:,5]) + Fnodesall[:,2]*(Fnodesall[:,3]*Fnodesall[:,7]-Fnodesall[:,4]*Fnodesall[:,6])
	indchange = [0,1,3,4,2,5,6,7,8]
	Ftotal = Ftotal[:,indchange]
	F = torch.tensor(Ftotal)
	Ftotal = F

	nelems = 8*np.ones((connectivity.shape[0],),dtype=np.int32)
	elems2 = np.column_stack((nelems,connectivity))
	elems2 = elems2.flatten()
	celltypes = 12*np.ones(connectivity.shape[0],dtype=np.int32)
	grid = pv.UnstructuredGrid(elems2, celltypes, x_nodes.numpy())
	grid.point_data['displacement'] = u_nodes.numpy()
	grid.point_data['DG'] = Fnodesall
	grid.point_data['detF'] = detF
	grid.save(path+'/EUCLIDvtkcheck.vtk')
	# computing deformation gradient at quadrature points
	# dim=3
	# voigtMap = [[0,1,4],[2,3,5],[6,7,8]]
	#
	# F=torch.zeros(numElements,9)
	# for a in range(numNodesPerElement):
	# 	for i in range(dim):
	# 		for j in range(dim):
	# 			F[:,voigtMap[i][j]] += u[a][:,i] * gradNa[a][:,j]
	#
	# F[:,0] += 1.0
	# F[:,3] += 1.0
	# F[:,8] += 1.0

	if('strain' in noiseType):
		F += noiseLevel * torch.randn_like(F)
		print('Applying noise to strains:',noiseLevel)
	#computing detF
	J = computeJacobian(F)

	#computing Cauchy-Green strain: C = F^T F
	C = computeCauchyGreenStrain(F)

	#computing strain invariants
	I1, I2, I3 = computeStrainInvariants(C)

	# a_mult is defined in config
	Ia = C @ a_mult
	Ib = C @ b_mult

	#activate gradients
	Ijunk = I1
	Ijunk.requires_grad = True
	I1.requires_grad = True
	I2.requires_grad = True
	I3.requires_grad = True
	Ia.requires_grad = True
	Ib.requires_grad = True

	#computing invariant derivaties
	dI1dF = computeStrainInvariantDerivatives(F,1)
	dI2dF = computeStrainInvariantDerivatives(F,2)
	dI3dF = computeStrainInvariantDerivatives(F,3)
	dIadF = computeStrainInvariantDerivatives(F,4)
	dIbdF = computeStrainInvariantDerivatives(F,5)

	# Ijunk.detach()
	#computing extended set of nonlinear features
	featureSet = FeatureSet()
	# featureSet.features = computeFeatures_torch(Ijunk, Ijunk, Ijunk, Ijunk, Ijunk, I1, I2, I3, Ia, Ib) #don't detach, need it for autograd
	featureSet.features = computeFeatures_torch(I1, I2, I3, Ia, Ib)
	if(AD==True):
		def differentiateFeaturesWithInvariants(features,I):
			"""
			Compute derivatives of the features with respect to the invariants of the Cauchy-Green strain tensor using automatic differentiation.

			_Input Arguments_

			- `features` - features

			- `I` - invariant

			_Output Arguments_

			- `d_feature_dI` - derivative

			---

			"""
			d_feature_dI = torch.zeros(features.shape[0],features.shape[1])
			for i in range(features.shape[1]):
				temp = torch.autograd.grad(features[:,i:(i+1)],I,torch.ones(I.shape[0],1),create_graph=True,allow_unused=True)[0]
				# temp = torch.autograd.grad(features[:,i:(i+1)],I,torch.ones(I.shape[0],1),create_graph=True,allow_unused=True)[0]
				if(type(temp)!=type(None)):
					d_feature_dI[:,i:(i+1)] = temp
			return d_feature_dI
		featureSet.d_features_dI3 = differentiateFeaturesWithInvariants(featureSet.features,I3)
		featureSet.d_features_dI1 = differentiateFeaturesWithInvariants(featureSet.features,Ijunk)
		featureSet.d_features_dI2 = differentiateFeaturesWithInvariants(featureSet.features,I2)
		featureSet.d_features_dIa = differentiateFeaturesWithInvariants(featureSet.features,Ia)
		featureSet.d_features_dIb = differentiateFeaturesWithInvariants(featureSet.features,Ib)


		#detach features now:
		featureSet.features = featureSet.features.detach()
		featureSet.d_features_dI1 = featureSet.d_features_dI1.detach()
		featureSet.d_features_dI2 = featureSet.d_features_dI2.detach()
		featureSet.d_features_dI3 = featureSet.d_features_dI3.detach()
		featureSet.d_features_dIa = featureSet.d_features_dIa.detach()
		featureSet.d_features_dIb = featureSet.d_features_dIb.detach()

	#density
	if os.path.exists(path+'/output_density.csv'):
		df = pd.read_csv(path+'/output_density.csv',dtype=np.float64)
		density = df.density[0]
	else:
		print('\nWARNING: density file not found\n')
		density = 0.

	df = pd.read_csv(path+'/output_elements.csv',dtype=np.float64)

	connectivity = []
	for i in range(nodesperlelem):
		temp = df['node'+str(i+1)].round().astype(int)-1
		connectivity.append(torch.tensor(temp.tolist()))
	connectivityTotal = connectivity
	print(connectivityTotal[0].shape)
	for a in range(nodesperlelem):
		connectivityTotal[a]=connectivityTotal[a].repeat(quadpoints.shape[0])
	print(connectivityTotal[0].shape)
	# qpWeightsTotal = (torch.ones((numElements*(quadpoints.shape[0]),))*(np.linalg.norm(x_nodes[connectivity[0][0],:]-x_nodes[connectivity[1][0],:]))**3.0)/8.0
	# qpWeightsTotal = torch.tensor((np.linalg.norm(x_nodes[connectivity[0],:]-x_nodes[connectivity[1],:],axis=1))*(np.linalg.norm(x_nodes[connectivity[0],:]-x_nodes[connectivity[3],:],axis=1))*(np.linalg.norm(x_nodes[connectivity[0],:]-x_nodes[connectivity[4],:],axis=1)))/8.0
	qpWeightsTotal = torch.tensor(qpWeightsTotal.squeeze())
	lumped_mass = torch.zeros(numNodes, 3)

	# for e in range(numElements):
	# 	for a1 in range(nodesperlelem):
	# 		lumped_mass[connectivity[a1][e]][0] += density * qpWeights[e]/nodesperlelem
	# 		lumped_mass[connectivity[a1][e]][1] += density * qpWeights[e]/nodesperlelem
	# 		lumped_mass[connectivity[a1][e]][2] += density * qpWeights[e]/nodesperlelem

	# lumped mass_acceleration | size: numNodes x 2
	lumped_mass_acceleration = torch.mul(lumped_mass,a_nodes)

	# dataset = FemDataset(path,
	# 	x_nodes, u_nodes, v_nodes, a_nodes, dirichlet_nodes,
	# 	reactions,
	# 	connectivityTotal, gradNaTotal, NaTotal, qpWeightsTotal,
	# 	F, J, C,
	# 	I1, I2, I3, Ia, Ib,
	# 	dI1dF, dI2dF, dI3dF, dIadF, dIbdF,
	# 	featureSet,
	# 	density,
	# 	lumped_mass_acceleration)
	
	dataset = FemDataset(path,
		x_nodes, u_nodes, v_nodes, a_nodes, dirichlet_nodes,
		reactions,
		connectivityTotal, gradNaTotal, qpWeightsTotal,
		F, J, C,
		I1, I2, I3, Ia, Ib,
		dI1dF, dI2dF, dI3dF, dIadF, dIbdF,
		featureSet,
		density,
		lumped_mass_acceleration)

	print('-----------------------------------------------------\n')

	return dataset
