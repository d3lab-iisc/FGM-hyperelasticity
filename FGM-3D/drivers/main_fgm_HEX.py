import os
import gc
import sys
import platform

sys.path.insert(1, '../core/')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from unsupervised_hyperelasticity import *
from features_library import *
from post_process import *
from helper import *
from config import *
from models import MODELS, set_model, get_latex_label
from preprocess_data_hetero import extractSystemOfEquations

# torch
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

cpu_info = platform.processor()
if torch.cuda.is_available():
    gpu_info = torch.cuda.get_device_name(0)
else:
    gpu_info = "No CUDA GPU detected"

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
# torch.autograd.set_detect_anomaly(True)   # debugging only: slows training, results are identical

# numpy
np.random.seed(0)

# others:
import shutil

# custom:
from data_definitions import *
# from load_fem_data_brick_reduced import *
from load_fem_data_brick_Full import *
from utilities import *
from paraview_plot import*
from visuals_plot import *
import time
from datetime import datetime

processing_start = time.time()

results = []
fem_dir = '../main_data'

# Map noise level strings to numerical values
noise_map = {
	'none': 0,
	'dic': 5e-4,
	'high': 1e-3,
	'custom': 2e-3,
}

def collect_result(result):
	global results
	arr, ind = result
	results[ind]=arr

def main(model_name=None):
	"""
	Unsupervised discovery of the spatially varying parameters of a hyperelastic model (FGM, C3D8 mesh).

	Usage:
		python main_fgm_HEX.py <model> <material_name> <noise_level>

	<model>       : one of the models in models.py (NH, MR, ISH, HGO, ...)
	<noise_level> : one of noise_map (none, DIC, high, custom)
	"""
	rng = np.random.default_rng(seed)
	show_current_time()

	prefix = '../drivers'

	# Default values
	fem_material = None
	noiseLevel = 0.0

	# -------------------------------
	# Parse command-line arguments
	# -------------------------------
	args = sys.argv[1:]
	if model_name is None:
		if len(args) < 3:
			raise ValueError(f"Usage: python main_fgm_HEX.py <model> <material_name> <noise_level>, <model> one of {list(MODELS.keys())}")
		model_name = args[0]
		args = args[1:]

	if len(args) >= 2:
		fem_material = args[0]
		noise_level = args[1]
		noise_str = args[1].lower()
		noise_map_lower = {k.lower(): v for k, v in noise_map.items()}	# case-insensitive: 'DIC' or 'dic'

		if noise_str in noise_map_lower:
			noiseLevel = noise_map_lower[noise_str]
		else:
			raise ValueError(f"Unknown noise level '{noise_str}'. Choose from: {list(noise_map.keys())}")
	else:
		raise ValueError(f"Usage: python main_fgm_HEX.py <model> <material_name> <noise_level>, <model> one of {list(MODELS.keys())}")

	# -------------------------------
	# Model selection (models.py)
	# -------------------------------
	model = set_model(model_name)
	params = model['params']			# e.g. ['C10', 'inv_D1'] for NH
	numFeatures = len(params)

	if not os.path.isdir(os.path.join(fem_dir, fem_material)):
		raise FileNotFoundError(f"Dataset '{fem_material}' not found in {fem_dir}. Available: {sorted(os.listdir(fem_dir))}")

	# ----------------------------------------------------
	# Create noised copy if noiseLevel is not zero
	# ----------------------------------------------------

	if noiseLevel != 0.0:
		noised_material = f"{fem_material}_CORR_noised_{noiseLevel:.0e}".replace("+0", "")
		src_dir = os.path.join(fem_dir, fem_material)
		dst_dir = os.path.join(fem_dir, noised_material)

		if not os.path.exists(dst_dir):
			shutil.copytree(src_dir, dst_dir)
			print(f"Created correlated noised folder: {dst_dir}")
		else:
			print(f"Noised folder already exists: {dst_dir}")

		# Redirect to noised folder for all processing
		fem_material = noised_material

	# -------------------------------------------------------------------------------------------------------------------------------------------------
	# Data preprocessing and system assembly
	# -------------------------------------------------------------------------------------------------------------------------------------------------
	fem_dir_pp = fem_dir+'/'+fem_material
	print("fem_dir_pp:", fem_dir_pp)

	loadstep = 6
	X_elem2 = pd.read_csv(os.path.join(fem_dir, fem_material, str(6), 'output_elements.csv')) # Read as a DataFrame
	X_elem = X_elem2['node1'].to_numpy()	 # Extract 'node1' column and convert it to a NumPy array
	step_folder = f'{fem_dir_pp}/{str(loadstep)}'
	elements_file = os.path.join(step_folder, 'output_elements.csv')
	elements_data = pd.read_csv(elements_file)
	element_connectivity = elements_data[['node1', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7', 'node8']].values
	connectivity = element_connectivity - 1  # Convert to zero-based indexing

	cluster_values = np.arange(X_elem.shape[0])+1

#-----------------------------------------------------------
	# Ground truth theta extraction
#-----------------------------------------------------------
	step_folder = f'{fem_dir_pp}/{str(loadstep)}'

	groundtruth_file = os.path.join(step_folder, 'output_elements_matProperty.csv')
	groundtruth_elements_data = pd.read_csv(groundtruth_file)

	missing_gt = [p for p in params if p not in groundtruth_elements_data.columns]
	if missing_gt:
		raise ValueError(f"Model '{model_name}' needs ground-truth columns {params} in {groundtruth_file}, "
						 f"missing: {missing_gt}. Is '{fem_material}' a dataset for the {model['label']} model?")

	# theta_gt_param['C10'], theta_gt_param['inv_D1'], ...
	theta_gt_param = {p: groundtruth_elements_data[p].values for p in params}

	theta_gt = np.column_stack([theta_gt_param[p] for p in params]).reshape(-1)

	# create a folder named 'results' if it doesn't exist
	results_dir = os.path.join(fem_dir_pp, 'results')
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)

	X_nodes = pd.read_csv(os.path.join(fem_dir, fem_material, str(6), 'output_nodes.csv'))
	node_coords = X_nodes[['x', 'y', 'z']].values

	def compute_elem_length_C3D8(node_coords, connectivity):

		elem_nodes = node_coords[connectivity]
		# 5-tet decomposition indices
		tets = np.array([
			[0,1,3,4],
			[1,2,3,6],
			[1,4,5,6],
			[3,4,6,7],
			[1,3,4,6]
		])

		total_volume = 0.0
		for tet in tets:
			n1 = elem_nodes[:, tet[0], :]
			n2 = elem_nodes[:, tet[1], :]
			n3 = elem_nodes[:, tet[2], :]
			n4 = elem_nodes[:, tet[3], :]

			v1 = n2 - n1
			v2 = n3 - n1
			v3 = n4 - n1

			det = np.einsum('ij,ij->i',v1,np.cross(v2, v3))
			total_volume += np.sum(np.abs(det) / 6.0)

		n_elements = connectivity.shape[0]
		avg_volume = total_volume / n_elements
		print("total_volume:", total_volume)
		print("number of elements:", n_elements)
		element_length = avg_volume ** (1.0 / 3.0)

		return element_length

	elem_length = compute_elem_length_C3D8(node_coords, connectivity)

	print(f"Element length scale: {elem_length:.4f}")

	loadsteps_list = model['loadsteps_list']

	def build_G_sum(loadsteps_list):
		G_sum = None
		for step in loadsteps_list:

			print(f"Loadstep [{step}]")
			step_folder = os.path.join(fem_dir_pp, str(step))
			G_step_file = os.path.join(step_folder, 'G_step.pt')
			if os.path.exists(G_step_file):
				print(f"Reading G_step for loadstep {step}")
				G_step = torch.load(G_step_file, map_location=device)
			else:
				print("Extracting system of equations for loadstep", step)
				# A1 (free-dof equations) is returned in memory, no A1.csv / features.csv is written
				A_free = extractSystemOfEquations(fem_dir_pp, [step], cluster_values.copy(), noiseLevel)[0]
				print("System of equations extracted.")

				print("Processing FEM data for G matrix assembly...")
				A_free_gpu = A_free.to(device)

				G_step = A_free_gpu.T @ A_free_gpu
				print("Saving G_step matrix...")
				torch.save(G_step.clone().detach(), G_step_file)

				del A_free, A_free_gpu

			show_current_time()

			if G_sum is None:
				G_sum = G_step
			else:
				G_sum += G_step
			gc.collect()
			torch.cuda.empty_cache()

		return G_sum

	G_sum = build_G_sum(loadsteps_list)
	print(f"All Loadstep Processed. G_sum is on {device} with shape {G_sum.shape}")

	def build_Sigma_matrix(node_coords, connectivity, length_scale, jitter=1e-6):
		# --- Compute Centroids for ALL elements ---
		# Centroid = average of node coordinates for that element
		# connectivity shape: (M, 6) -> We gather coords -> (M, 6, 3) -> mean -> (M, 3)
		elem_coords = node_coords[connectivity]
		centroids = elem_coords.mean(axis=1) # Shape: (Total_Elems, 3)

		# Convert to Torch for distance calculation
		X = torch.tensor(centroids, dtype=torch.float64, device=device)

		# --- Compute Distance Matrix ---
		# dist_matrix[i, j] = Euclidean distance between centroid i and j
		dist_matrix = torch.cdist(X, X, p=2)
		Sigma = torch.exp(- (dist_matrix**2) / (2 * length_scale**2))
		I = torch.eye(Sigma.shape[0], dtype=torch.float64, device=device)
		Sigma = Sigma + jitter * I

		print("Inverting Covariance Matrix...")
		Sigma_inv = torch.linalg.inv(Sigma)

		return Sigma_inv

	# Set parameters (per model, see models.py)
	ls_factor = model['ls_factor']
	length_scale = ls_factor * elem_length
	lambda_gp = model['lambda_gp']		# GP regularization weight (original 1e-4)
	lambda_reg = model['lambda_reg']	# regularization weight (original 1e-5)
	eps = 1e-06

	Sigma_inv = build_Sigma_matrix(
		node_coords,
		connectivity,
		length_scale=length_scale
	)

	print(f"Sigma Matrix Shape: {Sigma_inv.shape}")

	def optimize_FGM(model_name=None,no_epochs=10000,lr=0.005):
		# Initialize alpha
		if model_name is not None:
			print(f"Loading initial model from {model_name}")
			alpha = torch.load(os.path.join(fem_dir_pp, f'{model_name}.pt'), map_location=device)
		else:
			alpha = torch.randn(G_sum.shape[0], 1, dtype=torch.float64, device=device, requires_grad=True)

		alpha.requires_grad=True

		# Optimizer
		optimizer = torch.optim.Adam([alpha], lr=lr, betas=(0.95, 0.999),
									eps=1e-08, weight_decay=0, amsgrad=False)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=no_epochs, eta_min=1e-6)

		means = torch.zeros_like(alpha[0::numFeatures]).to(device)

		history = {
        "epoch": [],
        "lr": [],
        "loss_total": [],
        "loss_main": [],
        "loss_reg": [],
        "loss_smooth": []
    	}

		for step in range(no_epochs):
			optimizer.zero_grad()

			# Reparameterization for positivity
			x = alpha**2 + eps

			# Term 1: x^T G x
			term_main = x.T @ G_sum @ x

			# Separate the interleaved vectors
			# x_param[params[0]] corresponds to [x1, x1+numFeatures, ...] (indices 0, numFeatures, ...)
			# x_param[params[1]] corresponds to indices 1, 1+numFeatures, ... and so on
			x_param = {p: x[j::numFeatures] for j, p in enumerate(params)}

			# Regularization on the first parameter (C10)
			term_reg = lambda_reg * torch.sum(1.0 / (x_param[params[0]] + eps))

			# Calculate x^T * Sigma_inv * x for every parameter
			# This is the "Mahalanobis Distance" from the mean prior
			loss_smooth = 0.0
			for p in params:
				loss_smooth = loss_smooth + (x_param[p]-means).T @ Sigma_inv @ (x_param[p]-means)
			term_smooth = lambda_gp * loss_smooth

			# Total Loss
			loss = term_main + term_reg + term_smooth

			# Backprop and optimization step
			loss.backward(retain_graph=True)

			# Save scalar history before optimizer step
			history["epoch"].append(step)
			history["lr"].append(optimizer.param_groups[0]["lr"])
			history["loss_total"].append(loss.item())
			history["loss_main"].append(term_main.item())
			history["loss_reg"].append(term_reg.item())
			history["loss_smooth"].append(term_smooth.item())

			optimizer.step()
			scheduler.step()

			if step % model['log_every'] == 0:
				# Get current LR for logging
				current_lr = optimizer.param_groups[0]['lr']
				print(f"Step {step}, Loss: {loss.item():.6f}, LR: {current_lr:.1e} | "
					f"Main: {term_main.item():.6f}, Reg: {term_reg.item():.6f}, Smooth: {term_smooth.item():.6f}")

			# if step % 10000 == 0 and step > 0:
			# 	torch.save(alpha.clone().detach(), os.path.join(fem_dir_pp, f'model_fgm_hex_step{step}.pt'))

		x_opt = (alpha**2).detach()

		# Save CSV
		hist_df = pd.DataFrame(history)
		csv_path = os.path.join(results_dir, "loss_history.csv")
		hist_df.to_csv(csv_path, index=False)

		# Plot total loss only
		fig, ax = plt.subplots(figsize=(10, 6))
		ax.plot(hist_df["epoch"], hist_df["loss_total"], label="Total Loss", linewidth=2)
		ax.set_xlabel("Epoch")
		ax.set_ylabel("Total Loss")
		ax.set_title("Total Loss vs Epochs")
		ax.set_yscale("log")
		ax.legend()
		ax.grid(True, alpha=0.3)
		plt.tight_layout()
		fig.savefig(os.path.join(results_dir, "total_loss_vs_epochs.png"), dpi=300)
		plt.close(fig)

		print(f"Saved loss CSV to: {csv_path}")
		print(f"Saved plots to: {results_dir}")

		return x_opt, alpha.detach()

	processing_end = time.time()

	optimization_start = time.time()

	x_opt, alpha = optimize_FGM(model_name=None, no_epochs=model['no_epochs'], lr=model['lr'])
	print("Optimization completed.")
	model_fgm = alpha.clone().detach()
	torch.save(model_fgm, os.path.join(fem_dir_pp, 'model_fgm_hex.pt'))
	print("All components strictly > 0:", bool(torch.all(x_opt > 0)))

	## Post-processing
	# alpha = torch.load(os.path.join(fem_dir_pp, f'model_fgm_hex.pt'), map_location=device)
	# x_opt = (alpha**2).detach()

	theta = x_opt.cpu().numpy()
	optimization_end = time.time()


	def calculate_elasto_factor(theta):
		print("\n<<< Calculating Elasto Factor >>>")
		all_elasto_factor = []   # to store for all loadsteps

		for step in loadsteps_list:
			A_fix = pd.read_csv(f'{fem_dir_pp}/{str(step)}/A2.csv', header=None).to_numpy()
			b_fix = pd.read_csv(f'{fem_dir_pp}/{str(step)}/b2.csv', header=None).to_numpy()


			if step == 1:
				k_index = np.array([1,2,3,4]) - 1   # Biaxial Tension XY
			elif step == 2:
				k_index = np.array([3,4,5,6]) - 1   # Biaxial Tension XZ
			elif step == 3:
				k_index = np.array([1,2,5,6]) - 1   # Biaxial Tension YZ
			elif step == 4:
				k_index = np.array([3,4]) - 1  		# Uniaxial Tension X
			elif step == 5:
				k_index = np.array([1,2]) - 1  		# Uniaxial Tension Y
			elif step == 6:
				k_index = np.array([5,6]) - 1  		# Uniaxial Tension Z
			else:
				print("Invalid loadstep for elasto factor calculation.")
				# k_index = np.array([1,2,3,4,5,6]) - 1   # Triaxial Tension Case DOF-based

			elasto_factor = b_fix[k_index] / (A_fix[k_index, :] @ theta)
			elasto_factor = elasto_factor.flatten()
			print(f"Loadstep {step}, Elasto factors for k={k_index+1}: "f"[{', '.join(f'{val:.4f}' for val in elasto_factor)}]")
			elasto_factor =np.mean(elasto_factor)
			print(f"Loadstep {step}, Mean Elasto factor: {elasto_factor:.4f}")
			all_elasto_factor.append(elasto_factor)

		all_elasto_factor = np.array(all_elasto_factor)
		mean_elasto_factor = np.mean(all_elasto_factor)
		print(f"\nMean Elasto Factor across Loadsteps {loadsteps_list}: {mean_elasto_factor:.4f}")
		return mean_elasto_factor


	mean_elasto_factor = calculate_elasto_factor(theta)
	theta = theta*mean_elasto_factor
	theta = theta.flatten()

	# theta_param['C10'], theta_param['inv_D1'], ...
	theta_param = {p: theta[j::numFeatures] for j, p in enumerate(params)}

	for step in loadsteps_list:
		step_folder = os.path.join(fem_dir_pp, str(step))

		if os.path.exists(step_folder):
			elements_path = os.path.join(step_folder, 'output_elements_matProperty.csv')
			elements_df = pd.read_csv(elements_path)
			for p in params:
				elements_df[f'{p}_pred'] = theta_param[p]

			output_path = os.path.join(step_folder, 'output_elements_matProperty.csv')
			elements_df.to_csv(output_path, index=False)
			print(f"Prediction saved to: {output_path}")
		else:
			print(f"Step folder missing: {step_folder}")

	runparaview_hex(main_folder=fem_dir_pp,num_steps=len(loadsteps_list), theta=theta, params=params)

	if model['gmm_modes'] is not None:
		plot_histogram_gmm(theta, results_dir, bins=50, n_modes=model['gmm_modes'], params=params)

	processing_time = processing_end - processing_start
	optimization_time = optimization_end - optimization_start
	computational_time = processing_time + optimization_time
	print(f"\n{'Process':<22} | {'Seconds':>10} | {'Minutes':>10}")
	print("-" * 50)
	print(f"{'Processing Time':<22} | {processing_time:>10.2f} | {processing_time/60:>10.2f}")
	print(f"{'Optimization Time':<22} | {optimization_time:>10.2f} | {optimization_time/60:>10.2f}")
	print(f"{'Total Compute Time':<22} | {computational_time:>10.2f} | {computational_time/60:>10.2f}")

	show_current_time()
	# Saving info.txt
	with open(os.path.join(results_dir, 'info.txt'), 'w') as f:
			f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			f.write(f"CPU: {cpu_info}\n")
			f.write(f"GPU: {gpu_info}\n")
			f.write(f"------------------------------------------------------------\n")
			f.write(f"Material Type: {fem_material}\n")
			f.write(f"Model: {model_name} ({model['label']}), parameters: {params}\n")
			f.write(f"Noise Level: {noiseLevel:.0e}\n")
			f.write(f"C3D8 Mesh Elements: {connectivity.shape[0]}\n")
			f.write(f"------------------------------------------------------------\n")
			f.write(f"Number of Features: {getNumberOfFeatures()}\n")
			f.write(f"Loadsteps processed: {loadsteps_list}\n")
			f.write(f"Epochs: {model['no_epochs']}, lr: {model['lr']}\n")
			f.write(f"lambda_gp: {lambda_gp:.0e}\n")
			f.write(f"lambda_reg: {lambda_reg:.0e}\n")
			f.write(f"Element length: {elem_length:.4f}\n")
			f.write(f"Length scale factor: {ls_factor:.2f}\n")
			f.write(f"Mean Elasto Factor: {mean_elasto_factor:.4f}\n")
			f.write(f"------------------------------------------------------------\n")
			f.write(f"Processing time: {processing_time:.2f} seconds or {processing_time/60:.2f} minutes\n")
			f.write(f"Optimization time: {optimization_time:.2f} seconds or {optimization_time/60:.2f} minutes\n")
			f.write(f"Total Compute time: {computational_time:.2f} seconds or {computational_time/60:.2f} minutes\n")

	#-------------------------------
	# Plots for every parameter: theta_C10 (I1-3), theta_inv_D1 (J-1)^2, ...
	#-------------------------------
	for p in params:
		theta_label = get_latex_label(p)		# e.g. r"$\theta_{C10}$"

		plot_histogram_comparison(
			results_dir=results_dir,
			theta=theta_param[p],
			theta_gt=theta_gt_param[p],
			legends_name=(rf"Pred, {theta_label}", rf"GT, {theta_label}"),
			xlabel=rf"{theta_label} (MPa)",
			ylabel="Density",
			filename=f"hist_{p}"
			)


		plot_hexbin_parity(
			results_dir=results_dir,
			theta=theta_param[p],
			theta_gt=theta_gt_param[p],
			legends_name=("1:1", "Hexbin count"),
			xlabel=rf"GT, {theta_label} (MPa)",
			ylabel=rf"Pred, {theta_label} (MPa)",
			filename=f"hexbin_{p}"
		)

if __name__=='__main__':
	main()
