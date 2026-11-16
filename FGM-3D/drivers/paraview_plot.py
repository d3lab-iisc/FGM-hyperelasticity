import os
import pandas as pd
import pyvista as pv
import numpy as np
from models import get_model


def runparaview_hex(main_folder, num_steps, theta=None, params=None):

	# Parameter names in theta order (default: active model in models.py)
	if params is None:
		params = get_model()['params']
	numFeatures = len(params)

	# Hex element cell type for VTK
	vtk_hex_type = 12

	# create a folder named 'results' if it doesn't exist
	results_dir = os.path.join(main_folder, 'results')
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)

	# Loop through each step folder
	for step in range(num_steps, num_steps + 1):
		step_folder = os.path.join(main_folder, str(step))
		
		# Load nodal data
		nodal_data = pd.read_csv(f'{main_folder}/{step}/output_nodes.csv')
		
		# Load connectivity data (only required node columns)
		connectivity_file = os.path.join(step_folder, 'output_elements.csv')
		connectivity_data = pd.read_csv(connectivity_file, usecols=['node1', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7', 'node8'])

		# Ground truth (only the columns that exist in the file)
		connectivity_file2 = os.path.join(step_folder, 'output_elements_matProperty.csv')
		connectivity_data2 = pd.read_csv(connectivity_file2)

		# Predicted parameters, interleaved per element: [p1, p2, ..., p1, p2, ...]
		idxs = np.arange(theta.shape[0]//numFeatures)
		theta_full = {p: theta[numFeatures*idxs + j] for j, p in enumerate(params)}

		# Create PyVista Mesh for current step
		points = nodal_data[['x', 'y', 'z']].values
		# Ensure points array is consistent
		if len(points) != len(nodal_data):
			print(f"Error: Mismatch in points array length for step {step}.")
			continue

		# Construct cell array: flatten connectivity and prepend the number of points per cell
		num_points_per_cell = 8	 # Hex elements have 8 nodes

		connectivity_data = connectivity_data.values.astype(int) - 1
		cells = np.hstack([[num_points_per_cell] + list(connectivity_data[i]) for i in range(connectivity_data.shape[0])])
		# Create a cell types array (same type for each cell)
		cell_types = np.full(connectivity_data.shape[0], vtk_hex_type, dtype=np.uint8)

		# Check sizes
		print("\n<<< Paraview Visualization >>>")
		print("Number of cells:", connectivity_data.shape[0])
		print("Number of points:", points.shape[0])
		if connectivity_data.shape[0] > 0:
			try:
				grid = pv.UnstructuredGrid(cells, cell_types, points)

				# Add displacement vectors to the mesh
				displacements = nodal_data[['ux', 'uy', 'uz']].values
				grid.point_data['Displacement'] = displacements
				
				for p in params:
					# PREDICTION
					grid.cell_data[f'P_{p}'] = theta_full[p]

					# GROUND TRUTH and ERROR
					if p in connectivity_data2.columns:
						grid.cell_data[f'GT_{p}'] = connectivity_data2[p].values
						grid.cell_data[f'error_{p}_abs'] = np.abs(connectivity_data2[p].values - theta_full[p])
					else:
						print(f"No ground truth column '{p}' in {connectivity_file2}, skipping GT/error.")

				grid.save(f'{results_dir}/paraview_output.vtk')
				
			except Exception as inst:
				print(f"Error creating UnstructuredGrid for step {step}: {inst}")
				
		else:
			print(f"No cells found for step {step}. Skipping...")

def runparaview_noise(step_folder, num_steps=6, noising_term=None, denoised_noise_term=None, u_denoised=None):

	# Hex element cell type for VTK
	vtk_hex_type = 12
	
	# Load nodal data
	nodal_data = pd.read_csv(f'{step_folder}/output_nodes.csv')
	
	# Load connectivity data (only required node columns)
	connectivity_file = os.path.join(step_folder, 'output_elements.csv')
	connectivity_data = pd.read_csv(connectivity_file, usecols=['node1', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7', 'node8'])

	# Create PyVista Mesh for current step
	points = nodal_data[['x', 'y', 'z']].values
	# Ensure points array is consistent
	if len(points) != len(nodal_data):
		print(f"Error: Mismatch in points array length for step {step_folder}.")

	# Construct cell array: flatten connectivity and prepend the number of points per cell
	num_points_per_cell = 8	 # Hex elements have 8 nodes

	connectivity_data = connectivity_data.values.astype(int) - 1
	cells = np.hstack([[num_points_per_cell] + list(connectivity_data[i]) for i in range(connectivity_data.shape[0])])
	# Create a cell types array (same type for each cell)
	cell_types = np.full(connectivity_data.shape[0], vtk_hex_type, dtype=np.uint8)

	# Check sizes
	print("\n<<< Paraview Visualization >>>")
	print(f"Step {step_folder}")
	print("Number of cells:", connectivity_data.shape[0])
	print("Number of points:", points.shape[0])
	if connectivity_data.shape[0] > 0:
		try:
			grid = pv.UnstructuredGrid(cells, cell_types, points)
			displacements = nodal_data[['ux', 'uy', 'uz']].values
			grid.point_data['noising_term'] = noising_term
			grid.point_data['Displacement'] = displacements
			grid.point_data['Noisy_Disp.'] = displacements + noising_term
			# grid.point_data['Denoised_Disp.'] = u_denoised
			# grid.point_data['Denoised_Noise'] = denoised_noise_term
			grid.save(f'{step_folder}/step_output_noise.vtk')
			
		except Exception as inst:
			print(f"Error creating UnstructuredGrid for step {step_folder}: {inst}")
			
	else:
		print(f"No cells found for step {step_folder}. Skipping...")

if __name__=='__main__':
	# runparaview_hex needs the predicted theta, it is called from main_fgm_HEX.py
	print("Usage: called from main_fgm_HEX.py, e.g. runparaview_hex(main_folder, num_steps, theta=theta, params=['C10','inv_D1'])")

