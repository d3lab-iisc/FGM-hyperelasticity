import os
import sys
sys.path.insert(1, '../drivers/')
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import r2_score
matplotlib.pyplot.rcParams["font.family"] = "serif"
matplotlib.pyplot.rcParams["mathtext.fontset"] = "dejavuserif"
from unsupervised_hyperelasticity import *
import numpy as np
import seaborn as sns
import pandas as pd
from features_library import *
from models import set_model


def groundtruth_materialwise(fem_mat, params, n_modes, loadstep=6):
	"""
	Ground-truth parameters per material, shape (n_modes, numFeatures), sorted by the first parameter.
	Only possible for piecewise-constant datasets with exactly n_modes distinct materials
	(e.g. CylinderinCube_NeoHookean). For graded datasets there is no unique "material"
	ground truth, so it must be given explicitly (theta_gt argument / command line).
	"""
	groundtruth_file = os.path.join('../main_data', fem_mat, str(loadstep), 'output_elements_matProperty.csv')
	groundtruth_elements_data = pd.read_csv(groundtruth_file)

	missing_gt = [name for name in params if name not in groundtruth_elements_data.columns]
	if missing_gt:
		raise ValueError(f"No ground truth columns {missing_gt} in {groundtruth_file}")

	materials = np.unique(groundtruth_elements_data[params].values, axis=0)
	if materials.shape[0] != n_modes:
		raise ValueError(
			f"{fem_mat} has {materials.shape[0]} distinct ground-truth parameter sets (graded material), "
			f"not {n_modes} distinct materials. Give the ground truth of the plotted material explicitly, e.g.\n"
			f"    python post_process.py {fem_mat} <model> <mode_id> {','.join(['<'+p+'>' for p in params])}")
	return materials[np.argsort(materials[:, 0])]


def post_proc_short(fem_mat, model_name, mode_id=None, theta_gt=None):
	"""
	Energy-path plots W(gamma) for the materials found by plot_histogram_gmm.

	_Input Arguments_

	- `fem_mat`    - dataset folder in ../main_data

	- `model_name` - model in models.py (NH, MR, ISH, HGO, ...)

	- `mode_id`    - which GMM material to plot (0-based):
	                 None      -> one figure per material,
	                 int       -> the same material for every parameter,
	                 list      -> one material per parameter, e.g. [0, 0, 2] for (C10, inv_D1, k1)

	- `theta_gt`   - ground truth of the plotted material(s), shape (numFeatures,) or
	                 (number of figures, numFeatures), e.g. [2.0, 20.0, 2.0] for HGO;
	                 None -> read from the dataset (only for piecewise-constant datasets, see groundtruth_materialwise)

	"""
	model = set_model(model_name)
	params = model['params']

	results_dir = os.path.join('../main_data', fem_mat, 'results')
	wide_csv = os.path.join(results_dir, "theta_mean_variance_materialwise_wide.csv")
	if not os.path.exists(wide_csv):
		raise FileNotFoundError(f"{wide_csv} not found. Run the driver with 'gmm_modes' set for model '{model_name}' in models.py.")
	df = pd.read_csv(wide_csv)
	df = df.sort_values("material_id").reset_index(drop=True)

	num_feats= getNumberOfFeatures()
	n_modes = len(df)

	field_names = params

	theta_mean = np.zeros((n_modes, num_feats), dtype=np.float64)
	theta_variance = np.zeros((n_modes, num_feats), dtype=np.float64)

	for j, name in enumerate(field_names):
		theta_mean[:, j] = df[f"{name}_mean"].to_numpy()
		theta_variance[:, j] = df[f"{name}_variance"].to_numpy()

	print("Parameters:", params)
	print("theta_mean (rows = materials):\n", theta_mean)
	print("theta_variance (rows = materials):\n", theta_variance)

	# Which material(s) to plot: list of (file tag, material index per parameter)
	if mode_id is None:
		selections = [(f"_material{m+1}", [m]*num_feats) for m in range(n_modes)]
	elif np.isscalar(mode_id):
		selections = [("", [int(mode_id)]*num_feats)]
	else:
		selections = [("", [int(m) for m in mode_id])]
		if len(selections[0][1]) != num_feats:
			raise ValueError(f"mode_id needs one entry per parameter {params}, got {mode_id}")

	cols = np.arange(num_feats)

	# Ground truth, one row per figure
	if theta_gt is None:
		theta_gt_all = groundtruth_materialwise(fem_mat, params, n_modes)
		theta_gt = np.array([theta_gt_all[ids, cols] for tag, ids in selections])
	theta_gt = np.atleast_2d(np.asarray(theta_gt, dtype=np.float64))
	if theta_gt.shape != (len(selections), num_feats):
		raise ValueError(f"theta_gt must have shape ({len(selections)}, {num_feats}) for parameters {params}, got {theta_gt.shape}")
	print("theta_gt (rows = figures):\n", theta_gt)

	for (tag, ids), gt in zip(selections, theta_gt):
		plot_energy_paths(results_dir, fig_title=model_name, file_tag=tag,
						  theta_mean=theta_mean[ids, cols],
						  theta_variance=theta_variance[ids, cols],
						  theta_gt=gt)


def plot_energy_paths(results_dir, fig_title, file_tag, theta_mean, theta_variance, theta_gt):

	fig = plt.figure(figsize=(4, 24))

	gs = GridSpec(6, 1, wspace=0.4, hspace=0.5)

	axes = []
	axes.append(fig.add_subplot(gs[0,0]))
	axes.append(fig.add_subplot(gs[1,0]))
	axes.append(fig.add_subplot(gs[2,0]))
	axes.append(fig.add_subplot(gs[3,0]))
	axes.append(fig.add_subplot(gs[4,0]))
	axes.append(fig.add_subplot(gs[5,0]))

	# plt.tight_layout(pad=6)
	plt.rcParams.update({'font.size':12})
	fig.suptitle(fig_title, fontsize=21, x=0.5, y=0.92)

	valid = ['(i)','(ii)','(iii)','(iv)','(v)','(vi)']

	def energy_plot(id, deformation):
		gamma, W_mean, W_plus, W_minus, W_gt = predict_energy_path_FGM(theta_mean, theta_variance, theta_gt, deformation)

		# GT
		sns.lineplot(x=gamma, y=W_gt,  ax=axes[id], color='red', label='True')
		# # mean
		sns.lineplot(x=gamma, y=W_mean,	 ax=axes[id], color='black', label='Mean')
		R2 = r2_score(W_gt,W_mean)
		temp = str(R2)
		axes[id].fill_between(gamma, W_minus, W_plus, color='silver', label='95-perc.')
		#labels
		axes[id].set_xlabel(r'Deformation ($\gamma$)',fontsize=16)
		axes[id].set_ylabel(r'W($\gamma$)',fontsize=16)
		label = valid[id]
		title = deformation.title().replace("_", " ")
		axes[id].set_title(f"{label} " + title, fontsize=14, fontweight='bold')
		# axes[id].set_title(valid[id] + ' ' + deformation.title().replace("_"," "), fontsize=16)
		axes[id].tick_params(axis='x',labelsize =12)
		axes[id].tick_params(axis='y',labelsize =12)
		axes[id].set_ylim(0,1.3*np.max(W_gt))
		handles, labels = axes[id].get_legend_handles_labels()
		axes[id].legend(handles=[handles[0],handles[1],handles[2]],
			labels=[labels[0],labels[1],labels[2]],
			fontsize=14, ncol=1, loc=2, facecolor='white', framealpha=0.,borderpad=0.,labelspacing=0.2,handlelength=0.7)
		axes[id].text(0.24,0.54,'R$^2$ = '+temp[0:5],ha='center',va='center',transform=axes[id].transAxes,zorder=2,fontsize=13)
		axes[id].set_aspect('auto')  # OR try 'equal' if Y range allows

	energy_plot(0,'tension')
	energy_plot(1,'simple_shear')
	energy_plot(2,'pure_shear')
	energy_plot(3,'biaxial_tension')
	energy_plot(4,'compression')
	energy_plot(5,'biaxial_compression')


	if results_dir is not None:
		if(results_dir[-1]=='/'):
			results_dir = results_dir[0:-1]
		# plt.savefig(f"{results_dir}/W_{fig_title}{file_tag}.pdf", bbox_inches='tight')
		plt.savefig(f"{results_dir}/W_{fig_title}{file_tag}.png", bbox_inches='tight', dpi=300)

		svg_dir = os.path.join(results_dir, "svg")
		os.makedirs(svg_dir, exist_ok=True)

		plt.savefig(f"{svg_dir}/W_{fig_title}{file_tag}.svg", bbox_inches='tight')
		print("Post-process completed")
	plt.close(fig)


if __name__ == "__main__":
    # Usage: python post_process.py <material_name> <model> [mode_id] [theta_gt]
    #   mode_id : omitted (all materials), "1" (same material for all parameters) or "0,0,1" (per parameter), 0-based
    #   theta_gt: ground truth of the plotted material, e.g. "2,20,2" for HGO (C10, inv_D1, k1);
    #             omitted -> read from the dataset (piecewise-constant datasets only)
    # Example: python post_process.py GradedSpherical_Holzapfel HGO 0,0,1 2,20,2
    if len(sys.argv) < 3:
        raise ValueError("Usage: python post_process.py <material_name> <model> [mode_id] [theta_gt]")
    FEM_mat = sys.argv[1]
    model_name = sys.argv[2]
    mode_id = None
    theta_gt = None
    if len(sys.argv) >= 4:
        mode_id = [int(m) for m in sys.argv[3].split(',')]
        if len(mode_id) == 1:
            mode_id = mode_id[0]
    if len(sys.argv) >= 5:
        theta_gt = [float(v) for v in sys.argv[4].split(',')]
    post_proc_short(FEM_mat, model_name, mode_id, theta_gt)
