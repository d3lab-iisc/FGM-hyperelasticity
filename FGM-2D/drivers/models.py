import torch

#=====================================================================
# MODEL LIBRARY (2D, in-plane specimens)
#=====================================================================
# Each model is defined ONCE here. The driver, feature library, ParaView
# export, GMM histograms and post-processing all read from this table.
#
# Every model has the form  W = sum_i theta_i * feature_i(K1, K2, J, Iat, Ibt)
#   K1  = I1 * I3^(-1/3) - 3          (isochoric 1st invariant)
#   K2  = I2 * I3^(-2/3) - 3          (isochoric 2nd invariant)
#   J   = sqrt(I3)
#   Iat = Ia * J^(-2/3)               (isochoric fibre invariant, fibre a in config.py)
#   Ibt = Ib * J^(-2/3)               (isochoric fibre invariant, fibre b in config.py)
# The 2D kinematics (F33 = 1/det(F_2D), F13 = F23 = F31 = F32 = 0) are applied in
# load_fem_data_brick_Full.py and preprocess_data_hetero.py, not here.
#
# To add a model:
#   1. write a features_<NAME>(K1, K2, J, Iat, Ibt) function returning a list of features
#   2. add an entry to MODELS with 'params' in the SAME order as the features
#   3. for synthetic data, output_elements_matProperty.csv needs ground-truth columns named as 'params'
#   4. run: python main_fgm_HEX.py <NAME> <material_name> <noise_level>
#=====================================================================


def features_NH(K1, K2, J, Iat, Ibt):
    # Neo-Hookean (incompressible): C10
    return [K1]


# Settings shared by all models; a model entry only lists what differs.
DEFAULTS = {
    'no_epochs':      60001,
    'lr':             0.005,
    'ls_factor':      3.0,                  # GP length scale = ls_factor * element length
    'lambda_gp':      1e-4,                 # GP smoothness weight
    'lambda_reg':     1e-5,                 # regularization weight on the first parameter
    'loadsteps_list': None,                 # None = all load step folders found in the dataset
    'log_every':      500,                  # print/log interval (epochs)
    'gmm_modes':      None,                 # number of GMM modes for plot_histogram_gmm (None = skip)
}

MODELS = {
    'NH': {
        'label':     'Neo-Hookean',
        'params':    ['C10'],
        'features':  features_NH,
    },
}

# Per-dataset training settings (override DEFAULTS and MODELS).
# A noised copy (<dataset>_CORR_noised_<level>) uses the settings of its source dataset.
DATASETS = {
    'Synthetic_CheckerPattern_NeoHookean': {'no_epochs': 60001,  'ls_factor': 3.0},   # smooth checkerboard (synthetic)
    'Exp_Homogeneous_NeoHookean':          {'no_epochs': 80001,  'ls_factor': 3.0},   # homogeneous specimen with hole
    'Exp_Circular_Inclusion_NeoHookean':   {'no_epochs': 60001,  'ls_factor': 3.0},   # circular-inclusion specimen
    'Exp_Concentric_Square_NeoHookean':    {'no_epochs': 100001, 'ls_factor': 3.0},   # concentric-square inclusion specimen
}

# Latex labels used in plots (fallback: \theta_{<param>})
latex_labels = {
    'C10':        r"$\theta_{C10}$",
}


def get_latex_label(param):
    return latex_labels.get(param, rf"$\theta_{{{param}}}$")


def get_dataset_settings(fem_material):
    """
    Settings of a dataset from DATASETS ({} if it is not listed).
    Noised copies (<dataset>_CORR_noised_...) match their source dataset.
    """
    matches = [name for name in DATASETS if fem_material == name or fem_material.startswith(name + '_CORR_noised_')]
    return dict(DATASETS[max(matches, key=len)]) if matches else {}


#---------------------------------------------------------------------
# Active model (set once by the driver, read by features_library etc.)
#---------------------------------------------------------------------
_active_model_name = None


def get_model(model_name=None):
    """
    Return the settings of a model (DEFAULTS merged with its MODELS entry).
    Without an argument, the active model is returned.
    """
    if model_name is None:
        model_name = get_model_name()
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODELS.keys())}")
    model = dict(DEFAULTS)
    model.update(MODELS[model_name])
    model['name'] = model_name
    return model


def get_model_name():
    if _active_model_name is None:
        raise RuntimeError("No model selected. Call set_model(<name>) first, e.g. set_model('NH').")
    return _active_model_name


def set_model(model_name):
    """
    Select the model used by computeFeatures_torch, getNumberOfFeatures and the plots.
    Checks that the number of features matches the number of parameters.
    """
    global _active_model_name
    model = get_model(model_name)

    # Evaluate the features once (undeformed state: K1 = K2 = 0, J = 1, Iat = Ibt = 1) to count them
    one = torch.ones(1, 1)
    zero = torch.zeros(1, 1)
    features_ref = model['features'](zero, zero, one, one, one)

    if len(features_ref) != len(model['params']):
        raise ValueError(f"Model '{model_name}': {len(features_ref)} features but "
                         f"{len(model['params'])} params {model['params']}.")

    _active_model_name = model_name
    print(f"Model: {model_name} ({model['label']}), parameters: {model['params']}")
    return model
