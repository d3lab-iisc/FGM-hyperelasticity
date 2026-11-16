import torch

#=====================================================================
# MODEL LIBRARY
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
#
# To add a model:
#   1. write a features_<NAME>(K1, K2, J, Iat, Ibt) function returning a list of features
#   2. add an entry to MODELS with 'params' in the SAME order as the features
#   3. the dataset's output_elements_matProperty.csv needs ground-truth columns named as 'params'
#   4. run: python main_fgm_HEX.py <NAME> <material_name> <noise_level>
#=====================================================================


def features_NH(K1, K2, J, Iat, Ibt):
    # Neo-Hookean: C10, inv_D1
    return [K1,
            (J-1.0)**2.0]


def features_MR(K1, K2, J, Iat, Ibt):
    # Mooney-Rivlin: C10, C01, inv_D1
    return [K1,
            K2,
            (J-1.0)**2.0]


def features_ISH(K1, K2, J, Iat, Ibt):
    # Isihara: C10, C01, C20, inv_D1
    return [K1,
            K2,
            torch.pow(K1,2.0),
            (J-1.0)**2.0]


def features_HGO(K1, K2, J, Iat, Ibt, k2h=1.0):
    # Modified HGO: C10, inv_D1, k1  (k2h is fixed, not discovered)
    return [K1,
            (J-1.0)**2.0,
            1.0/2./k2h*(torch.exp(k2h*torch.pow(Iat - 1.,2.))-1.)]


# Settings shared by all models; a model entry only lists what differs.
DEFAULTS = {
    'no_epochs':      60001,
    'lr':             0.005,
    'ls_factor':      1.25,                 # GP length scale = ls_factor * element length
    'lambda_gp':      1e-4,                 # GP smoothness weight
    'lambda_reg':     1e-5,                 # regularization weight on the first parameter
    'loadsteps_list': [1,2,3,4,5,6],
    'log_every':      1000,                 # print/log interval (epochs)
    'gmm_modes':      None,                 # number of GMM modes for plot_histogram_gmm (None = skip)
}

MODELS = {
    'NH': {
        'label':     'Neo-Hookean',
        'params':    ['C10', 'inv_D1'],
        'features':  features_NH,
        'log_every': 500,
    },
    'MR': {
        'label':     'Mooney-Rivlin',
        'params':    ['C10', 'C01', 'inv_D1'],
        'features':  features_MR,
    },
    'ISH': {
        'label':     'Isihara',
        'params':    ['C10', 'C01', 'C20', 'inv_D1'],
        'features':  features_ISH,
    },
    'HGO': {
        'label':     'Holzapfel-Gasser-Ogden',
        'params':    ['C10', 'inv_D1', 'k1'],
        'features':  features_HGO,
        'no_epochs': 150001,
        'lr':        0.5,
        'gmm_modes': 2,
    },
}

# Latex labels used in plots (fallback: \theta_{<param>})
latex_labels = {
    'C10':        r"$\theta_{C10}$",
    'C01':        r"$\theta_{C01}$",
    'C20':        r"$\theta_{C20}$",
    'inv_D1':     r"$\theta_{1/D1}$",
    'k1':         r"$\theta_{k1}$",
}


def get_latex_label(param):
    return latex_labels.get(param, rf"$\theta_{{{param}}}$")


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
