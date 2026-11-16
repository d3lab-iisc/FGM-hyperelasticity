import sys
# sys.path.insert(1, '../drivers/')
sys.path.insert(2, '../core/')
import numpy as np
import torch
# from core_spike_slab import *
from helper import *
from features_library import *
from config import *
from features_library import getNumberOfFeatures
from models import get_model
from utilities import computeCauchyGreenStrain, computeStrainInvariants


def predict_energy_path_FGM(theta_mean, theta_variance, theta_gt, deformation):
    """
    Strain energy W along a deformation path for ONE material, for any model in models.py.
    Every model is linear in its parameters, W = sum_i theta_i * feature_i(F), so
        W_mean = features(F) @ theta_mean
        W_var  = features(F)^2 @ theta_variance
    The features are those of the active model (set_model), the fibre direction comes from config.py (a_mult, b_mult).

    _Input Arguments_

    - `theta_mean`     - mean of the parameters of the material, shape (numFeatures,)

    - `theta_variance` - variance of the parameters of the material, shape (numFeatures,)

    - `theta_gt`       - ground-truth parameters of the material, shape (numFeatures,)

    - `deformation`    - tension, simple_shear, pure_shear, compression, biaxial_tension, biaxial_compression

    """
    params = get_model()['params']
    theta_mean     = np.asarray(theta_mean, dtype=np.float64).reshape(-1)
    theta_variance = np.asarray(theta_variance, dtype=np.float64).reshape(-1)
    theta_gt       = np.asarray(theta_gt, dtype=np.float64).reshape(-1)

    #----------------------------------------

    gamma = np.linspace(0.00001,1.,num=50)

    F11 = np.zeros_like(gamma) + 1.
    F12 = np.zeros_like(gamma)
    F21 = np.zeros_like(gamma)
    F22 = np.zeros_like(gamma) + 1.
    F13 = np.zeros_like(gamma)
    F31 = np.zeros_like(gamma)
    F32 = np.zeros_like(gamma)
    F23 = np.zeros_like(gamma)
    F33 = np.zeros_like(gamma) + 1.

    if deformation == 'tension':
        F11 = 1.+gamma

    elif deformation == 'simple_shear':
        F12 = gamma

    elif deformation == 'pure_shear':
        F11 = 1.+gamma
        F22 = 1./(1.+gamma)

    elif deformation == 'compression':
        F11 = 1./(1.+gamma)

    elif deformation == 'biaxial_tension':
        F11 = 1.+gamma
        F22 = 1.+gamma

    elif deformation == 'biaxial_compression':
        F11 = 1./(1.+gamma)
        F22 = 1./(1.+gamma)

    # Deformation gradient in the Voigt ordering used in utilities.py
    F = torch.tensor(np.column_stack((F11, F12, F21, F22, F13, F23, F31, F32, F33)), dtype=torch.float64)

    C = computeCauchyGreenStrain(F)
    I1, I2, I3 = computeStrainInvariants(C)

    # a_mult and b_mult are defined in config (fibre directions)
    Ia = C @ a_mult
    Ib = C @ b_mult

    # Energy feature terms of the active model, shape (len(gamma), numFeatures)
    features = computeFeatures_torch(I1, I2, I3, Ia, Ib).detach().cpu().numpy()

    # print a tabular format with the Ground truth, mean and variance of the parameters
    print(f"\n{get_model()['label']} model parameters:")
    print("Parameter\tGround Truth\tMean\tVariance")
    for j, p in enumerate(params):
        print("{}\t\t{:.2f}\t\t{:.2f}\t{:.4f}".format(p, theta_gt[j], theta_mean[j], theta_variance[j]))

    energy_gt   = features @ theta_gt
    energy_mean = features @ theta_mean
    W_var = (features**2) @ theta_variance
    energy_std = np.sqrt(W_var)
    energy_plus  = energy_mean + 1.96*energy_std
    energy_minus = energy_mean - 1.96*energy_std

    print("\nEnergy prediction completed for deformation: ", deformation)
    return gamma, energy_mean, energy_plus, energy_minus, energy_gt
