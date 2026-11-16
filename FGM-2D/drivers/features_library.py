import numpy as np
import torch
from helper import *
from config import *
from models import get_model, get_model_name


def get_mass_type():
    """

    """

    mass_type = "lumped" #Can be made 'consistent' otherwise

    return mass_type


def computeFeatures_torch(I1, I2, I3, Ia, Ib):
    """
    Compute the features dependent on the right Cauchy-Green strain invariants.
    The features (and their order) are those of the active model in models.py,
    selected with set_model(<name>).

    _Input Arguments_

    - `I1` - 1st invariant

    - `I2` - 2nd invariant

    - `I3` - 3rd invariant

    - `Ia` - Invariant along anisotropy direction.

    - `Ib` - Invariant along second anisotropy direction.

    _Output Arguments_

    - `x` - features

    ---

    """
    eps = 0
    numFeatures = getNumberOfFeatures()
    x = torch.zeros(I1.shape[0],numFeatures)
    K1 = (I1+ eps) * torch.pow(I3+ eps,-1/3) - 3.0
    # Some standard terms
    # K1 = I1 * torch.pow(I3,-1/3) - 3.0
    K2 = ((I2+ eps) * torch.pow(I3+ eps,-2/3))**1.0 - 3.0**1.0
    J = torch.sqrt(I3+eps)
    Iat = Ia * torch.pow(J,-2./3.)
    Ibt = Ib * torch.pow(J,-2./3.)
    i=-1;

    # Features of the active model (defined in models.py)
    for feature in get_model()['features'](K1, K2, J, Iat, Ibt):
        i+=1; x[:,i:(i+1)] = feature

    ####################################################
    # REFERENCE CATALOGUE (not active).
    # To use any of the terms below, define a model in models.py.
    ####################################################
    # Generalized Mooney-Rivlin:

    #linear
    # i+=1; x[:,i:(i+1)] = K1
    # i+=1; x[:,i:(i+1)] = K2
    # # #quadratic
    # i+=1; x[:,i:(i+1)] = torch.pow(K1,2.0)
    # i+=1; x[:,i:(i+1)] = K1 * K2
    # i+=1; x[:,i:(i+1)] = torch.pow(K2,2.0)
    # # # #cubic
    # i+=1; x[:,i:(i+1)] = torch.pow(K1,3.0)
    # # i+=1; x[:,i:(i+1)] = (K1**2) * K2
    # # i+=1; x[:,i:(i+1)] = K1 * (K2**2)
    # # i+=1; x[:,i:(i+1)] = K2**3
    # #quartic
    # i+=1; x[:,i:(i+1)] = torch.pow(K1,4.0)
    # i+=1; x[:,i:(i+1)] = (K1**3) * K2
    # i+=1; x[:,i:(i+1)] = (K1**2) * (K2**2)
    # i+=1; x[:,i:(i+1)] = K1 * (K2**3)
    # i+=1; x[:,i:(i+1)] = K2**4

    ######################################################
    #volumetric term:
    # i+=1; x[:,i:(i+1)] = (J-1.0)**2.0#(J-1/J)**2.0
    # i+=1; x[:,i:(i+1)] = torch.pow(J-1,4.0)
    # i+=1; x[:,i:(i+1)] = torch.pow(J-1,6.0)
    # i+=1; x[:,i:(i+1)] = torch.pow(J-1,8.0)
    # i+=1; x[:,i:(i+1)] = (J-1)**10

    # ####################################################
    # Gent-Thomas:
    # i+=1; x[:,i:(i+1)] = torch.log(I2 * torch.pow(I3,-2/3)/3.0)
    #
    # ####################################################
    # #Arruda Boyce
    # lambda_chain = torch.sqrt(I1 * torch.pow(J,-2./3.)/3.)
    # xtilde = lambda_chain / torch.sqrt(torch.tensor([28.]))
    # s1 = (torch.sign(0.841-xtilde)+1.)/2.
    # s2 = (torch.sign(xtilde-0.841)+1.)/2.
    # beta_chain1 = 1.31*torch.tan(1.59*xtilde)+0.91*xtilde
    # beta_chain2 = torch.tensor([1.])/(torch.sign(xtilde) - xtilde)
    # beta_chain = s1 * beta_chain1 + s2*beta_chain2

    # R1 = beta_chain*lambda_chain + torch.sqrt(torch.tensor([28.]))*torch.log(beta_chain/torch.sinh(beta_chain))
    # i+=1; x[:,i:(i+1)] = torch.tensor([10.])* (torch.sqrt(torch.tensor([28.]))*R1 - 1.5164)
    #
    #
    # ####################################################
    # #Ogden
    # I1_tilde = J**(-2/3)*I1 + 1e-13
    # I1t_0 =torch.tensor([3]) + 1e-13
    # J_0 = torch.tensor([1]) + 1e-13
    # mu_ogden = 1
    # alpha_ogden = 0.65
    # W_offset = 1/alpha_ogden * 2. * (0.5**alpha_ogden*(I1t_0  +  torch.sqrt(  (I1t_0-1/(J_0**(2./3.)))**2 - 4*J_0**(2./3.)) - 1/(J_0**(2./3.)) )**alpha_ogden+( 0.5*I1t_0 - 0.5*torch.sqrt(  (I1t_0-1/(J_0**(2./3.)))**2 - 4*J_0**(2./3.))  - 0.5/(J_0**(2./3.)) )**alpha_ogden + J_0**(-alpha_ogden*2./3.) ) * mu_ogden
    # W_truth = 1/alpha_ogden * 2. * (0.5**alpha_ogden*(I1_tilde  +  torch.sqrt(  (I1_tilde-1/(J**(2./3.)))**2 - 4*J**(2./3.)) - 1/(J**(2./3.)) )**alpha_ogden+( 0.5*I1_tilde - 0.5*torch.sqrt(  (I1_tilde-1/(J**(2./3.)))**2 - 4*J**(2./3.))  - 0.5/(J**(2./3.)) )**alpha_ogden + J**(-alpha_ogden*2./3.) ) * mu_ogden - W_offset
    # i+=1; x[:,i:(i+1)] = W_truth
    # ####################################################
    # #Ogden2ndterm
    # I1_tilde = J**(-2/3)*I1 + 1e-13
    # I1t_0 =torch.tensor([3]) + 1e-13
    # J_0 = torch.tensor([1]) + 1e-13
    # mu_ogden = 1
    # alpha_ogden = 2.5
    # W_offset = 1/alpha_ogden * 2. * (0.5**alpha_ogden*(I1t_0  +  torch.sqrt(  (I1t_0-1/(J_0**(2./3.)))**2 - 4*J_0**(2./3.)) - 1/(J_0**(2./3.)) )**alpha_ogden+( 0.5*I1t_0 - 0.5*torch.sqrt(  (I1t_0-1/(J_0**(2./3.)))**2 - 4*J_0**(2./3.))  - 0.5/(J_0**(2./3.)) )**alpha_ogden + J_0**(-alpha_ogden*2./3.) ) * mu_ogden
    # W_truth = 1/alpha_ogden * 2. * (0.5**alpha_ogden*(I1_tilde  +  torch.sqrt(  (I1_tilde-1/(J**(2./3.)))**2 - 4*J**(2./3.)) - 1/(J**(2./3.)) )**alpha_ogden+( 0.5*I1_tilde - 0.5*torch.sqrt(  (I1_tilde-1/(J**(2./3.)))**2 - 4*J**(2./3.))  - 0.5/(J**(2./3.)) )**alpha_ogden + J**(-alpha_ogden*2./3.) ) * mu_ogden - W_offset
    # i+=1; x[:,i:(i+1)] = W_truth
    # ####################################################
    # #Ogden3rdTerm
    # I1_tilde = J**(-2/3)*I1 + 1e-13
    # I1t_0 =torch.tensor([3]) + 1e-13
    # J_0 = torch.tensor([1]) + 1e-13
    # mu_ogden = 1
    # alpha_ogden = 1
    # W_offset = 1/alpha_ogden * 2. * (0.5**alpha_ogden*(I1t_0  +  torch.sqrt(  (I1t_0-1/(J_0**(2./3.)))**2 - 4*J_0**(2./3.)) - 1/(J_0**(2./3.)) )**alpha_ogden+( 0.5*I1t_0 - 0.5*torch.sqrt(  (I1t_0-1/(J_0**(2./3.)))**2 - 4*J_0**(2./3.))  - 0.5/(J_0**(2./3.)) )**alpha_ogden + J_0**(-alpha_ogden*2./3.) ) * mu_ogden
    # W_truth = 1/alpha_ogden * 2. * (0.5**alpha_ogden*(I1_tilde  +  torch.sqrt(  (I1_tilde-1/(J**(2./3.)))**2 - 4*J**(2./3.)) - 1/(J**(2./3.)) )**alpha_ogden+( 0.5*I1_tilde - 0.5*torch.sqrt(  (I1_tilde-1/(J**(2./3.)))**2 - 4*J**(2./3.))  - 0.5/(J**(2./3.)) )**alpha_ogden + J**(-alpha_ogden*2./3.) ) * mu_ogden - W_offset
    # i+=1; x[:,i:(i+1)] = W_truth
    # ####################################################
    # #Anisotropy 1 fibre
    # i+=1; x[:,i:(i+1)] = torch.pow(Iat - 1.,2.)
    # i+=1; x[:,i:(i+1)] = torch.pow(Iat - 1.,3.)
    # i+=1; x[:,i:(i+1)] = torch.pow(Iat - 1.,4.)
    # ####################################################
    # #Anisotropy 2 fibre
    # i+=1; x[:,i:(i+1)] = torch.pow(Ibt - 1.,2.)
    # i+=1; x[:,i:(i+1)] = torch.pow(Ibt - 1.,3.)
    # i+=1; x[:,i:(i+1)] = torch.pow(Ibt - 1.,4.)
    # ####################################################
    # # Holzapfel features (These features are filtered out during chain formation. They are used only to evaluate ground-truth Holzapfel model)
    # k1h = 0.9;
    # k2h = 0.6;
    # i+=1; x[:,i:(i+1)] = torch.exp(k2h*torch.pow(Iat - 1.,2.))-1.;
    # i+=1; x[:,i:(i+1)] = k1h/2./k2h*(torch.exp(k2h*torch.pow(Iat - 1.,2.))-1.);
    # i+=1; x[:,i:(i+1)] = k1h/2./k2h*(torch.exp(k2h*torch.pow(Ibt - 1.,2.))-1.);
    #
    # . . .  add more models in models.py

    if(i!=numFeatures-1):
        raise ValueError(f"Model '{get_model_name()}' returned {i+1} features, expected {numFeatures}.")
    return x


def getNumberOfFeatures():
    """
    Compute number of features (= number of parameters of the active model).

    _Input Arguments_

    - _none_

    _Output Arguments_

    - `numFeatures` - number of features

    ---

    """
    return len(get_model()['params'])



def computeFeatures_numpy(F11, F12, F21, F22, F13, F23, F31, F32, F33):
    #Adapted from first paper's code with modifications
    #(replacing torch with numpy)

    if(len(F11.shape) == 1):
        F11 = np.expand_dims(F11, axis=-1)
        F12 = np.expand_dims(F12, axis=-1)
        F21 = np.expand_dims(F21, axis=-1)
        F22 = np.expand_dims(F22, axis=-1)
        F13 = np.expand_dims(F13, axis=-1)
        F23 = np.expand_dims(F23, axis=-1)
        F31 = np.expand_dims(F31, axis=-1)
        F32 = np.expand_dims(F32, axis=-1)
        F33 = np.expand_dims(F33, axis=-1)

    C11 = F11**2 + F21**2 + F31**2
    C12 = F11*F12 + F21*F22 + F31*F32
    C13 = F11*F13 + F21*F23 + F31*F33
    C21 = F11*F12 + F21*F22 + F31*F32
    C22 = F12**2 + F22**2 + F32**2
    C23 = F12*F13 + F22*F23 + F32*F33
    C31 = F11*F13 + F21*F23 + F31*F33
    C32 = F12*F13 + F22*F23 + F32*F33
    C33 = F13**2 + F23**2 + F33**2

    I1 = C11 + C22 + C33
    I2 = C11*C22 + C11*C33 + C22*C33 - C12*C21 - C13*C31 - C23*C32
    I3 = (-C13*C22*C31 + C12*C23*C31 + C13*C21*C32 - C11*C23*C32 - C12*C21*C33 + C11*C22*C33)
    Ia = C11*a1**2.0 + 2.0*C12*a1*a2 + C22*a2**2.0
    Ib = C11*b1**2.0 + 2.0*C12*b1*b2 + C22*b2**2.0

    I1 = torch.tensor(I1)
    I2 = torch.tensor(I2)
    I3 = torch.tensor(I3)
    Ia = torch.tensor(Ia)
    Ib = torch.tensor(Ib)


    x = computeFeatures_torch(I1, I2, I3, Ia, Ib).detach().cpu().numpy()

    return x
