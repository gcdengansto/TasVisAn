import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from datetime import datetime
from lmfit import Parameters, fit_report, minimize
import inspy as npy
import h5py

PG002_D_SPACING     = 3.355  # Angstroms
NI_LATTICE_CONSTANT = 3.524  # Angstroms

def descend_obj(obj, sep='\t'):
    #Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
    #return the list of key (namelist) and the data title (col_list)
    #the key name (e.g. /entry1/data ) can be used to get the data by open hdf file f= h5py.File, and using f(key)
    #col_list is the key (e.g. s1, s2)
    namelist = []
    col_list = []
    allmotornames= ['Pt.',  's1', 's2', 'qm', 'qh', 'qk', 'ql', 'ei', 'vei', 'ef', 'en', 'time', 'bm1_counts', 'bm2_counts', 'dummy_motor',
                    'VS_left', 'VS_right', 'ps_right', 'ps_left', 'ps_top', 'ps_bottom', 'pa_right', 'pa_left', 'pa_top', 'pa_bottom', 
                    'm1', 'm2', 'mtilt', 'mtrans',  'sgl', 'sgu', 'stl', 'stu', 'a1', 'a2', 'atrans', 'atilt', 'ahfocus', 'avfocus', 
                    'pghf', 'pgvf', 'cuhf', 'cuvf','sensorValueA','sensorValueB','sensorValueC','sensorValueD','setpoint1','setpoint2']
    if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
        for key in obj.keys():
            #print(key)
            if key in allmotornames:
                #print(sep+'group','+',key,':',obj[key])
                #print(obj[key].name)
                namelist.append(obj[key].name)
                col_list.append(key)
            templist1, templist2 = descend_obj(obj[key],sep=sep+'\t')
            namelist = namelist + templist1
            col_list = col_list + templist2
    elif type(obj) == h5py._hl.dataset.Dataset:
        for key in obj.attrs.keys():
            if key in allmotornames:
                #print(sep+'data:\t','-',key,':',obj.attrs[key])
                pass
    
    return namelist, col_list


def h5dump(path, group='/'):
    #print HDF5 file metadata
    #group: you can give a specific group, defaults to the root group

    with h5py.File(path,'r') as file:
         namelist, col_list = descend_obj(file[group])
    #print(namelist)
    #print(col_list)
    return namelist, col_list


def deg2rad(anglex):
    return anglex*np.pi/180.0

def AnglesToQhkl(m2, s1, s2, a2, sgu, sgl, UBMat):

    d_pg002 = PG002_D_SPACING  #d spacing of pg
    phi     = s2

    #m1rad  = deg2rad(m2/2.0)
    m2rad  = deg2rad(m2)

    #a1rad  = deg2rad(a2/2.0)
    a2rad  = deg2rad(a2)

    s1rad  = deg2rad(s1)
    phirad = deg2rad(phi)
    s2rad  = phirad

    sgu_rad = deg2rad(sgu)
    sgl_rad = deg2rad(sgl)


    ki  =   np.pi/np.sin(m2rad/2)/d_pg002

    kf  =   np.pi/np.sin(a2rad/2)/d_pg002

    q   =   np.sqrt(ki*ki+kf*kf-2*ki*kf*np.cos(phirad))

    theta_rad = np.arctan((ki-kf*np.cos(phirad))/kf/np.sin(phirad))

    omega_rad = s1rad-theta_rad

    MatN = np.matrix([(1.0,         0.0,                    0.0             ), 
                      (0.0,         np.cos(sgu_rad),        -np.sin(sgu_rad)), 
                      (0.0,         np.sin(sgu_rad),        np.cos(sgu_rad))])
    
    
    MatM = np.matrix([( np.cos(sgl_rad),             0.0,          np.sin(sgl_rad)), 
                      (             0.0,             1.0,                      0.0), 
                      (-np.sin(sgl_rad),             0.0,          np.cos(sgl_rad))])




    MatO = np.matrix([( np.cos(omega_rad),             -np.sin(omega_rad),                      0.0), 
                      ( np.sin(omega_rad),              np.cos(omega_rad),                      0.0), 
                      (               0.0,                            0.0,                      1.0)])



    MatTh = np.matrix([( np.cos(theta_rad),             -np.sin(theta_rad),                      0.0), 
                       ( np.sin(theta_rad),              np.cos(theta_rad),                      0.0), 
                       (               0.0,                            0.0,                      1.0)])

    MatTemp = np.matmul(MatN.I, MatM.I)

    MatTemp = np.matmul(MatTemp, MatO.I)

    Qtheta = np.array([[q],
                       [0],
                       [0]])
    
    Qnu   = np.matmul(MatTemp, Qtheta)

    Qhkl  = np.matmul(UBMat.I, Qnu)

    Qhkl  = Qhkl/2/np.pi

    return Qhkl

def AnglesToQhklArray(m2, s1, s2, a2, sgu, sgl, UBMat):
    """
    the parameters of this function are arrays or dataframe
    
    """

    d_pg002 = PG002_D_SPACING  #d spacing of pg
    phi     = s2

    #m1rad  = deg2rad(m2/2.0)
    m2rad  = deg2rad(m2)

    #a1rad  = deg2rad(a2/2.0)
    a2rad  = deg2rad(a2)

    s1rad  = deg2rad(s1)
    phirad = deg2rad(phi)
    s2rad  = phirad

    sgu_rad = deg2rad(sgu)
    sgl_rad = deg2rad(sgl)


    ki  =   np.pi/np.sin(m2rad/2)/d_pg002

    kf  =   np.pi/np.sin(a2rad/2)/d_pg002

    q   =   np.sqrt(ki*ki+kf*kf-2*ki*kf*np.cos(phirad))

    theta_rad = np.arctan((ki-kf*np.cos(phirad))/kf/np.sin(phirad))

    omega_rad = s1rad-theta_rad

    MatN = np.matrix([(1.0,         0.0,                    0.0             ), 
                      (0.0,         np.cos(sgu_rad),        -np.sin(sgu_rad)), 
                      (0.0,         np.sin(sgu_rad),        np.cos(sgu_rad))])
    
    
    MatM = np.matrix([( np.cos(sgl_rad),             0.0,          np.sin(sgl_rad)), 
                      (             0.0,             1.0,                      0.0), 
                      (-np.sin(sgl_rad),             0.0,          np.cos(sgl_rad))])




    MatO = np.matrix([( np.cos(omega_rad),             -np.sin(omega_rad),                      0.0), 
                      ( np.sin(omega_rad),              np.cos(omega_rad),                      0.0), 
                      (               0.0,                            0.0,                      1.0)])



    MatTh = np.matrix([( np.cos(theta_rad),             -np.sin(theta_rad),                      0.0), 
                       ( np.sin(theta_rad),              np.cos(theta_rad),                      0.0), 
                       (               0.0,                            0.0,                      1.0)])

    MatTemp = np.matmul(MatN.I, MatM.I)

    MatTemp = np.matmul(MatTemp, MatO.I)

    Qtheta = np.array([[q],
                       [0],
                       [0]])
    
    Qnu   = np.matmul(MatTemp, Qtheta)

    Qhkl  = np.matmul(UBMat.I, Qnu)

    Qhkl  = Qhkl/2/np.pi

    return Qhkl

def strisfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def strisint(num):
    try:
        int(num)
        return True
    except ValueError:
        return False

def gaussian_residual(pars, x, data=None):
    """Model a decaying sine wave and subtract data."""
    vals = pars.valuesdict()
    A    = vals['A']
    x0   = vals['x0']
    w    = vals['w']
    bg   = vals['bg']

    model = A/(w*np.sqrt(np.pi/(4*np.log(2)))) * np.exp(-4*np.log(2)*(x-x0)**2/w**2) + bg
    if data is None:
        return model
    return model - data

'''
def gaussian_residual(pars: Parameters, x: np.ndarray, data: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate the difference between the Gaussian peak calculated from the parameters in pars and the real data.

    Parameters:
    pars (Parameters): Parameters containing 'A', 'x0', 'w', and 'bg'.
    x (np.ndarray): The range of x values.
    data (Optional[np.ndarray]): The real data to compare against. If None, the function returns the model.

    Returns:
    np.ndarray: The residuals (model - data) if data is provided, otherwise the model.
    """
    # Validate parameters
    required_keys = ['A', 'x0', 'w', 'bg']
    for key in required_keys:
        if key not in pars:
            raise ValueError(f"Parameter '{key}' is missing from 'pars'.")

    vals = pars.valuesdict()
    A = vals['A']
    x0 = vals['x0']
    w = vals['w']
    bg = vals['bg']

    # Calculate the Gaussian model
    model = A / (w * np.sqrt(np.pi / (4 * np.log(2)))) * np.exp(-4 * np.log(2) * (x - x0) ** 2 / w ** 2) + bg
    
    if data is None:
        return model
    
    # Calculate and return the residuals
    return model - data

'''
    
def lorentzian_residual(pars, x, data=None):
    """Model a decaying sine wave and subtract data."""
    vals = pars.valuesdict()
    A    = vals['A']
    x0   = vals['x0']
    w    = vals['w']
    bg   = vals['bg']

    model = A*w/((x-x0)**2+w**2/4)/np.pi/2.0 + bg
    if data is None:
        return model
    return model - data

def gaussian(pars, x):
    """Model a decaying sine wave and subtract data."""
    vals = pars.valuesdict()
    A    = vals['A']
    x0   = vals['x0']
    w    = vals['w']
    #bg   = vals['bg']

    return A/(w*np.sqrt(np.pi/(4*np.log(2)))) * np.exp(-4*np.log(2)*(x-x0)**2/w**2) 

def lorentzian(pars, x, data=None):
    """Model a decaying sine wave and subtract data."""
    vals = pars.valuesdict()
    A    = vals['A']
    x0   = vals['x0']
    w    = vals['w']
    #bg   = vals['bg']

    return A*w/((x-x0)**2+w**2/4)/np.pi/2.0 

def three_peaks_on_slope_residual(pars, x, data=None):
    """Model a decaying sine wave and subtract data."""
    vals = pars.valuesdict()

    #peak 1
    A    = vals['A_1']
    x0   = vals['x0_1']
    w    = vals['w_1']
    GL   = vals['GL_1']

    peak_params  =  Parameters()
    peak_params.add('A',    value  =  A)
    peak_params.add('x0',    value  =  x0)
    peak_params.add('w',   value  =  w)

    model = 0
    if GL == 1:
        model = gaussian(peak_params, x)
    elif GL == 2:
        model = lorentzian(peak_params, x)
    elif GL == 0:
        model = model + 0

    #peak 2
    A    = vals['A_2']
    x0   = vals['x0_2']
    w    = vals['w_2']
    GL   = vals['GL_2']

    peak_params['A'].value   =  A
    peak_params['x0'].value  =  x0
    peak_params['w'].value   =  w

    if GL == 1: #gaussian
        model = model + gaussian(peak_params, x)
    elif GL == 2:
        model = model + lorentzian(peak_params, x)
    elif GL == 0:
        model = model + 0

    #peak 3
    A    = vals['A_3']
    x0   = vals['x0_3']
    w    = vals['w_3']
    GL   = vals['GL_3']

    peak_params['A'].value   =  A
    peak_params['x0'].value  =  x0
    peak_params['w'].value   =  w

    if GL == 1:
        model = model + gaussian(peak_params, x)
    elif GL == 2:
        model = model + lorentzian(peak_params, x)
    elif GL == 0:
        model = model + 0

    #slope background
    bg    = vals['bg']
    slope = vals['slope']

    model = model + slope*x + bg
    
    if data is None:
        return model
    return model - data

def lorenz_residual(pars, x, data=None):
    """Model a lorenz and subtract data."""
    vals = pars.valuesdict()
    A   = vals['A']
    kai = vals['kai']
    x0  = vals['x0']
    bg  = vals['bg']
    
    model = A/(kai**2+(x-x0)**2)    + bg
    if data is None:
        return model
    return model - data

def lorenzsq_residual(pars, x, data=None):
    """Model a decaying lorenzsq and subtract data."""
    vals = pars.valuesdict()
    A   = vals['A']
    kai = vals['kai']
    x0  = vals['x0']
    bg  = vals['bg']
    
    model = A/(kai**2+(x-x0)**2)**2    + bg
    if data is None:
        return model
    return model - data

#this is a comparison of lorentz and lorentzsq
def lorenz_gaussian_residual(pars, x, data=None):
    """Model a decaying lorenz + gaussian and subtract data."""
    vals = pars.valuesdict()
    A    = vals['A']
    kai = vals['kai']
    x0  = vals['x0']
    bg  = vals['bg']
    B   = vals['B']
    w   = vals['w']
    
    model = A/(kai**2+(x-x0)**2)**2    +  B/(w*np.sqrt(np.pi/(4*np.log(2)))) * np.exp(-4*np.log(2)*(x-x0)^2/w^2) + bg
    if data is None:
        return model
    return model - data  

def lorentzian_asym(pars, x, data=None):
    """Model a decaying sine wave and subtract data."""
    vals   = pars.valuesdict()
    A      = vals['A']
    gamma0 = vals['gamma0']
    x0     = vals['x0']
    asym   = vals['asym']
    bg     = vals['bg']
    res_w  = vals['res_w']
    
    step=x[1]-x[0]

    x_range=np.linspace(x0-3*step, x0+3*step, 7)
    res_g  = 1/(res_w*np.sqrt(np.pi/(4*np.log(2)))) * np.exp(-4*np.log(2)*(x_range-x0)**2/res_w**2)

    gg  = 2*gamma0/(1+np.exp(asym*(x-x0)))

    model =  A*gg/(gg**2+(x-x0)**2)-A*gg/(gg**2+(x+x0)**2)

    conv_model = np.convolve(model, res_g, 'same') + bg
    
    
    if data is None:
        return conv_model
    return conv_model - data


def ni_s2_residual(pars, x, data=None):
    """Model a decaying sine wave and subtract data."""
    vals        = pars.valuesdict()
    #m1         = vals['m1']
    s2_offset   = vals['s2_offset']
    wavelen     = vals['wavelen']
    
    ni_hkl      = [[0.5, 0.5, 0.5],
                   [1, 0, 0],
                   [1, 1, 0],
                   [0.5, 0.5, 1.5],
                   [1, 1, 1],
                   [2, 0, 0]]


    ni_a        = NI_LATTICE_CONSTANT
    
    calc_s2=np.array([])
    
    for ii in  range(len(x)):
        s2 = -(np.arcsin(wavelen*np.sqrt(ni_hkl[ii][0]**2 + ni_hkl[ii][1]**2 + ni_hkl[ii][2]**2)/ni_a/2)*360/np.pi+s2_offset)
        calc_s2 = np.append(calc_s2,s2)

    if data is None:
        return calc_s2
    return calc_s2 - data

def ni_s2_residual_30p5meV(pars, x, data=None):
    """Model a decaying sine wave and subtract data."""
    vals = pars.valuesdict()
    #m1    = vals['m1']
    s2_offset   = vals['s2_offset']
    wavelen     = vals['wavelen']
    
    ni_hkl=[[0.5, 0.5, 0.5],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
            [2, 0, 0],
            [2, 2, 0]]


    ni_a = NI_LATTICE_CONSTANT
    
    calc_s2=np.array([])
    
    for ii in  range(len(x)):
        s2 = -(np.arcsin(wavelen*np.sqrt(ni_hkl[ii][0]**2 + ni_hkl[ii][1]**2 + ni_hkl[ii][2]**2)/ni_a/2)*360/np.pi+s2_offset)
        calc_s2 = np.append(calc_s2,s2)

    if data is None:
        return calc_s2
    return calc_s2 - data
    

def ni_s2_residual_cu(pars, x, data=None):
    """Model a decaying sine wave and subtract data."""
    vals = pars.valuesdict()
    #m1    = vals['m1']
    s2_offset   = vals['s2_offset']
    wavelen     = vals['wavelen']
    
    ni_hkl=[[1, 1, 1],
            [2, 0, 0],
            [2, 1, 1],[2, 2, 0],[3, 1, 1],[4, 0, 0]]


    ni_a = NI_LATTICE_CONSTANT
    
    calc_s2=np.array([])
    
    for ii in  range(len(x)):
        s2 = -(np.arcsin(wavelen*np.sqrt(ni_hkl[ii][0]**2 + ni_hkl[ii][1]**2 + ni_hkl[ii][2]**2)/ni_a/2)*360/np.pi+s2_offset)
        calc_s2 = np.append(calc_s2,s2)

    if data is None:
        return calc_s2
    return calc_s2 - data
    
def fit_peak(x, data=None, func='G', initial=None):
    
    fitted_data =  pd.DataFrame([])
    fitted_params   =  pd.DataFrame(columns = ['A', 'A_err', 'w','w_err', 'x0', 'x0_err', 'bg', 'bg_err'])
    dataX=x
    dataY=data
    if isinstance(dataY, pd.DataFrame):
        dataX = dataX.to_numpy()
    if isinstance(data, pd.DataFrame):
        dataY = dataY.to_numpy()
    dataX=dataX[np.logical_not(np.isnan(dataX))]
    dataY=dataY[np.logical_not(np.isnan(dataY))]
    #print(dataX)
    #print(dataY)
    peak_params  =  Parameters()
    if initial is None:
        pointnum=len(dataY)
        """
        peak_params.add('A',    value  =  dataY[pointnum//2])
        peak_params.add('w',    value  =  np.abs(dataX[2*pointnum//3]-dataX[pointnum//3]))
        peak_params.add('x0',   value  =  dataX[pointnum//2])
        peak_params.add('bg',   value  =  (dataY[0]+dataY[pointnum-1])//2)
        """
        max_idx = np.argmax(dataY)
        peak_params.add('A',  value = dataY[max_idx])  # peak height
        peak_params.add('w',  value = 4.0*np.abs(dataX[pointnum-1] - dataX[0])/pointnum)  # width set 4 times of stepsize
        peak_params.add('x0', value = dataX[max_idx])  # peak center
        peak_params.add('bg', value = (dataY[0] + dataY[pointnum-1]) / 2)  # background, keep as before
        #print(peak_params)
    else: 
        peak_params.add('A',    value  =  initial[0])
        peak_params.add('w',    value  =  initial[1])
        peak_params.add('x0',   value  =  initial[2])
        peak_params.add('bg',   value  =  initial[3])

    #print(peak_params)

    if func   == "G":
        out   = minimize(gaussian_residual, peak_params, args=(dataX,),   kws={'data': dataY})
    elif func == "L":
        out   = minimize(lorentzian_residual, peak_params, args=(dataX,), kws={'data': dataY})
    else:
        print("Choose wrong function!")
        return
        
    #print(fit_report(out))

    xrange      =  np.linspace(dataX[0], dataX[-1], 101)
    fitted_data =  pd.merge(fitted_data, pd.DataFrame(xrange, columns=['X']), how='outer', left_index=True, right_index=True,suffixes=('', ''))

    if   func == "G":
        y_fitted    =  gaussian_residual(out.params, xrange)
    elif func == "L":
        y_fitted    =  lorentzian_residual(out.params, xrange)
    else:
        print("Choose wrong function!")
        return
    
    fitted_data =  pd.merge(fitted_data, pd.DataFrame(y_fitted, columns=['Y_fit']), how='outer', left_index=True, right_index=True,suffixes=('', ''))
    
    new_row ={}
    for name, param in out.params.items():
        #print('{}: {} +/- {}'.format(name, param.value, param.stderr))
        new_row[name]        =  [param.value]
        new_row[name+"_err"] =  [param.stderr]

    #fitted_params = fitted_params.append(new_row, ignore_index=True)
    newpd_row =    pd.DataFrame.from_dict(new_row,orient='columns')
    #print(fitted_params)
    newpd_row = newpd_row.astype(fitted_params.dtypes.to_dict())

    pd.set_option('display.expand_frame_repr', False)
    fitted_params =pd.concat([fitted_params, newpd_row],  join='outer')
    #print(fitted_params)
    #print(fitted_params)
    #print(fitted_data)
        
    return fitted_params, fitted_data




"""

def SqwDemo(H, K, L, W, p):
    #Example Scattering function for convolution tests
    
    #print(isinstance(W,np.ndarray))
    #print(np.shape(W))
    en1     =    p[0]   #peak 1
    en2     =    p[1]   #peak 2
    
    
    Gamma1 =     p[2]
    Gamma2 =     p[3]  #this is the reason, gamma2 was not used. 
    IntRatio=    p[4]

    temp   =     p[7]
    
    A1=1/((W-en1)**2+Gamma1**2)
    B1=1/((W+en1)**2+Gamma1**2)
    
    A2=1/((W-en2)**2+Gamma2**2)
    B2=1/((W+en2)**2+Gamma2**2)
    
    BF=1+1/(1-np.exp(W/temp/0.08713))

    
    sqw0=Gamma1*BF*(A1-B1) + IntRatio*Gamma2*BF*(A2-B2)
    sqw1=Gamma1*BF*(A1-B1) + IntRatio*Gamma2*BF*(A2-B2)

    
    sqw = np.vstack((sqw0, sqw1, sqw1))

    return sqw
"""

def angle2(x, y, z, h, k, l, lattice):
    r"""Function necessary for Prefactor functions
    """
    latticestar = npy.tools._star(lattice)[-1]
    #print(latticestar)
    #print( 2 * np.pi * (h * x + k * y + l * z) / npy.tools._modvec([x, y, z], lattice) / npy.tools._modvec([h, k, l], latticestar))

    return np.arccos( 2 * np.pi * (h * x + k * y + l * z) / npy.tools._modvec([x, y, z], lattice) / npy.tools._modvec([h, k, l], latticestar))

def SqwDemo(H, K, L, W, p):
    #Example Scattering function for convolution tests
    
    en1     =    p[0]   #peak 1
    en2     =    p[1]   #peak 2
    IntRatio=    p[2]
    Gamma1 =     p[3]
    Gamma2 =     p[4]  #this is the reason, gamma2 was not used. 
    #Int   =     p[5]
    #Bg    =     p[6]
    temp   =     p[7]
    
    A1=1/((W-en1)**2+Gamma1**2)
    B1=1/((W+en1)**2+Gamma1**2)
    
    A2=1/((W-en2)**2+Gamma2**2)
    B2=1/((W+en2)**2+Gamma2**2)
    
    BF=1+1/(1-np.exp(-W/temp/0.08713))
    
    sqw0=Gamma1*BF*(A1-B1) + IntRatio*Gamma2*BF*(A2-B2)
    sqw1=Gamma1*BF*(A1-B1) + IntRatio*Gamma2*BF*(A2-B2)

    sqw = np.vstack((sqw0, sqw1, sqw1))

    return sqw

def AsymSqwDemo(H,K,L,W,p):
    #J. Phys.: Condens. Matter 9 (1997) 1599–1608. Printed in the UK PII: S0953-8984(97)73830-X
    #An inelastic neutron scattering study of the Kondo semiconductor CeNiSn in high magnetic field
    #S Raymondy, L P Regnaulty, T Satoz, H Kadowakiz, N Pykax,
    #G Nakamotok, T Takabatakek, H Fujiik, Y Isikawa{, G Lapertoty and J Flouquet

    Delta1     =    p[0]   #peak 1
    Delta2     =    p[1]   #peak 2
    ratio      =    p[2]
    Gamma1     =    p[3]
    Gamma2     =    p[4]  #this is the reason, gamma2 was not used. 
    #Int   =     p[5]
    #Bg    =     p[6]
    temp       =    p[7]
    #print(p)

    B1 = 1/np.sqrt(W**2-(complex(Delta1, Gamma1))**2)

    C1 = np.abs(complex(Delta1, Gamma1))

    D1 = C1*W*B1.real*B1.real

    BF1= 1+1/(1-np.exp(-W/temp/0.08713))

    B2 = 1/np.sqrt(W**2-(complex(Delta2, Gamma2))**2)

    C2 = np.abs(complex(Delta2, Gamma2))

    D2 = C2*W*B2.real*B2.real

    BF2= 1+1/(1-np.exp(-W/temp/0.08713))
    #print(ratio)

    sqw0 = BF1*D1  + ratio*BF2*D2

    sqw = np.vstack((sqw0, sqw0, sqw0))

    return sqw

def AsymSqwDemoB(H,K,L,W,p):
    #J. Phys.: Condens. Matter 9 (1997) 1599–1608. Printed in the UK PII: S0953-8984(97)73830-X
    #An inelastic neutron scattering study of the Kondo semiconductor CeNiSn in high magnetic field
    #S Raymondy, L P Regnaulty, T Satoz, H Kadowakiz, N Pykax,
    #G Nakamotok, T Takabatakek, H Fujiik, Y Isikawa{, G Lapertoty and J Flouquet

    Delta1     =    p[0]   #peak 1
    Delta2     =    p[1]   #peak 2
    Gamma1     =    p[2]
    Gamma2     =    p[3]   #this is the reason, gamma2 was not used. 
    A1         =    p[4]   #Intensity
    A2        =     p[5]   #Intensity
    Bg         =    p[6]
    temp       =    p[7]

    B1 = 1/np.sqrt(W**2-(complex(Delta1, Gamma1))**2)

    C1 = np.abs(complex(Delta1, Gamma1))

    D1 = C1*W*B1.real*B1.real

    BF1= 1+1/(1-np.exp(-W/temp/0.08713))

    B2 = 1/np.sqrt(W**2-(complex(Delta2, Gamma2))**2)

    C2 = np.abs(complex(Delta2, Gamma2))

    D2 = C2*W*B2.real*B2.real

    BF2= 1+1/(1-np.exp(-W/temp/0.08713))

    sqw0 = BF1*D1*A1  + BF2*D2*A2 + Bg

    sqw = np.vstack((sqw0, sqw0, sqw0))

    return sqw

def AsymLorentzianSqw(H,K,L,W,p):
    """Model a decaying sine wave and subtract data."""

    Delta1     =    p[0]   #peak 1
    Delta2     =    p[1]   #peak 2
    ratio      =    p[2]
    Gamma1     =    p[3]
    Gamma2     =    p[4]  #this is the reason, gamma2 was not used. 
    #Int   =     p[5]
    #Bg    =     p[6]
    asym       =    p[7]
    
    #step=x[1]-x[0]
    #print(step)
    #x_range=np.linspace(x0-3*step,x0+3*step, 7 )  # x_range is W  
    #res_g  = 1/(res_w*np.sqrt(np.pi/(4*np.log(2)))) * np.exp(-4*np.log(2)*(W-x0)**2/res_w**2)
    #print('res_g:')
    #print(res_g)
    
    gg1  = 2*Gamma1/(1+np.exp(asym*(W-Delta1)))
    gg2  = 2*Gamma2/(1+np.exp(asym*(W-Delta2)))

    #print('gg')
    #print(gg)
    
    int1 =  gg1/(gg1**2+(W-Delta1)**2)-gg1/(gg1**2+(W+Delta1)**2)
    int2 =  gg2/(gg2**2+(W-Delta2)**2)-gg2/(gg2**2+(W+Delta2)**2)
    sqw0= int1 + ratio*int2
    #print('model:')
    #print(model)
    sqw = np.vstack((sqw0, sqw0, sqw0))
    return sqw



def PrefDemo(H, K, L, W, EXP, p):
    #Prefactor example for convolution tests
    
    [sample, rsample] = EXP.get_lattice()

    q2 = npy.tools._modvec([H, K, L], rsample) ** 2
    if len(p) > 8:
        AA =   p[8]
        aa =   p[9]
        BB =   p[10]
        bb =   p[11]
        CC =   p[12]
        cc =   p[13]
        DD =   p[14]

        # Now, use the Jane Brown approximate expression for Co2+
        sd = q2 / (16 * np.pi ** 2)
        ff = AA*np.exp(-aa*sd) + BB*np.exp(-bb*sd) + CC*np.exp(-cc*sd) + DD
    else:
        ff= np.ones_like(q2)

    alphax = angle2(1, 0, 0, H, K, L, sample)
    alphay = angle2(0, 1, 0, H, K, L, sample)
    alphaz = angle2(0, 0, 1, H, K, L, sample)

    polx = np.sin(alphax) ** 2
    poly = np.sin(alphay) ** 2
    polz = np.sin(alphaz) ** 2

    prefactor = np.zeros((3, len(H)))
    prefactor[0, :] = ff ** 2.0 * polx * p[5]
    prefactor[1, :] = ff ** 2.0 * poly * p[5]
    prefactor[2, :] = ff ** 2.0 * polz * p[5]

    bgr = np.ones(H.shape) * p[6]

    return [prefactor, bgr]

def PrefPhononDemo(H, K, L, W, EXP, p):
    #Prefactor example for convolution tests
    
    [sample, rsample] = EXP.get_lattice()

    alphax = angle2(1, 0, 0, H, K, L, sample)
    alphay = angle2(0, 1, 0, H, K, L, sample)
    alphaz = angle2(0, 0, 1, H, K, L, sample)

    polx = np.sin(alphax) ** 2
    poly = np.sin(alphay) ** 2
    polz = np.sin(alphaz) ** 2

    prefactor = np.zeros((3, len(H)))
    prefactor[0, :] = polx * p[5]
    prefactor[1, :] = poly * p[5]
    prefactor[2, :] = polz * p[5]

    bgr = np.ones(H.shape) * p[6]

    return [prefactor, bgr]

def SqwDemoQscan(H, K, L, W, p):
    # for the q scan

    
    # Extract the three parameters contained in "p":
    q1    =   p[0]                    # peak1 q position
    q2    =   p[1]                    # peak1 q position
    ratio =   p[2]                    # Intensity ratio 
    w1    =   p[3] 
    w2    =   p[4]                    # peak width

    A1=np.zeros(W.shape)
    
    A2=np.zeros(W.shape)

    if np.abs(H[0]-H[-1])/len(H) > 0.0005:
        A1=1/(w1*np.sqrt(np.pi/(4*np.log(2)))) * np.exp(-4*np.log(2)*(H-q1)**2/w1**2)
        A2=1/(w2*np.sqrt(np.pi/(4*np.log(2)))) * np.exp(-4*np.log(2)*(H-q2)**2/w2**2)
        #print(A1)     
        #print("this is executed")
    if np.abs(K[0]-K[-1])/len(K) > 0.0005:
        A1=1/(w1*np.sqrt(np.pi/(4*np.log(2)))) * np.exp(-4*np.log(2)*(K-q1)**2/w1**2)
        A2=1/(w2*np.sqrt(np.pi/(4*np.log(2)))) * np.exp(-4*np.log(2)*(K-q2)**2/w2**2)
        #print("K")     
    if np.abs(L[0]-L[-1])/len(L) > 0.0005:
        A1=1/(w1*np.sqrt(np.pi/(4*np.log(2)))) * np.exp(-4*np.log(2)*(L-q1)**2/w1**2)
        A2=1/(w2*np.sqrt(np.pi/(4*np.log(2)))) * np.exp(-4*np.log(2)*(L-q2)**2/w2**2)   
        #print("L")             

    sqw0 = A1 + ratio*A2
    #print(sqw0)
    sqw1 = np.zeros(sqw0.shape)

    sqw = np.vstack((sqw0, sqw1, sqw1))

    return sqw


def PrefDemoQscan(H, K, L, W, EXP, p):
    #Prefactor example for convolution tests
    
    [sample, rsample] = EXP.get_lattice()

    q_2 = npy.tools._modvec([H, K, L], rsample) ** 2

    intensity = p[5]
    bgr       = p[6]
    #temp = p[7]
    #Mn2+
    AA =   p[8]
    aa =   p[9]
    BB =   p[10]
    bb =   p[11]
    CC =   p[12]
    cc =   p[13]
    DD =   p[14]

    # Now, use the Jane Brown approximate expression for Co2+
    sd = q_2 / (16 * np.pi ** 2)
    ff = AA*np.exp(-aa*sd) + BB*np.exp(-bb*sd) + CC*np.exp(-cc*sd) + DD

    alphax = angle2(1, 0, 0, H, K, L, sample)
    alphay = angle2(0, 1, 0, H, K, L, sample)
    alphaz = angle2(0, 0, 1, H, K, L, sample)

    polx = np.sin(alphax) ** 2
    poly = np.sin(alphay) ** 2
    polz = np.sin(alphaz) ** 2

    prefactor = np.zeros((3, len(H)))
    prefactor[0, :] = ff ** 2.0 * polx * intensity
    prefactor[1, :] = 0 
    prefactor[2, :] = 0 

    bgr = np.ones(H.shape) * bgr

    return [prefactor, bgr]

        
       
def ConvResidualDemo(parameters, hklw, exp):

    p1      =  float(parameters['p1'])
    p2      =  float(parameters['p2'])
    w1      =  float(parameters['w1'])
    w2      =  float(parameters['w2'])
    ratio   =  float(parameters['ratio'])
    height  =  float(parameters['height'])
    bg      =  float(parameters['bg'])
    temp    =  float(parameters['temp'])

    parlist =  [p1, p2,  w1, w2, ratio, height, bg, temp]
    #print("+++++++++")
    #print(hklw[0:4,:])

    conv    =  exp.ResConv(SqwDemo, PrefDemo, nargout=2, hkle=hklw[0:4,:], METHOD='fix', ACCURACY=[5,5], p=parlist)

    return conv-hklw[5,:]

def save_fit_params(scanno, hklw, finalpar, outfile="fittedparams.txt"):
    
    with open(outfile, 'a+') as f:
        f.write("Fitting to scan #{}_at_({}, {}, {})\n".format(scanno, hklw['h'][0], hklw['k'][0], hklw['l'][0]))
        f.write("    e1      e1err      e2      e2err      ratio      rtoerr      w1      w1err      w2      w2err      intensity      interr      bg      bgerr\n")
        f.write("{:6.4f}    {:6.4f}    {:6.4f}    {:6.4f}    {:6.4f}    {:6.4f}    {:6.4f}    {:6.4f}    {:6.4f}    {:6.4f}    {:6.4f}    {:6.4f}    {:6.4f}    {:6.4f}\n".format(finalpar['e1'][0],finalpar['e1err'][0],finalpar['e2'][0],
             finalpar['e2err'][0],finalpar['ratio'][0], finalpar['rtoerr'][0], finalpar['w1'][0], finalpar['w1err'][0],finalpar['w2'][0],finalpar['w2err'][0],finalpar['int'][0],finalpar['interr'][0],finalpar['bg'][0],finalpar['bgerr'][0]))
        f.write("\n")

    print("The parameters are written in the file {}!".format(outfile))
    return

def SelFormFactor(magion=''):
    formfactList = [
        {"magion": "Sc0", "AA": 0.251200, "aa": 90.029600, "BB": 0.329000, "bb": 39.402100, "CC": 0.423500, "cc": 14.322200, "DD": -0.004300 },
        {"magion": "Sc1", "AA": 0.488900, "aa": 51.160300, "BB": 0.520300, "bb": 14.076400, "CC": -0.028600, "cc": 0.179200, "DD": 0.018500 },
        {"magion": "Sc2", "AA": 0.504800, "aa": 31.403500, "BB": 0.518600, "bb": 10.989700, "CC": -0.024100, "cc": 1.183100, "DD": 0.000000 },
        {"magion": "Ti0", "AA": 0.465700, "aa": 33.589800, "BB": 0.549000, "bb": 9.879100, "CC": -0.029100, "cc": 0.323200, "DD": 0.012300 },
        {"magion": "Ti1", "AA": 0.509300, "aa": 36.703300, "BB": 0.503200, "bb": 10.371300, "CC": -0.026300, "cc": 0.310600, "DD": 0.011600 },
        {"magion": "Ti2", "AA": 0.509100, "aa": 24.976300, "BB": 0.516200, "bb": 8.756900, "CC": -0.028100, "cc": 0.916000, "DD": 0.001500 },
        {"magion": "Ti3", "AA": 0.357100, "aa": 22.841300, "BB": 0.668800, "bb": 8.930600, "CC": -0.035400, "cc": 0.483300, "DD": 0.009900 },
        {"magion": "V0", "AA": 0.408600, "aa": 28.810900, "BB": 0.607700, "bb": 8.543700, "CC": -0.029500, "cc": 0.276800, "DD": 0.012300 },
        {"magion": "V1", "AA": 0.444400, "aa": 32.647900, "BB": 0.568300, "bb": 9.097100, "CC": -0.228500, "cc": 0.021800, "DD": 0.215000 },
        {"magion": "V2", "AA": 0.408500, "aa": 23.852600, "BB": 0.609100, "bb": 8.245600, "CC": -0.167600, "cc": 0.041500, "DD": 0.149600 },
        {"magion": "V3", "AA": 0.359800, "aa": 19.336400, "BB": 0.663200, "bb": 7.617200, "CC": -0.306400, "cc": 0.029600, "DD": 0.283500 },
        {"magion": "V4", "AA": 0.310600, "aa": 16.816000, "BB": 0.719800, "bb": 7.048700, "CC": -0.052100, "cc": 0.302000, "DD": 0.022100 },
        {"magion": "Cr0", "AA": 0.113500, "aa": 45.199000, "BB": 0.348100, "bb": 19.493100, "CC": 0.547700, "cc": 7.354200, "DD": -0.009200 },
        {"magion": "Cr1", "AA": -0.097700, "aa": 0.047000, "BB": 0.454400, "bb": 26.005400, "CC": 0.557900, "cc": 7.489200, "DD": 0.083100 },
        {"magion": "Cr2", "AA": 1.202400, "aa": -0.005500, "BB": 0.415800, "bb": 20.547500, "CC": 0.603200, "cc": 6.956000, "DD": -1.221800 },
        {"magion": "Cr3", "AA": -0.309400, "aa": 0.027400, "BB": 0.368000, "bb": 17.035500, "CC": 0.655900, "cc": 6.523600, "DD": 0.285600 },
        {"magion": "Cr4", "AA": -0.232000, "aa": 0.043300, "BB": 0.310100, "bb": 14.951800, "CC": 0.718200, "cc": 6.172600, "DD": 0.204200 },
        {"magion": "Mn0", "AA": 0.243800, "aa": 24.962900, "BB": 0.147200, "bb": 15.672800, "CC": 0.618900, "cc": 6.540300, "DD": -0.010500 },
        {"magion": "Mn1", "AA": -0.013800, "aa": 0.421300, "BB": 0.423100, "bb": 24.668000, "CC": 0.590500, "cc": 6.654500, "DD": -0.001000 },
        {"magion": "Mn2", "AA": 0.422000, "aa": 17.684000, "BB": 0.594800, "bb": 6.005000, "CC": 0.004300, "cc": -0.609000, "DD": -0.021900 },
        {"magion": "Mn3", "AA": 0.419800, "aa": 14.282900, "BB": 0.605400, "bb": 5.468900, "CC": 0.924100, "cc": -0.008800, "DD": -0.949800 },
        {"magion": "Mn4", "AA": 0.376000, "aa": 12.566100, "BB": 0.660200, "bb": 5.132900, "CC": -0.037200, "cc": 0.563000, "DD": 0.001100 },
        {"magion": "Mn5", "AA": 0.292400, "aa": 11.665500, "BB": 0.740500, "bb": 5.074100, "CC": -1.788300, "cc": 0.005900, "DD": 1.755700 },
        {"magion": "Fe0", "AA": 0.070600, "aa": 35.008500, "BB": 0.358900, "bb": 15.358300, "CC": 0.581900, "cc": 5.560600, "DD": -0.011400 },
        {"magion": "Fe1", "AA": 0.125100, "aa": 34.963300, "BB": 0.362900, "bb": 15.514400, "CC": 0.522300, "cc": 5.591400, "DD": -0.010500 },
        {"magion": "Fe2", "AA": 0.026300, "aa": 34.959700, "BB": 0.366800, "bb": 15.943500, "CC": 0.618800, "cc": 5.593500, "DD": -0.011900 },
        {"magion": "Fe3", "AA": 0.397200, "aa": 13.244200, "BB": 0.629500, "bb": 4.903400, "CC": -0.031400, "cc": 0.349600, "DD": 0.004400 },
        {"magion": "Fe4", "AA": 0.378200, "aa": 11.380000, "BB": 0.655600, "bb": 4.592000, "CC": -0.034600, "cc": 0.483300, "DD": 0.000500 },
        {"magion": "Co0", "AA": 0.413900, "aa": 16.161600, "BB": 0.601300, "bb": 4.780500, "CC": -0.151800, "cc": 0.021000, "DD": 0.134500 },
        {"magion": "Co1", "AA": 0.099000, "aa": 33.125200, "BB": 0.364500, "bb": 15.176800, "CC": 0.547000, "cc": 5.008100, "DD": -0.010900 },
        {"magion": "Co2", "AA": 0.433200, "aa": 14.355300, "BB": 0.585700, "bb": 4.607700, "CC": -0.038200, "cc": 0.133800, "DD": 0.017900 },
        {"magion": "Co3", "AA": 0.390200, "aa": 12.507800, "BB": 0.632400, "bb": 4.457400, "CC": -0.150000, "cc": 0.034300, "DD": 0.127200 },
        {"magion": "Co4", "AA": 0.351500, "aa": 10.778500, "BB": 0.677800, "bb": 4.234300, "CC": -0.038900, "cc": 0.240900, "DD": 0.009800 },
        {"magion": "Ni0", "AA": -0.017200, "aa": 35.739200, "BB": 0.317400, "bb": 14.268900, "CC": 0.713600, "cc": 4.566100, "DD": -0.014300 },
        {"magion": "Ni1", "AA": 0.070500, "aa": 35.856100, "BB": 0.398400, "bb": 13.804200, "CC": 0.542700, "cc": 4.396500, "DD": -0.011800 },
        {"magion": "Ni2", "AA": 0.016300, "aa": 35.882600, "BB": 0.391600, "bb": 13.223300, "CC": 0.605200, "cc": 4.338800, "DD": -0.013300 },
        {"magion": "Ni3", "AA": 0.001200, "aa": 34.999800, "BB": 0.346800, "bb": 11.987400, "CC": 0.666700, "cc": 4.251800, "DD": -0.014800 },
        {"magion": "Ni4", "AA": -0.009000, "aa": 35.861400, "BB": 0.277600, "bb": 11.790400, "CC": 0.747400, "cc": 4.201100, "DD": -0.016300 },
        {"magion": "Cu0", "AA": 0.090900, "aa": 34.983800, "BB": 0.408800, "bb": 11.443200, "CC": 0.512800, "cc": 3.824800, "DD": -0.012400 },
        {"magion": "Cu1", "AA": 0.074900, "aa": 34.965600, "BB": 0.414700, "bb": 11.764200, "CC": 0.523800, "cc": 3.849700, "DD": -0.012700 },
        {"magion": "Cu2", "AA": 0.023200, "aa": 34.968600, "BB": 0.402300, "bb": 11.564000, "CC": 0.588200, "cc": 3.842800, "DD": -0.013700 },
        {"magion": "Cu3", "AA": 0.003100, "aa": 34.907400, "BB": 0.358200, "bb": 10.913800, "CC": 0.653100, "cc": 3.827900, "DD": -0.014700 },
        {"magion": "Cu4", "AA": -0.013200, "aa": 30.681700, "BB": 0.280100, "bb": 11.162600, "CC": 0.749000, "cc": 3.817200, "DD": -0.016500 },
        {"magion": "Y0", "AA": 0.591500, "aa": 67.608100, "BB": 1.512300, "bb": 17.900400, "CC": -1.113000, "cc": 14.135900, "DD": 0.008000 },
        {"magion": "Zr0", "AA": 0.410600, "aa": 59.996100, "BB": 1.054300, "bb": 18.647600, "CC": -0.475100, "cc": 10.540000, "DD": 0.010600 },
        {"magion": "Zr1", "AA": 0.453200, "aa": 59.594800, "BB": 0.783400, "bb": 21.435700, "CC": -0.245100, "cc": 9.036000, "DD": 0.009800 },
        {"magion": "Nb0", "AA": 0.394600, "aa": 49.229700, "BB": 1.319700, "bb": 14.821600, "CC": -0.726900, "cc": 9.615600, "DD": 0.012900 },
        {"magion": "Nb1", "AA": 0.457200, "aa": 49.918200, "BB": 1.027400, "bb": 15.725600, "CC": -0.496200, "cc": 9.157300, "DD": 0.011800 },
        {"magion": "Mo0", "AA": 0.180600, "aa": 49.056800, "BB": 1.230600, "bb": 14.785900, "CC": -0.426800, "cc": 6.986600, "DD": 0.017100 },
        {"magion": "Mo1", "AA": 0.350000, "aa": 48.035400, "BB": 1.030500, "bb": 15.060400, "CC": -0.392900, "cc": 7.479000, "DD": 0.013900 },
        {"magion": "Tc0", "AA": 0.129800, "aa": 49.661100, "BB": 1.165600, "bb": 14.130700, "CC": -0.313400, "cc": 5.512900, "DD": 0.019500 },
        {"magion": "Tc1", "AA": 0.267400, "aa": 48.956600, "BB": 0.956900, "bb": 15.141300, "CC": -0.238700, "cc": 5.457800, "DD": 0.016000 },
        {"magion": "Ru0", "AA": 0.106900, "aa": 49.423800, "BB": 1.191200, "bb": 12.741700, "CC": -0.317600, "cc": 4.912500, "DD": 0.021300 },
        {"magion": "Ru1", "AA": 0.441000, "aa": 33.308600, "BB": 1.477500, "bb": 9.553100, "CC": -0.936100, "cc": 6.722000, "DD": 0.017600 },
        {"magion": "Rh0", "AA": 0.097600, "aa": 49.882500, "BB": 1.160100, "bb": 11.830700, "CC": -0.278900, "cc": 4.126600, "DD": 0.023400 },
        {"magion": "Rh1", "AA": 0.334200, "aa": 29.756400, "BB": 1.220900, "bb": 9.438400, "CC": -0.575500, "cc": 5.332000, "DD": 0.021000 },
        {"magion": "Pd0", "AA": 0.200300, "aa": 29.363300, "BB": 1.144600, "bb": 9.599300, "CC": -0.368900, "cc": 4.042300, "DD": 0.025100 },
        {"magion": "Pd1", "AA": 0.503300, "aa": 24.503700, "BB": 1.998200, "bb": 6.908200, "CC": -1.524000, "cc": 5.513300, "DD": 0.021300 },
        {"magion": "Ce2", "AA": 0.295300, "aa": 17.684600, "BB": 0.292300, "bb": 6.732900, "CC": 0.431300, "cc": 5.382700, "DD": -0.019400 },
        {"magion": "Nd2", "AA": 0.164500, "aa": 25.045300, "BB": 0.252200, "bb": 11.978200, "CC": 0.601200, "cc": 4.946100, "DD": -0.018000 },
        {"magion": "Nd3", "AA": 0.054000, "aa": 25.029300, "BB": 0.310100, "bb": 12.102000, "CC": 0.657500, "cc": 4.722300, "DD": -0.021600 },
        {"magion": "Sm2", "AA": 0.090900, "aa": 25.203200, "BB": 0.303700, "bb": 11.856200, "CC": 0.625000, "cc": 4.236600, "DD": -0.020000 },
        {"magion": "Sm3", "AA": 0.028800, "aa": 25.206800, "BB": 0.297300, "bb": 11.831100, "CC": 0.695400, "cc": 4.211700, "DD": -0.021300 },
        {"magion": "Eu2", "AA": 0.075500, "aa": 25.296000, "BB": 0.300100, "bb": 11.599300, "CC": 0.643800, "cc": 4.025200, "DD": -0.019600 },
        {"magion": "Eu3", "AA": 0.020400, "aa": 25.307800, "BB": 0.301000, "bb": 11.474400, "CC": 0.700500, "cc": 3.942000, "DD": -0.022000 },
        {"magion": "Gd2", "AA": 0.063600, "aa": 25.382300, "BB": 0.303300, "bb": 11.212500, "CC": 0.652800, "cc": 3.787700, "DD": -0.019900 },
        {"magion": "Gd3", "AA": 0.018600, "aa": 25.386700, "BB": 0.289500, "bb": 11.142100, "CC": 0.713500, "cc": 3.752000, "DD": -0.021700 },
        {"magion": "Tb2", "AA": 0.054700, "aa": 25.508600, "BB": 0.317100, "bb": 10.591100, "CC": 0.649000, "cc": 3.517100, "DD": -0.021200 },
        {"magion": "Tb3", "AA": 0.017700, "aa": 25.509500, "BB": 0.292100, "bb": 10.576900, "CC": 0.713300, "cc": 3.512200, "DD": -0.023100 },
        {"magion": "Dy2", "AA": 0.130800, "aa": 18.315500, "BB": 0.311800, "bb": 7.664500, "CC": 0.579500, "cc": 3.146900, "DD": -0.022600 },
        {"magion": "Dy3", "AA": 0.115700, "aa": 15.073200, "BB": 0.327000, "bb": 6.799100, "CC": 0.582100, "cc": 3.020200, "DD": -0.024900 },
        {"magion": "Ho2", "AA": 0.099500, "aa": 18.176100, "BB": 0.330500, "bb": 7.855600, "CC": 0.592100, "cc": 2.979900, "DD": -0.023000 },
        {"magion": "Ho3", "AA": 0.056600, "aa": 18.317600, "BB": 0.336500, "bb": 7.688000, "CC": 0.631700, "cc": 2.942700, "DD": -0.024800 },
        {"magion": "Er2", "AA": 0.112200, "aa": 18.122300, "BB": 0.346200, "bb": 6.910600, "CC": 0.564900, "cc": 2.761400, "DD": -0.023500 },
        {"magion": "Er3", "AA": 0.058600, "aa": 17.980200, "BB": 0.354000, "bb": 7.096400, "CC": 0.612600, "cc": 2.748200, "DD": -0.025100 },
        {"magion": "Tm2", "AA": 0.098300, "aa": 18.323600, "BB": 0.338000, "bb": 6.917800, "CC": 0.587500, "cc": 2.662200, "DD": -0.024100 },
        {"magion": "Tm3", "AA": 0.058100, "aa": 15.092200, "BB": 0.278700, "bb": 7.801500, "CC": 0.685400, "cc": 2.793100, "DD": -0.022400 },
        {"magion": "Yb2", "AA": 0.085500, "aa": 18.512300, "BB": 0.294300, "bb": 7.373400, "CC": 0.641200, "cc": 2.677700, "DD": -0.021300 },
        {"magion": "Yb3", "AA": 0.041600, "aa": 16.094900, "BB": 0.284900, "bb": 7.834100, "CC": 0.696100, "cc": 2.672500, "DD": -0.022900 },
        {"magion": "Pr3", "AA": 0.050400, "aa": 24.998900, "BB": 0.257200, "bb": 12.037700, "CC": 0.714200, "cc": 5.003900, "DD": -0.021900 },
        {"magion": "U3", "AA": 0.505800, "aa": 23.288200, "BB": 1.346400, "bb": 7.002800, "CC": -0.872400, "cc": 4.868300, "DD": 0.019200 },
        {"magion": "U4", "AA": 0.329100, "aa": 23.547500, "BB": 1.083600, "bb": 8.454000, "CC": -0.434000, "cc": 4.119600, "DD": 0.021400 },
        {"magion": "U5", "AA": 0.365000, "aa": 19.803800, "BB": 3.219900, "bb": 6.281800, "CC": -2.607700, "cc": 5.301000, "DD": 0.023300 },
        {"magion": "Np3", "AA": 0.515700, "aa": 20.865400, "BB": 2.278400, "bb": 5.893000, "CC": -1.816300, "cc": 4.845700, "DD": 0.021100 },
        {"magion": "Np4", "AA": 0.420600, "aa": 19.804600, "BB": 2.800400, "bb": 5.978300, "CC": -2.243600, "cc": 4.984800, "DD": 0.022800 },
        {"magion": "Np5", "AA": 0.369200, "aa": 18.190000, "BB": 3.151000, "bb": 5.850000, "CC": -2.544600, "cc": 4.916400, "DD": 0.024800 },
        {"magion": "Np6", "AA": 0.292900, "aa": 17.561100, "BB": 3.486600, "bb": 5.784700, "CC": -2.806600, "cc": 4.870700, "DD": 0.026700 },
        {"magion": "Pu3", "AA": 0.384000, "aa": 16.679300, "BB": 3.104900, "bb": 5.421000, "CC": -2.514800, "cc": 4.551200, "DD": 0.026300 },
        {"magion": "Pu4", "AA": 0.493400, "aa": 16.835500, "BB": 1.639400, "bb": 5.638400, "CC": -1.158100, "cc": 4.139900, "DD": 0.024800 },
        {"magion": "Pu5", "AA": 0.388800, "aa": 16.559200, "BB": 2.036200, "bb": 5.656700, "CC": -1.451500, "cc": 4.255200, "DD": 0.026700 },
        {"magion": "Pu6", "AA": 0.317200, "aa": 16.050700, "BB": 3.465400, "bb": 5.350700, "CC": -2.810200, "cc": 4.513300, "DD": 0.028100 },
        {"magion": "Am2", "AA": 0.474300, "aa": 21.776100, "BB": 1.580000, "bb": 5.690200, "CC": -1.077900, "cc": 4.145100, "DD": 0.021800 },
        {"magion": "Am3", "AA": 0.423900, "aa": 19.573900, "BB": 1.457300, "bb": 5.872200, "CC": -0.905200, "cc": 3.968200, "DD": 0.023800 },
        {"magion": "Am4", "AA": 0.373700, "aa": 17.862500, "BB": 1.352100, "bb": 6.042600, "CC": -0.751400, "cc": 3.719900, "DD": 0.025800 },
        {"magion": "Am5", "AA": 0.295600, "aa": 17.372500, "BB": 1.452500, "bb": 6.073400, "CC": -0.775500, "cc": 3.661900, "DD": 0.027700 },
        {"magion": "Am6", "AA": 0.230200, "aa": 16.953300, "BB": 1.486400, "bb": 6.115900, "CC": -0.745700, "cc": 3.542600, "DD": 0.029400 },
        {"magion": "Am7", "AA": 0.360100, "aa": 12.729900, "BB": 1.964000, "bb": 5.120300, "CC": -1.356000, "cc": 3.714200, "DD": 0.031600 }
    ]
    
    for item in formfactList:
        if magion.upper() == item["magion"].upper():
            print(f"The form factor of {magion} is found.")
            return item
    print(f"The mangetic ion {magion} is not found.")
    return None





def curie_weiss_law(T, par_C, par_Tc):
    inv_kai= T/par_C - par_Tc/par_C
    return inv_kai


    
def residual_curie_weiss(pars, x, data=None):
    """Model """
    vals = pars.valuesdict()
    par_C   = vals['C']
    par_Tc  = vals['Tc']
    
    model = x/par_C - par_Tc/par_C
    if data is None:
        return model
    return model - data









def create_html_plots(dataframes_config, output_file="output_dashboard.html", cols=4):
    """
    Create an HTML file with plotly visualizations for multiple dataframes
    
    Parameters:
    dataframes_config (list): List of dictionaries with format:
                             {'filename': filename, 
                              'df': dataframe, 
                              'x_col': x_axis_column,
                              'y_cols': [list_of_y_columns],
                              'plot_type': 'line'/'bar'/'scatter'/etc.}
    output_file (str): Path to save the HTML output
    cols (int): Number of columns in the layout
    """
    # Sort dataframes by modification time (most recent first)
    for config in dataframes_config:
        try:
            config['mod_time'] = os.path.getmtime(config['filename'])
        except:
            # If filename isn't a real file path, use current time
            config['mod_time'] = datetime.now().timestamp()
    
    # Sort by modification time (most recent first)
    dataframes_config.sort(key=lambda x: x['mod_time'], reverse=True)
    
    # Create HTML file
    with open(output_file, 'w') as f:
        # HTML header
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Visualization Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }
                .grid-container {
                    display: grid;
                    grid-template-columns: repeat(""" + str(cols) + """, 1fr);
                    gap: 20px;
                    padding: 20px;
                }
                .plot-container {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                h1 {
                    text-align: center;
                }
                h3 {
                    margin-top: 0;
                    color: #333;
                }
                @media (max-width: 1200px) {
                    .grid-container {
                        grid-template-columns: repeat(3, 1fr);
                    }
                }
                @media (max-width: 900px) {
                    .grid-container {
                        grid-template-columns: repeat(2, 1fr);
                    }
                }
                @media (max-width: 600px) {
                    .grid-container {
                        grid-template-columns: 1fr;
                    }
                }
            </style>
        </head>
        <body>
            <h1>Data Visualization Dashboard</h1>
            <div class="grid-container">
        """)
        
        # Add each plot
        for config in dataframes_config:
            plot_div = generate_plotly(
                config['df'], 
                os.path.basename(config['filename']),
                config.get('x_col'),
                config.get('y_cols', []),
                config.get('plot_type', 'line')
            )
            
            # Write plot container
            f.write(f"""
            <div class="plot-container">
                <h3>{os.path.basename(config['filename'])}</h3>
                {plot_div}
            </div>
            """)
        
        # HTML footer
        f.write("""
            </div>
        </body>
        </html>
        """)
    
    print(f"Dashboard created successfully at {output_file}")

def generate_plotly(df, title, x_col=None, y_cols=None, plot_type='line'):
    """
    Generate a plotly visualization for a dataframe
    
    Parameters:
    df (DataFrame): The pandas DataFrame to plot
    title (str): Title for the plot
    x_col (str): Column name to use for x-axis
    y_cols (list): List of column names to use for y-axis
    plot_type (str): Type of plot ('line', 'bar', 'scatter', etc.)
    """
    fig = go.Figure()
    
    # Set default x and y columns if not specified
    if x_col is None:
        x_col = df.columns[0] if len(df.columns) > 0 else None
    
    if y_cols is None or len(y_cols) == 0:
        if len(df.columns) > 1:
            y_cols = df.columns[1:]
        else:
            y_cols = [df.columns[0]] if len(df.columns) > 0 else []
    
    # Create the plot based on specified type
    for y_col in y_cols:
        if plot_type == 'line':
            fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='lines+markers', name=y_col))
        elif plot_type == 'bar':
            fig.add_trace(go.Bar(x=df[x_col], y=df[y_col], name=y_col))
        elif plot_type == 'scatter':
            fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='markers', name=y_col))
        elif plot_type == 'area':
            fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], fill='tozeroy', name=y_col))
        elif plot_type == 'box':
            fig.add_trace(go.Box(y=df[y_col], name=y_col))
        elif plot_type == 'histogram':
            fig.add_trace(go.Histogram(x=df[y_col], name=y_col))
        else:  # Default to line plot
            fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='lines+markers', name=y_col))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title="Value",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Convert plot to HTML div
    plot_div = pyo.plot(fig, output_type='div', include_plotlyjs=False)
    
    return plot_div




# Create a contour plot using matplotlib
def matplotlib_contour(data, xtitle='x', ytitle= 'y', ztitle= 'counts', zminmax=[0, 1000], output_file=None):
    # Extract data
    x = data[xtitle].values
    y = data[ytitle].values
    z = data[ztitle].values
    
    # Create a regular grid for contour plot
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate z values on the regular grid
    #zi = griddata((x, y), z, (xi, yi), method='cubic')
    # Interpolate z values on the regular grid
    zi = griddata((x, y), z, (xi, yi), method='linear')  # Change to 'linear' to avoid extrapolation issues
    zi = np.clip(zi, zminmax[0], zminmax[1])  # Clip values to defined range
    zi = np.nan_to_num(zi, nan=zminmax[0])  # Replace NaN with min value
    
    # Create the contour plot
    fig, ax = plt.subplots(figsize=(10,8))
    contour = plt.contourf(xi, yi, zi, levels=15, vmin=zminmax[0], vmax=zminmax[1], cmap='viridis')
    plt.colorbar(contour, label='Counts')
    plt.scatter(x, y, s=1, color='black', alpha=0.5)  # Show original data points
    plt.title('Contour Map of Measurement Data')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True, linestyle='--', alpha=0.7)
    if output_file:
        plt.savefig('contour_plot_matplotlib.png', dpi=300, bbox_inches='tight')
    #plt.show()
    return fig, ax



def plotly_contour(data, xtitle='x', ytitle= 'y', ztitle= 'counts', zminmax=[0, 1000], output_file=None):
    # Extract data
    x = data[xtitle].values
    y = data[ytitle].values
    z = data[ztitle].values
    
    # Create a regular grid for contour plot
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate z values on the regular grid
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    
    # Create the contour plot
    fig = go.Figure(data=go.Contour(
        x=xi[0],
        y=yi[:,0],
        z=zi,
        colorscale='Viridis',
        contours=dict(
            showlabels=True,
            labelfont=dict(size=12, color='white')
        ),zmin=zminmax[0],zmax=zminmax[1], ncontours=20,
        colorbar=dict(title='Counts'),
        hoverinfo='x+y+z'
    ))
    
    # Add scatter points for original data
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=3,
            color='black',
            opacity=0.5
        ),
        name='Data Points',
        hoverinfo='x+y+text',
        hovertext=[f'Count: {count:.2f}' for count in z]
    ))
    
    # Update layout
    fig.update_layout(
        title='Contour Map of Measurement Data',
        xaxis_title='X Position',
        yaxis_title='Y Position',
        width=900,
        height=700
    )
    
    # Save to HTML file if specified
    if output_file:
        fig.write_html(output_file)
    
    # Show the figure
    #fig.show()
    
    return fig    


def customize_plot_axis(ax, title, xlabel="X-axis", ylabel="Y-axis"):
    # Title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Labels
    ax.set_xlabel(xlabel, fontsize=18, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=18, labelpad=10)
    
    # Tick labels
    ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6, pad=5)
    ax.tick_params(axis='both', which='minor', width=1.5, length=4)
    
    # Tick spacing
    #ax.xaxis.set_major_locator(MultipleLocator(2))
    #ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    #ax.yaxis.set_major_locator(MultipleLocator(0.5))
    #ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    
    # Frame appearance
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # Grid
    #ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
    
    # Legend
    ax.legend(fontsize=12, frameon=True, fancybox=True, framealpha=0.8, 
              edgecolor='black', borderpad=1, handlelength=3)
    ax.text(0.02, 0.98, '(a)', transform=ax.transAxes,
        fontsize=14,  va='top', ha='left') #fontweight='bold',